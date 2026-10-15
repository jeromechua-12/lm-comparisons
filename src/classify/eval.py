import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.classify.model import Classifier, extract
from src.classify.train import LMS, N_CLASSES, OUT, load_lm, transform
from src.data import load_ag_news
from src.lstm.train import get_device

REPS = ["lstm_mean", "transformer_mean", "lstm_mean_pca300"]
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
N_EXAMPLES = 5


def main():
    """Evaluate each best-epoch classifier on the AG News test set and collect LSTM vs Transformer errors."""
    dev = get_device()
    X_test, y_test = load_ag_news()["test"]
    y = np.array(y_test)
    n_toks = sum(len(seq) + 2 for seq in X_test)

    feats = {}
    extract_times = {}
    lm = None
    for name in LMS:
        del lm
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        lm, vocab = load_lm(name, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        feats[name] = extract(lm, X_test, vocab)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        extract_times[name] = time.perf_counter() - start
    unk = np.array([sum(tok not in vocab.stoi for tok in seq) / max(len(seq), 1) for seq in X_test])

    results = []
    preds = {}
    for rep in REPS:
        ckpt = torch.load(OUT / f"{rep}.pt", map_location=dev, weights_only=False)
        clf = Classifier(ckpt["in_dim"], N_CLASSES).to(dev)
        clf.load_state_dict(ckpt["state"])
        clf.eval()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            x = transform(feats[ckpt["lm"]], ckpt["scaler"], ckpt["pca"]).to(dev)
            pred = clf(x).argmax(1).cpu().numpy()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        infer_time = extract_times[ckpt["lm"]] + time.perf_counter() - start
        with open(Path("checkpoints") / ckpt["lm"] / f"{ckpt['lm']}_eval.json") as f:
            lm_ppl = json.load(f)["test_ppl"]
        wrong = pred != y
        preds[rep] = pred
        results.append({
            "rep": rep,
            "lm": ckpt["lm"],
            "pca_dim": ckpt["pca_dim"],
            "in_dim": ckpt["in_dim"],
            "best_epoch": ckpt["best_epoch"],
            "num_params": ckpt["num_params"],
            "lm_test_ppl": lm_ppl,
            "test_acc": accuracy_score(y, pred),
            "test_f1": f1_score(y, pred, average="macro"),
            "per_class_f1": dict(zip(CLASSES, f1_score(y, pred, average=None).tolist())),
            "confusion": confusion_matrix(y, pred).tolist(),
            "unk_rate_wrong": float(unk[wrong].mean()),
            "unk_rate_right": float(unk[~wrong].mean()),
            "infer_time_s": infer_time,
            "articles_per_s": len(X_test) / infer_time,
            "tokens_per_s": n_toks / infer_time,
        })

    lstm_p = preds["lstm_mean"]
    tf_p = preds["transformer_mean"]
    cases = {
        "lstm_wrong_transformer_right": (lstm_p != y) & (tf_p == y),
        "transformer_wrong_lstm_right": (tf_p != y) & (lstm_p == y),
        "both_wrong": (lstm_p != y) & (tf_p != y),
    }
    errors = {}
    for case, mask in cases.items():
        idx = np.flatnonzero(mask)
        examples = []
        for i in idx[:N_EXAMPLES]:
            examples.append({
                "text": " ".join(X_test[i]),
                "true": CLASSES[y[i]],
                "lstm": CLASSES[lstm_p[i]],
                "transformer": CLASSES[tf_p[i]],
                "unk_rate": float(unk[i]),
            })
        errors[case] = {"count": len(idx), "examples": examples}
    agree = float((lstm_p == tf_p).mean())

    print(f"AG News test articles={len(X_test)} tokens={n_toks:,}")
    print(f"{'rep':>18} {'lm_ppl':>7} {'dim':>4} {'test_acc':>8} {'test_f1':>8} {'unk_wrong':>9} {'unk_right':>9} {'infer_s':>8} {'art/s':>8}")
    for r in results:
        print(f"{r['rep']:>18} {r['lm_test_ppl']:>7.2f} {r['in_dim']:>4} {r['test_acc']:>8.4f} {r['test_f1']:>8.4f} {r['unk_rate_wrong']:>9.4f} {r['unk_rate_right']:>9.4f} {r['infer_time_s']:>8.2f} {r['articles_per_s']:>8,.0f}")
    print(f"{'per-class f1':>18} " + " ".join(f"{c:>9}" for c in CLASSES))
    for r in results:
        print(f"{r['rep']:>18} " + " ".join(f"{r['per_class_f1'][c]:>9.4f}" for c in CLASSES))
    for r in results:
        print(f"{r['rep']} confusion (rows true, cols pred: {', '.join(CLASSES)})")
        for c, row in zip(CLASSES, r["confusion"]):
            print(f"  {c:>9} " + " ".join(f"{v:>6}" for v in row))
    print(f"lstm_mean vs transformer_mean agreement={agree:.4f} " + " ".join(f"{k}={v['count']}" for k, v in errors.items()))

    res_path = OUT / "classify_eval.json"
    with open(res_path, "w") as f:
        json.dump({"device": str(dev), "test_articles": len(X_test), "test_tokens": n_toks, "results": results, "agreement": agree, "errors": errors}, f, indent=2)
    print(f"saved {res_path}")


if __name__ == "__main__":
    main()
