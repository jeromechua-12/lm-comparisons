import gc
import json
import time
from pathlib import Path

import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from src.classify.model import LM, Classifier, extract
from src.data import Sequences, load_ag_news
from src.lstm.model import LSTM
from src.lstm.train import get_device
from src.transformer.model import Transformer
from src.vocab import Vocab

LMS = {"lstm": Path("checkpoints/lstm/lstm.pt"), "transformer": Path("checkpoints/transformer/transformer.pt")}
POOLS = ["mean", "last"]
REPS = [("lstm", "mean", None), ("lstm", "last", None), ("transformer", "mean", None), ("transformer", "last", None), ("lstm", "mean", 300)]
N_CLASSES = 4
LR = 1e-3
EPOCHS = 20
BATCH = 256
SEED = 0
OUT = Path("checkpoints/classify")


def load_lm(name: str, dev: torch.device) -> tuple[LM, Vocab]:
    """Load the best LSTM or Transformer language model checkpoint and its vocab onto dev."""
    ckpt = torch.load(LMS[name], map_location=dev, weights_only=False)
    vocab = ckpt["vocab"]
    if name == "lstm":
        lm = LSTM(len(vocab), ckpt["emb_dim"], ckpt["hid_dim"], ckpt["dropout"], vocab.pad_id)
    else:
        lm = Transformer(len(vocab), ckpt["emb_dim"], ckpt["num_heads"], ckpt["num_layers"], ckpt["ff_dim"], ckpt["max_len"], ckpt["dropout"], vocab.pad_id)
    lm.load_state_dict(ckpt["state"])
    return lm.to(dev), vocab


def unk_rate(X: Sequences, vocab: Vocab) -> float:
    """Return the fraction of tokens that are out of vocabulary and map to <unk>."""
    toks = [tok for seq in X for tok in seq]
    return sum(tok not in vocab.stoi for tok in toks) / len(toks)


def transform(feats: Tensor, scaler: StandardScaler, pca: PCA | None) -> Tensor:
    """Apply the fitted PCA, if any, then standard scaling to features (num_articles, hidden)."""
    x = feats.numpy()
    if pca is not None:
        x = pca.transform(x)
    return torch.tensor(scaler.transform(x), dtype=torch.float32)


@torch.no_grad()
def evaluate(clf: Classifier, x: Tensor, y: Tensor) -> tuple[float, float, float]:
    """Return mean cross-entropy, accuracy and macro-F1 of the classifier on features x and labels y."""
    clf.eval()
    logits = clf(x)
    loss = nn.functional.cross_entropy(logits, y).item()
    pred = logits.argmax(1).cpu().numpy()
    true = y.cpu().numpy()
    return loss, accuracy_score(true, pred), f1_score(true, pred, average="macro")


def train_one_epoch(clf: Classifier, x: Tensor, y: Tensor, opt: torch.optim.Optimizer) -> float:
    """Run one shuffled Adam epoch over features x and labels y and return mean train cross-entropy."""
    clf.train()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    perm = torch.randperm(len(x), device=x.device)
    tot = 0.0
    for i in range(0, len(x), BATCH):
        idx = perm[i:i + BATCH]
        loss = loss_fn(clf(x[idx]), y[idx])
        opt.zero_grad()
        (loss / len(idx)).backward()
        opt.step()
        tot += loss.item()
    return tot / len(x)


def main():
    """Extract frozen LM features, train a linear classifier per representation and save each best epoch.
    LSTM-mean is also PCA-reduced to the Transformer's 300 dims to check whether the LSTM's wider 512-dim features alone explain any gap."""
    dev = get_device()
    data = load_ag_news()
    X_train, y_train = data["train"]
    X_val, y_val = data["validation"]
    y_tr = torch.tensor(y_train, device=dev)
    y_va = torch.tensor(y_val, device=dev)

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
        tr = extract(lm, X_train, vocab)
        va = extract(lm, X_val, vocab)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        extract_times[name] = time.perf_counter() - start
        print(f"extracted {name} features in {extract_times[name]:.2f}s")
        for pool in POOLS:
            feats[(name, pool)] = (tr[pool], va[pool])
    unk = unk_rate(X_train, vocab)

    results = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pool, dim in REPS:
        rep = f"{name}_{pool}" if dim is None else f"{name}_{pool}_pca{dim}"
        tr_f, va_f = feats[(name, pool)]
        start = time.perf_counter()
        x = tr_f.numpy()
        pca = None
        if dim is not None:
            pca = PCA(dim, random_state=SEED).fit(x)
            x = pca.transform(x)
        scaler = StandardScaler().fit(x)
        x_tr = torch.tensor(scaler.transform(x), dtype=torch.float32, device=dev)
        x_va = transform(va_f, scaler, pca).to(dev)
        torch.manual_seed(SEED)
        clf = Classifier(x_tr.size(1), N_CLASSES).to(dev)
        opt = torch.optim.Adam(clf.parameters(), lr=LR)
        print(rep)
        tr_losses = []
        val_losses = []
        val_accs = []
        val_f1s = []
        best_f1 = -1.0
        best_epoch = 0
        best_state = None
        for ep in range(1, EPOCHS + 1):
            tr_loss = train_one_epoch(clf, x_tr, y_tr, opt)
            val_loss, val_acc, val_f1 = evaluate(clf, x_va, y_va)
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            print(f"  epoch {ep:>2} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
        train_time = time.perf_counter() - start
        res = {
            "rep": rep,
            "lm": name,
            "pool": pool,
            "pca_dim": dim,
            "in_dim": x_tr.size(1),
            "lr": LR,
            "best_epoch": best_epoch,
            "val_acc": val_accs[best_epoch - 1],
            "val_f1": best_f1,
            "extract_time_s": extract_times[name],
            "train_time_s": train_time,
            "num_params": sum(p.numel() for p in clf.parameters()),
            "train_loss": tr_losses,
            "val_loss": val_losses,
            "val_acc_curve": val_accs,
            "val_f1_curve": val_f1s,
        }
        results.append(res)
        torch.save({"state": best_state, "scaler": scaler, "pca": pca, **res}, OUT / f"{rep}.pt")

    print(f"AG News classifiers train={len(X_train)} val={len(X_val)} unk_rate={unk:.4f} epochs={EPOCHS} batch={BATCH} lr={LR}")
    print(f"{'rep':>22} {'dim':>4} {'epoch':>5} {'val_acc':>8} {'val_f1':>8} {'extract_s':>9} {'train_s':>8} {'params':>7}")
    for r in results:
        print(f"{r['rep']:>22} {r['in_dim']:>4} {r['best_epoch']:>5} {r['val_acc']:>8.4f} {r['val_f1']:>8.4f} {r['extract_time_s']:>9.2f} {r['train_time_s']:>8.2f} {r['num_params']:>7,}")
    best = max(results, key=lambda r: r["val_f1"])
    print(f"best rep={best['rep']} epoch={best['best_epoch']} val_f1={best['val_f1']:.4f}")

    res_path = OUT / "classify_results.json"
    with open(res_path, "w") as f:
        json.dump({"device": str(dev), "unk_rate": unk, "results": results}, f, indent=2)
    print(f"saved {res_path} and {len(results)} checkpoints in {OUT}")


if __name__ == "__main__":
    main()
