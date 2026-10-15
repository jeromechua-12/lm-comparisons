import json
import math
import pickle
import time
from pathlib import Path

from src.data import load_data
from src.ngram.model import NGram
from src.vocab import Vocab

ORDERS = [2, 3]
KS = [0.001, 0.01, 0.1, 1.0]
OUT = Path("checkpoints/ngram")


def avg_nll(model: NGram, seqs: list[list[int]], add_k: float) -> float:
    """Return mean next-token negative log-likelihood, counting <eos> as a target."""
    n = model.max_order
    nll = 0.0
    n_toks = 0
    for seq in seqs:
        ids = model.add_boundaries(seq)
        for i in range(n - 1, len(ids)):
            prob = model.probability(ids[:i], ids[i], add_k)
            nll -= math.log(prob)
            n_toks += 1
    return nll / n_toks


def num_params(model: NGram) -> int:
    """Count stored entries: distinct histories plus distinct (history, word) pairs."""
    n_ngrams = sum(len(nxt) for nxt in model.ngram_counts.values())
    return len(model.history_counts) + n_ngrams


def main():
    """Fit an n-gram model per order and add-k, report validation results and save the overall best."""
    data = load_data()
    vocab = Vocab(data["train"])
    train = [vocab.encode(seq) for seq in data["train"]]
    valid = [vocab.encode(seq) for seq in data["validation"]]

    results = []
    best = None
    best_model = None
    for order in ORDERS:
        for k in KS:
            model = NGram(order, len(vocab), vocab.pad_id, vocab.bos_id, vocab.eos_id)
            start = time.perf_counter()
            model.fit(train)
            train_time = time.perf_counter() - start
            nll = avg_nll(model, valid, k)
            res = {
                "max_order": order,
                "add_k": k,
                "val_nll": nll,
                "val_ppl": math.exp(nll),
                "train_time_s": train_time,
                "num_params": num_params(model),
            }
            results.append(res)
            if best is None or nll < best["val_nll"]:
                best = res
                best_model = model

    print(f"NGram vocab={len(vocab)}")
    print(f"{'order':>5} {'add_k':>6} {'val_nll':>8} {'val_ppl':>9} {'train_s':>8} {'params':>11}")
    for r in results:
        print(f"{r['max_order']:>5} {r['add_k']:>6} {r['val_nll']:>8.4f} {r['val_ppl']:>9.2f} {r['train_time_s']:>8.2f} {r['num_params']:>11,}")
    print(f"best order={best['max_order']} add_k={best['add_k']} val_ppl={best['val_ppl']:.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "ngram_results.json"
    with open(res_path, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    ckpt_path = OUT / "ngram.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump({"model": best_model, "vocab": vocab, **best}, f)
    print(f"saved {res_path} and {ckpt_path}")


if __name__ == "__main__":
    main()
