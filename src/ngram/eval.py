import json
import math
import pickle
import time
from pathlib import Path

from src.data import load_data
from src.ngram.train import avg_nll

OUT = Path("checkpoints/ngram")


def main():
    """Evaluate the best validation n-gram model on the test set."""
    with open(OUT / "ngram.pkl", "rb") as f:
        ckpt = pickle.load(f)
    model = ckpt["model"]
    vocab = ckpt["vocab"]
    add_k = ckpt["add_k"]

    data = load_data()
    test = [vocab.encode(seq) for seq in data["test"]]
    n_toks = sum(len(seq) + 1 for seq in test)

    start = time.perf_counter()
    nll = avg_nll(model, test, add_k)
    infer_time = time.perf_counter() - start

    res = {
        "max_order": model.max_order,
        "add_k": add_k,
        "test_nll": nll,
        "test_ppl": math.exp(nll),
        "test_tokens": n_toks,
        "infer_time_s": infer_time,
        "tokens_per_s": n_toks / infer_time,
    }

    print(f"NGram order={res['max_order']} add_k={add_k}")
    print(f"{'test_nll':>9} {'test_ppl':>9} {'tokens':>9} {'infer_s':>8} {'tok/s':>9}")
    print(f"{res['test_nll']:>9.4f} {res['test_ppl']:>9.2f} {n_toks:>9,} {infer_time:>8.2f} {res['tokens_per_s']:>9,.0f}")

    res_path = OUT / "ngram_eval.json"
    with open(res_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {res_path}")


if __name__ == "__main__":
    main()
