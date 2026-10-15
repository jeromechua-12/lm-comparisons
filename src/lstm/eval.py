import json
import math
import time
from pathlib import Path

import torch

from src.data import load_data
from src.lstm.model import LSTM
from src.lstm.train import avg_nll, get_device, make_loader

OUT = Path("checkpoints/lstm")


def main():
    """Evaluate the best validation LSTM checkpoint on the test set."""
    dev = get_device()
    ckpt = torch.load(OUT / "lstm.pt", map_location=dev, weights_only=False)
    vocab = ckpt["vocab"]
    model = LSTM(len(vocab), ckpt["emb_dim"], ckpt["hid_dim"], vocab.pad_id).to(dev)
    model.load_state_dict(ckpt["state"])

    data = load_data()
    test = make_loader(data["test"], vocab, shuffle=False)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    nll, n_toks = avg_nll(model, test, vocab.pad_id, dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    infer_time = time.perf_counter() - start

    res = {
        "device": str(dev),
        "emb_dim": ckpt["emb_dim"],
        "hid_dim": ckpt["hid_dim"],
        "lr": ckpt["lr"],
        "best_epoch": ckpt["best_epoch"],
        "num_params": ckpt["num_params"],
        "test_nll": nll,
        "test_ppl": math.exp(nll),
        "test_tokens": n_toks,
        "infer_time_s": infer_time,
        "tokens_per_s": n_toks / infer_time,
    }

    print(f"LSTM hid={res['hid_dim']} lr={res['lr']} epoch={res['best_epoch']}")
    print(f"{'test_nll':>9} {'test_ppl':>9} {'tokens':>9} {'infer_s':>8} {'tok/s':>9}")
    print(f"{nll:>9.4f} {res['test_ppl']:>9.2f} {n_toks:>9,} {infer_time:>8.2f} {res['tokens_per_s']:>9,.0f}")

    res_path = OUT / "lstm_eval.json"
    with open(res_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {res_path}")


if __name__ == "__main__":
    main()
