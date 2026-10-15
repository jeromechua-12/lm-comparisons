import copy
import json
import math
import time
from collections import Counter
from pathlib import Path

import gensim.downloader
import torch
from gensim.models import KeyedVectors
from torch import Tensor

from src.data import Sequences, load_data
from src.lstm.model import LSTM
from src.lstm.train import avg_nll, get_device, make_loader, train_one_epoch
from src.vocab import Vocab

GLOVE = "glove-wiki-gigaword-300"
EMB = 300
HID = 512
LR = 1e-3
DROPOUT = 0.1
EPOCHS = 10
SEED = 0
OUT = Path("checkpoints/lstm_glove")


def glove_weights(kv: KeyedVectors, vocab: Vocab, seqs: Sequences) -> tuple[Tensor, dict]:
    """Stack GloVe vectors in vocab id order; missing tokens get N(0, std of found vectors), <pad> gets zeros."""
    found = []
    missing = []
    for idx, tok in enumerate(vocab.itos):
        if idx == vocab.pad_id:
            continue
        if tok in kv.key_to_index:
            found.append(idx)
        else:
            missing.append(idx)
    vecs = torch.tensor(kv[[vocab.itos[i] for i in found]])
    std = vecs.std().item()
    gen = torch.Generator().manual_seed(SEED)
    weights = torch.zeros(len(vocab), EMB)
    weights[found] = vecs
    weights[missing] = torch.randn(len(missing), EMB, generator=gen) * std

    counts = Counter()
    for seq in seqs:
        counts.update(vocab.encode(seq))
    n_toks = sum(counts.values())
    found_toks = sum(counts[i] for i in found)
    cov = {
        "vocab_types": len(vocab) - 1,
        "found_types": len(found),
        "missing_types": len(missing),
        "type_coverage": len(found) / (len(vocab) - 1),
        "token_coverage": found_toks / n_toks,
        "init_std": std,
        "missing_top": [vocab.itos[i] for i in missing[:30]],
    }
    return weights, cov


def main():
    """Train the Part 1 best LSTM config with frozen public GloVe input embeddings and save the best epoch."""
    dev = get_device()
    data = load_data()
    vocab = Vocab(data["train"])
    train = make_loader(data["train"], vocab, shuffle=True)
    valid = make_loader(data["validation"], vocab, shuffle=False)
    weights, cov = glove_weights(gensim.downloader.load(GLOVE), vocab, data["train"])
    print(f"glove types={cov['found_types']:,}/{cov['vocab_types']:,} ({cov['type_coverage']:.2%}) tokens={cov['token_coverage']:.2%} init_std={cov['init_std']:.4f}")

    torch.manual_seed(SEED)
    model = LSTM(len(vocab), EMB, HID, DROPOUT, vocab.pad_id, weights).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    print(f"hid={HID} lr={LR} emb=glove (frozen)")
    tr_losses = []
    val_losses = []
    best_nll = math.inf
    best_epoch = 0
    best_state = None
    start = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        tr_nll = train_one_epoch(model, train, opt, vocab.pad_id, dev)
        val_nll, _ = avg_nll(model, valid, vocab.pad_id, dev)
        tr_losses.append(tr_nll)
        val_losses.append(val_nll)
        print(f"  epoch {ep:>2} train_nll={tr_nll:.4f} val_nll={val_nll:.4f} val_ppl={math.exp(val_nll):.2f}")
        if val_nll < best_nll:
            best_nll = val_nll
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
    train_time = time.perf_counter() - start

    res = {
        "emb": "glove",
        "emb_dim": EMB,
        "hid_dim": HID,
        "dropout": DROPOUT,
        "lr": LR,
        "best_epoch": best_epoch,
        "val_nll": best_nll,
        "val_ppl": math.exp(best_nll),
        "train_time_s": train_time,
        "num_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in params),
        "coverage": cov,
        "train_loss": tr_losses,
        "val_loss": val_losses,
    }

    print(f"LSTM glove vocab={len(vocab)} epochs={EPOCHS}")
    print(f"{'hid':>4} {'lr':>6} {'epoch':>5} {'val_nll':>8} {'val_ppl':>9} {'train_s':>8} {'params':>11} {'trainable':>11}")
    print(f"{HID:>4} {LR:>6} {best_epoch:>5} {best_nll:>8.4f} {res['val_ppl']:>9.2f} {train_time:>8.2f} {res['num_params']:>11,} {res['trainable_params']:>11,}")

    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "lstm_glove_results.json"
    with open(res_path, "w") as f:
        json.dump({"device": str(dev), "results": [res], "best": res}, f, indent=2)
    ckpt_path = OUT / "lstm_glove.pt"
    torch.save({"state": best_state, "vocab": vocab, **res}, ckpt_path)
    print(f"saved {res_path} and {ckpt_path}")


if __name__ == "__main__":
    main()
