import copy
import json
import math
import platform
import time
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import load_data
from src.rnn.model import RNN
from src.vocab import Vocab, WikiDataset, collate

EMB = 300
DROPOUT = 0.1
HIDS = [256, 512]
LRS = [1e-3, 3e-3]
EPOCHS = 15
BATCH = 64
CLIP = 1.0
SEED = 0
OUT = Path("checkpoints/rnn")


def get_device() -> torch.device:
    """Pick CUDA if available and print the device and hardware in use."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"device=cuda hardware={torch.cuda.get_device_name(dev)}")
    else:
        dev = torch.device("cpu")
        print(f"device=cpu hardware={platform.processor() or platform.machine()}")
    return dev


def make_loader(seqs: list[list[str]], vocab: Vocab, shuffle: bool) -> DataLoader:
    """Wrap encoded WikiText sentences in a padded input/target DataLoader."""
    ds = WikiDataset(seqs, vocab)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, collate_fn=partial(collate, pad_id=vocab.pad_id))


@torch.no_grad()
def avg_nll(model: RNN, loader: DataLoader, pad_id: int, dev: torch.device) -> tuple[float, int]:
    """Return mean next-token NLL over non-pad targets and the number of such targets."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    nll = 0.0
    n_toks = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits, _ = model(x)
        nll += loss_fn(logits.flatten(0, 1), y.flatten()).item()
        n_toks += (y != pad_id).sum().item()
    return nll / n_toks, n_toks


def train_one_epoch(model: RNN, loader: DataLoader, opt: torch.optim.Optimizer, pad_id: int, dev: torch.device) -> float:
    """Run one Adam epoch with gradient clipping and return mean train NLL over non-pad targets."""
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    tot = 0.0
    n_toks = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits, _ = model(x)
        n = (y != pad_id).sum()
        loss = loss_fn(logits.flatten(0, 1), y.flatten())
        opt.zero_grad()
        (loss / n).backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        tot += loss.item()
        n_toks += n.item()
    return tot / n_toks


def main():
    """Train an RNN per hidden size and learning rate, report validation results and save the overall best."""
    dev = get_device()
    data = load_data()
    vocab = Vocab(data["train"])
    train = make_loader(data["train"], vocab, shuffle=True)
    valid = make_loader(data["validation"], vocab, shuffle=False)

    results = []
    best = None
    best_state = None
    for hid in HIDS:
        for lr in LRS:
            torch.manual_seed(SEED)
            model = RNN(len(vocab), EMB, hid, DROPOUT, vocab.pad_id).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            print(f"hid={hid} lr={lr}")
            tr_losses = []
            val_losses = []
            run_best = math.inf
            run_epoch = 0
            run_state = None
            start = time.perf_counter()
            for ep in range(1, EPOCHS + 1):
                tr_nll = train_one_epoch(model, train, opt, vocab.pad_id, dev)
                val_nll, _ = avg_nll(model, valid, vocab.pad_id, dev)
                tr_losses.append(tr_nll)
                val_losses.append(val_nll)
                print(f"  epoch {ep:>2} train_nll={tr_nll:.4f} val_nll={val_nll:.4f} val_ppl={math.exp(val_nll):.2f}")
                if val_nll < run_best:
                    run_best = val_nll
                    run_epoch = ep
                    run_state = copy.deepcopy(model.state_dict())
            train_time = time.perf_counter() - start
            res = {
                "emb_dim": EMB,
                "hid_dim": hid,
                "dropout": DROPOUT,
                "lr": lr,
                "best_epoch": run_epoch,
                "val_nll": run_best,
                "val_ppl": math.exp(run_best),
                "train_time_s": train_time,
                "num_params": sum(p.numel() for p in model.parameters()),
                "train_loss": tr_losses,
                "val_loss": val_losses,
            }
            results.append(res)
            if best is None or run_best < best["val_nll"]:
                best = res
                best_state = run_state

    print(f"RNN vocab={len(vocab)} epochs={EPOCHS} batch={BATCH}")
    print(f"{'hid':>4} {'lr':>6} {'epoch':>5} {'val_nll':>8} {'val_ppl':>9} {'train_s':>8} {'params':>11}")
    for r in results:
        print(f"{r['hid_dim']:>4} {r['lr']:>6} {r['best_epoch']:>5} {r['val_nll']:>8.4f} {r['val_ppl']:>9.2f} {r['train_time_s']:>8.2f} {r['num_params']:>11,}")
    print(f"best hid={best['hid_dim']} lr={best['lr']} epoch={best['best_epoch']} val_ppl={best['val_ppl']:.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "rnn_results.json"
    with open(res_path, "w") as f:
        json.dump({"device": str(dev), "results": results, "best": best}, f, indent=2)
    ckpt_path = OUT / "rnn.pt"
    torch.save({"state": best_state, "vocab": vocab, **best}, ckpt_path)
    print(f"saved {res_path} and {ckpt_path}")


if __name__ == "__main__":
    main()
