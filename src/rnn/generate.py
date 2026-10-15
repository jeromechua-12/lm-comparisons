import json
from pathlib import Path

import torch

from src.rnn.model import RNN
from src.rnn.train import get_device

PROMPTS = ["the game was released", "she was born in", "the river flows", "the war began", "the album received"]
TEMP = 0.8
MAX_LEN = 40
N_SAMPLES = 3
SEED = 0
OUT = Path("checkpoints/rnn")


def distinct2(seqs: list[list[str]]) -> float:
    """Return the fraction of unique bigrams across generated continuations."""
    pairs = [tuple(s[i:i + 2]) for s in seqs for i in range(len(s) - 1)]
    return len(set(pairs)) / max(len(pairs), 1)


def main():
    """Sample continuations of the shared prompts from the best RNN checkpoint and save them."""
    dev = get_device()
    ckpt = torch.load(OUT / "rnn.pt", map_location=dev, weights_only=False)
    vocab = ckpt["vocab"]
    model = RNN(len(vocab), ckpt["emb_dim"], ckpt["hid_dim"], ckpt["dropout"], vocab.pad_id).to(dev)
    model.load_state_dict(ckpt["state"])
    model.eval()

    gens = {}
    conts = []
    for i, prompt in enumerate(PROMPTS):
        ids = vocab.encode(prompt.split(), bos=True)
        torch.manual_seed(SEED + i)
        outs = []
        for _ in range(N_SAMPLES):
            cont = vocab.decode(model.generate(ids, vocab.eos_id, MAX_LEN, TEMP)[len(ids):])
            conts.append(cont)
            outs.append(" ".join(cont))
        gens[prompt] = outs
    toks = [tok for cont in conts for tok in cont]

    res = {
        "device": str(dev),
        "hid_dim": ckpt["hid_dim"],
        "best_epoch": ckpt["best_epoch"],
        "temp": TEMP,
        "max_len": MAX_LEN,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "avg_len": len(toks) / len(conts),
        "distinct2": distinct2(conts),
        "unk_rate": toks.count("<unk>") / max(len(toks), 1),
        "generations": gens,
    }

    print(f"RNN hid={res['hid_dim']} epoch={res['best_epoch']} temp={TEMP} max_len={MAX_LEN} samples={N_SAMPLES}")
    print(f"{'avg_len':>8} {'distinct2':>9} {'unk_rate':>8}")
    print(f"{res['avg_len']:>8.1f} {res['distinct2']:>9.3f} {res['unk_rate']:>8.3f}")
    for prompt, outs in gens.items():
        print(f"> {prompt}")
        for out in outs:
            print(f"  {out}")

    res_path = OUT / "rnn_generate.json"
    with open(res_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {res_path}")


if __name__ == "__main__":
    main()
