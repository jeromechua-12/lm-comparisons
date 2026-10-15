import json
import pickle
from pathlib import Path

import torch

PROMPTS = ["the game was released", "she was born in", "the river flows", "the war began", "the album received"]
TEMP = 0.8
MAX_LEN = 40
N_SAMPLES = 3
SEED = 0
OUT = Path("checkpoints/ngram")


def distinct2(seqs: list[list[str]]) -> float:
    """Return the fraction of unique bigrams across generated continuations."""
    pairs = [tuple(s[i:i + 2]) for s in seqs for i in range(len(s) - 1)]
    return len(set(pairs)) / max(len(pairs), 1)


def main():
    """Sample continuations of the shared prompts from the best n-gram model and save them."""
    with open(OUT / "ngram.pkl", "rb") as f:
        ckpt = pickle.load(f)
    model = ckpt["model"]
    vocab = ckpt["vocab"]
    add_k = ckpt["add_k"]

    gens = {}
    conts = []
    for i, prompt in enumerate(PROMPTS):
        ids = vocab.encode(prompt.split(), bos=True)
        torch.manual_seed(SEED + i)
        outs = []
        for _ in range(N_SAMPLES):
            cont = vocab.decode(model.generate(ids, MAX_LEN, TEMP, add_k)[len(ids):])
            conts.append(cont)
            outs.append(" ".join(cont))
        gens[prompt] = outs
    toks = [tok for cont in conts for tok in cont]

    res = {
        "max_order": model.max_order,
        "add_k": add_k,
        "temp": TEMP,
        "max_len": MAX_LEN,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "avg_len": len(toks) / len(conts),
        "distinct2": distinct2(conts),
        "unk_rate": toks.count("<unk>") / max(len(toks), 1),
        "generations": gens,
    }

    print(f"NGram order={model.max_order} add_k={add_k} temp={TEMP} max_len={MAX_LEN} samples={N_SAMPLES}")
    print(f"{'avg_len':>8} {'distinct2':>9} {'unk_rate':>8}")
    print(f"{res['avg_len']:>8.1f} {res['distinct2']:>9.3f} {res['unk_rate']:>8.3f}")
    for prompt, outs in gens.items():
        print(f"> {prompt}")
        for out in outs:
            print(f"  {out}")

    res_path = OUT / "ngram_generate.json"
    with open(res_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {res_path}")


if __name__ == "__main__":
    main()
