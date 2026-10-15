import json
import platform
import time
from pathlib import Path

from gensim.models import Word2Vec

from src.data import Sequences, load_data
from src.vocab import Vocab

DIM = 300
WINDOW = 5
NEG = 10
SAMPLE = 1e-4
EPOCHS = 10
WORKERS = 8
SEED = 0
OUT = Path("checkpoints/skipgram")


def to_corpus(seqs: Sequences, vocab: Vocab) -> Sequences:
    """Map sentences through the vocab so rare words become <unk> and each is wrapped in <bos>/<eos>."""
    corpus = []
    for seq in seqs:
        corpus.append(vocab.decode(vocab.encode(seq, bos=True, eos=True)))
    return corpus


def main():
    """Train skip-gram word vectors on the training split on CPU and save them with their cost."""
    print(f"device=cpu hardware={platform.processor() or platform.machine()} workers={WORKERS}")
    data = load_data()
    vocab = Vocab(data["train"])
    corpus = to_corpus(data["train"], vocab)
    n_toks = sum(len(seq) for seq in corpus)

    start = time.perf_counter()
    model = Word2Vec(
        corpus,
        vector_size=DIM,
        window=WINDOW,
        min_count=1,
        sg=1,
        negative=NEG,
        sample=SAMPLE,
        epochs=EPOCHS,
        workers=WORKERS,
        seed=SEED,
    )
    train_time = time.perf_counter() - start

    kv = model.wv
    missing = [tok for tok in vocab.itos if tok not in kv.key_to_index]
    res = {
        "device": "cpu",
        "dim": DIM,
        "window": WINDOW,
        "negative": NEG,
        "sample": SAMPLE,
        "epochs": EPOCHS,
        "workers": WORKERS,
        "corpus_tokens": n_toks,
        "vocab_size": len(vocab),
        "vectors": len(kv),
        "missing": missing,
        "train_time_s": train_time,
        "num_params": model.wv.vectors.size + model.syn1neg.size,
    }

    print(f"Skip-gram dim={DIM} window={WINDOW} negative={NEG} epochs={EPOCHS}")
    print(f"{'tokens':>10} {'vocab':>7} {'vectors':>8} {'missing':>8} {'train_s':>8} {'params':>11}")
    print(f"{n_toks:>10,} {len(vocab):>7,} {len(kv):>8,} {len(missing):>8} {train_time:>8.2f} {res['num_params']:>11,}")
    print(f"missing={missing}")

    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "skipgram_results.json"
    with open(res_path, "w") as f:
        json.dump(res, f, indent=2)
    kv_path = OUT / "skipgram.kv"
    kv.save(str(kv_path))
    print(f"saved {res_path} and {kv_path}")


if __name__ == "__main__":
    main()
