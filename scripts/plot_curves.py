import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT = Path("checkpoints")
OUT = Path("figures")


def main():
    """Plot Part 1 and Part 2 training and validation loss from the saved results JSONs."""
    OUT.mkdir(exist_ok=True)

    # part 1
    models = [("RNN", "rnn"), ("LSTM", "lstm"), ("Transformer", "transformer")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, (label, name) in zip(axes.flat, models):
        with open(CKPT / name / f"{name}_results.json") as f:
            run = json.load(f)["best"]
        epochs = range(1, len(run["val_loss"]) + 1)
        ax.plot(epochs, run["train_loss"], label="Training loss")
        ax.plot(epochs, run["val_loss"], label="Validation loss")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL")
        ax.legend()
    ax = axes.flat[3]
    for label, name in models:
        with open(CKPT / name / f"{name}_results.json") as f:
            run = json.load(f)["best"]
        ax.plot(range(1, len(run["val_loss"]) + 1), run["val_loss"], label=label)
    ax.set_title("Validation loss compared")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.legend()
    fig.suptitle("Part 1: training and validation loss")
    fig.tight_layout()
    fig.savefig(OUT / "part1_loss.png", dpi=150)
    plt.close(fig)

    # part 2
    models = [("LSTM, skip-gram (frozen)", "lstm_skipgram"), ("LSTM, GloVe (frozen)", "lstm_glove")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (label, name) in zip(axes, models):
        with open(CKPT / name / f"{name}_results.json") as f:
            run = json.load(f)["best"]
        epochs = range(1, len(run["val_loss"]) + 1)
        ax.plot(epochs, run["train_loss"], label="Training loss")
        ax.plot(epochs, run["val_loss"], label="Validation loss")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL")
        ax.legend()
    fig.suptitle("Part 2: training and validation loss")
    fig.tight_layout()
    fig.savefig(OUT / "part2_loss.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
