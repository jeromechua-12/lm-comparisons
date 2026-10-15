# Language Models Comparison

Compares n-gram, RNN, LSTM and Transformer next-token language models on WikiText-2 (Part 1), three input-embedding settings for the LSTM (Part 2), and frozen LM representations for AG News topic classification (Part 3).

## Data

Both datasets are downloaded automatically from the Hugging Face Hub on first use:

| Dataset | Hub ID | Used for |
| --- | --- | --- |
| WikiText-2 (raw) | `Salesforce/wikitext`, `wikitext-2-raw-v1` | Parts 1 and 2 |
| AG News | `fancyzhx/ag_news` | Part 3 |

The public GloVe vectors (`glove-wiki-gigaword-300`) are downloaded automatically through `gensim.downloader` in Part 2.

All preprocessing is done in `src/data.py` and runs inside each script, so it does not need to be called separately. WikiText-2 lines are stripped, headers and empty lines are dropped, `@-@`, `@,@` and `@.@` are joined, text is lower-cased and split into sentences on `.`. The vocabulary (`src/vocab.py`) keeps training tokens with frequency of at least 3 and maps the rest to `<unk>`. AG News is lower-cased, parenthesised text and HTML tags are removed, and a stratified 10% validation split is held out from train.

An optional `.env` file (for example with `HF_TOKEN`) is loaded if present.

## Dependencies

Python 3.13.13 with the packages in `pyproject.toml` (torch, datasets, gensim, scikit-learn, numpy, matplotlib, python-dotenv). Install with [uv](https://docs.astral.sh/uv/):

```
uv sync
```

Neural models were trained on an NVIDIA RTX 4070 Ti (12 GB). Scripts use CUDA if available and fall back to CPU.

## Running

Run every command from the repository root. No flags are required; all configurations are constants at the top of each script. Each step depends on the checkpoints of the steps before it, so run them in this order.

Part 1: language models

```
uv run python -m src.ngram.train
uv run python -m src.ngram.eval
uv run python -m src.ngram.generate

uv run python -m src.rnn.train
uv run python -m src.rnn.eval
uv run python -m src.rnn.generate

uv run python -m src.lstm.train
uv run python -m src.lstm.eval
uv run python -m src.lstm.generate

uv run python -m src.transformer.train
uv run python -m src.transformer.eval
uv run python -m src.transformer.generate
```

Part 2: embedding settings (the randomly initialised setting reuses the Part 1 LSTM)

```
uv run python -m src.skipgram.train

uv run python -m src.lstm_skipgram.train
uv run python -m src.lstm_skipgram.eval
uv run python -m src.lstm_skipgram.generate

uv run python -m src.lstm_glove.train
uv run python -m src.lstm_glove.eval
uv run python -m src.lstm_glove.generate
```

Part 3: classification (needs the Part 1 LSTM and Transformer checkpoints and eval files)

```
uv run python -m src.classify.train
uv run python -m src.classify.eval
```

Loss curve figures (saved to `figures/`):

```
uv run python scripts/plot_curves.py
```

## Configurations

`train` scripts run a small grid search and keep the configuration with the lowest validation loss. All neural LMs train for 10 epochs with batch size 64 and gradient clipping 1.0.

| Model | Fixed settings | Grid searched |
| --- | --- | --- |
| n-gram | add-k smoothing | order {2, 3}, k {0.001, 0.01, 0.1, 1.0} |
| RNN | embedding 300, dropout 0.1 | hidden {256, 512}, lr {1e-3, 3e-3} |
| LSTM | embedding 300, dropout 0.1 | hidden {256, 512}, lr {1e-3, 3e-3} |
| Transformer | embedding 300, 6 heads, feed-forward 1200, dropout 0.1 | layers {2, 4}, lr {1e-4, 5e-4} |
| Skip-gram | dimension 300, window 5, 10 negatives, sample 1e-4, 10 epochs | none |
| LSTM + skip-gram / GloVe | frozen 300-d embeddings, hidden 512, lr 1e-3, dropout 0.1 | none |
| Classifier | linear, lr 1e-3, 50 epochs, batch 256 | none |

All `generate` scripts use the same five prompts, temperature 0.8, up to 40 generated tokens and 3 samples per prompt.

## Random seeds

All seeds are 0 (`SEED = 0` in every script). This covers model initialisation, the AG News validation split, PCA, the GloVe missing-word initialisation and skip-gram training. Generation uses seed `0 + i` for prompt `i`. GPU runs may still differ slightly from the saved results because some CUDA kernels are non-deterministic.

## Results

All results are written to the `checkpoints/` folder, one subfolder per model:

| File | Contents |
| --- | --- |
| `<model>_results.json` | grid search, loss curves, training time, model size |
| `<model>_eval.json` | test loss, perplexity, inference time |
| `<model>_generate.json` | generated continuations |
| `<model>.pt` / `.pkl` / `.kv` | saved weights (not tracked in git) |

`checkpoints/classify/` holds `classify_results.json` and `classify_eval.json` for Part 3.
