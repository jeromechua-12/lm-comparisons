import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from src.data import Sequences
from src.lstm.model import LSTM
from src.transformer.model import Transformer
from src.vocab import Vocab

type LM = LSTM | Transformer


class Classifier(nn.Module):
    """Linear softmax classifier over fixed-length text representations."""

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return class logits (batch, n_classes) for features (batch, in_dim)."""
        return self.fc(x)


def hidden_states(lm: LM, x: Tensor) -> Tensor:
    """Return the final hidden states (batch, seq, hidden) of an LSTM or Transformer, before its output layer; hidden is 512 for the LSTM and 300 for the Transformer."""
    if isinstance(lm, LSTM):
        states, _ = lm.lstm(lm.emb(x))
        return states
    h = lm.pos(lm.emb(x))
    mask = lm.causal_mask(x.size(1), x.device)
    for layer in lm.layers:
        h = layer(h, mask)
    return h


def pool(states: Tensor, lens: Tensor) -> dict[str, Tensor]:
    """Pool states (batch, seq, hidden) into the mean over real tokens and the last real token state, each (batch, hidden), excluding padding."""
    pos = torch.arange(states.size(1), device=states.device)
    mask = (pos[None, :] < lens[:, None]).unsqueeze(-1)
    mean = (states * mask).sum(1) / lens[:, None]
    last = states[torch.arange(states.size(0), device=states.device), lens - 1]
    return {"mean": mean, "last": last}


@torch.no_grad()
def extract(lm: LM, X: Sequences, vocab: Vocab, batch: int = 256) -> dict[str, Tensor]:
    """Encode articles with <bos>/<eos>, run the frozen LM and return mean and last pooled features (num_articles, hidden) on CPU."""
    lm.eval()
    dev = next(lm.parameters()).device
    feats = {"mean": [], "last": []}
    for i in range(0, len(X), batch):
        ids = [torch.tensor(vocab.encode(toks, bos=True, eos=True)) for toks in X[i:i + batch]]
        lens = torch.tensor([len(s) for s in ids], device=dev)
        x = pad_sequence(ids, batch_first=True, padding_value=vocab.pad_id).to(dev)
        pooled = pool(hidden_states(lm, x), lens)
        for k in feats:
            feats[k].append(pooled[k].cpu())
    return {k: torch.cat(v) for k, v in feats.items()}
