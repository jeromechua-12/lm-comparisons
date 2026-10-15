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


def mean_pool(states: Tensor, lens: Tensor) -> Tensor:
    """Average states (batch, seq, hidden) over real tokens into (batch, hidden), excluding padding."""
    pos = torch.arange(states.size(1), device=states.device)
    mask = (pos[None, :] < lens[:, None]).unsqueeze(-1)
    return (states * mask).sum(1) / lens[:, None]


@torch.no_grad()
def extract(lm: LM, X: Sequences, vocab: Vocab, batch: int = 256) -> Tensor:
    """Encode articles with <bos>/<eos>, run the frozen LM and return mean pooled features (num_articles, hidden) on CPU."""
    lm.eval()
    dev = next(lm.parameters()).device
    feats = []
    for i in range(0, len(X), batch):
        ids = [torch.tensor(vocab.encode(toks, bos=True, eos=True)) for toks in X[i:i + batch]]
        lens = torch.tensor([len(s) for s in ids], device=dev)
        x = pad_sequence(ids, batch_first=True, padding_value=vocab.pad_id).to(dev)
        feats.append(mean_pool(hidden_states(lm, x), lens).cpu())
    return torch.cat(feats)
