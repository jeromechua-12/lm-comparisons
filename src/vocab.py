from collections import Counter

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data import Sequences

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = (PAD, UNK, BOS, EOS)


class Vocab:
    """Token-to-id mapping built from training sequences."""

    def __init__(self, seqs: Sequences, min_freq: int = 3):
        counts = Counter()
        for seq in seqs:
            counts.update(seq)
        self.itos = list(SPECIALS)
        for tok, freq in counts.most_common():
            if freq >= min_freq:
                self.itos.append(tok)
        self.stoi = {}
        for idx, tok in enumerate(self.itos):
            self.stoi[tok] = idx
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, toks: list[str], bos: bool = False, eos: bool = False) -> list[int]:
        """Map tokens to ids, optionally wrapping with <bos> and <eos>."""
        ids = []
        if bos:
            ids.append(self.bos_id)
        for tok in toks:
            ids.append(self.stoi.get(tok, self.unk_id))
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = False) -> list[str]:
        """Map ids back to tokens, optionally dropping special tokens."""
        toks = []
        for idx in ids:
            tok = self.itos[idx]
            if skip_special and tok in SPECIALS:
                continue
            toks.append(tok)
        return toks


class WikiDataset(Dataset):
    """Sentences encoded as id tensors wrapped in <bos> and <eos>."""

    def __init__(self, seqs: Sequences, vocab: Vocab):
        self.items = []
        for seq in seqs:
            ids = vocab.encode(seq, bos=True, eos=True)
            self.items.append(torch.tensor(ids))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tensor:
        return self.items[idx]


def collate(batch: list[Tensor], pad_id: int = 0) -> tuple[Tensor, Tensor]:
    """Right-pad a batch and split it into inputs and targets shifted by one."""
    padded = pad_sequence(batch, batch_first=True, padding_value=pad_id)
    inputs = padded[:, :-1]
    targets = padded[:, 1:]
    return inputs, targets
