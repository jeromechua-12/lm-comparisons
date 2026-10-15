from collections import Counter, defaultdict

import torch
from torch import Tensor

type History = tuple[int, ...]
type HistoryCounts = dict[History, int]
type NGramCounts = dict[History, dict[int, int]]


class NGram:
    """Fixed-order n-gram language model with add-k smoothing and no backoff and interpolation."""

    def __init__(self, max_order: int, vocab_size: int, pad_id: int, bos_id: int, eos_id: int):
        self.max_order = max_order
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.history_counts: HistoryCounts = Counter()
        self.ngram_counts: NGramCounts = defaultdict(Counter)

    def add_boundaries(self, seq: list[int]) -> list[int]:
        """Prepend (max_order - 1) <bos> tokens and append one <eos> token."""
        return [self.bos_id] * (self.max_order - 1) + seq + [self.eos_id]

    def build_ngram_counts(self, seqs: list[list[int]]) -> tuple[HistoryCounts, NGramCounts]:
        """Count each history and each (history, word) pair over boundary-padded sequences."""
        history_counts: HistoryCounts = Counter()
        ngram_counts: NGramCounts = defaultdict(Counter)
        n = self.max_order
        for seq in seqs:
            ids = self.add_boundaries(seq)
            for i in range(n - 1, len(ids)):
                hist = tuple(ids[i - n + 1:i])
                history_counts[hist] += 1
                ngram_counts[hist][ids[i]] += 1
        return history_counts, ngram_counts

    def fit(self, seqs: list[list[int]]) -> "NGram":
        """Build and store n-gram counts from encoded sequences without boundary tokens."""
        self.history_counts, self.ngram_counts = self.build_ngram_counts(seqs)
        return self

    def context(self, history_tokens: list[int]) -> History:
        """Keep the last (max_order - 1) tokens, left-padding with <bos> if too short."""
        k = self.max_order - 1
        padded = [self.bos_id] * k + history_tokens
        return tuple(padded[len(padded) - k:])

    def probability(self, history_tokens: list[int], word: int, add_k: float = 0.1) -> float:
        """Return add-k smoothed P(word | history), or 0 if the denominator is zero."""
        hist = self.context(history_tokens)
        next_counts = self.ngram_counts.get(hist, {})
        count = next_counts.get(word, 0)
        hist_count = self.history_counts.get(hist, 0)
        numer = count + add_k
        denom = hist_count + add_k * self.vocab_size
        if denom == 0:
            return 0.0
        return numer / denom

    def predict_next(self, history_tokens: list[int], add_k: float = 0.1) -> Tensor:
        """Return log P(w | history) for every word, or -inf if the denominator is zero."""
        hist = self.context(history_tokens)
        next_counts = self.ngram_counts.get(hist, {})
        counts = torch.zeros(self.vocab_size)
        for word, count in next_counts.items():
            counts[word] = count
        hist_count = self.history_counts.get(hist, 0)
        numer = counts + add_k
        denom = hist_count + add_k * self.vocab_size
        if denom == 0:
            return torch.full((self.vocab_size,), float("-inf"))
        return torch.log(numer / denom)

    def generate(self, prompt: list[int], max_len: int = 40, temp: float = 1.0, add_k: float = 0.1) -> list[int]:
        """Sample up to max_len tokens after the prompt, skipping <bos> and stopping at <eos>."""
        out = list(prompt)
        for _ in range(max_len):
            logp = self.predict_next(out, add_k)
            probs = torch.softmax(logp / temp, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt == self.eos_id:
                break
            if nxt != self.bos_id:
                out.append(nxt)
        return out
