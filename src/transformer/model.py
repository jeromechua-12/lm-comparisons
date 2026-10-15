import math

import torch
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with an optional mask over blocked positions."""

    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)
        self.w_o = nn.Linear(embed_size, embed_size)

    def scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return softmax(q k^T / sqrt(head_size)) v, with True positions in mask blocked."""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        return out

    def split_heads(self, x: Tensor) -> Tensor:
        """Reshape (batch, seq, embed) into (batch, heads, seq, head_size)."""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.head_size)
        return x.transpose(1, 2)

    def combine_heads(self, x: Tensor) -> Tensor:
        """Reshape (batch, heads, seq, head_size) back into (batch, seq, embed)."""
        batch, _, seq, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.embed_size)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Project x into per-head queries, keys and values, attend, then merge heads and project back to (batch, seq, embed)."""
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))
        attention_output = self.scaled_dot_product_attention(q, k, v, mask)
        combined = self.combine_heads(attention_output)
        output = self.w_o(combined)
        return output


class PositionWiseFeedForward(nn.Module):
    """Two-layer ReLU network applied independently at every position."""

    def __init__(self, embed_size: int, ff_size: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, ff_size)
        self.fc2 = nn.Linear(ff_size, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Expand each position from embed_size to ff_size, apply ReLU, then project back to (batch, seq, embed)."""
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        output = self.fc2(hidden)
        return output


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position signals added to token embeddings."""

    def __init__(self, embed_size: int, max_len: int):
        super().__init__()
        if embed_size % 2 != 0:
            raise ValueError("embed_size must be even for sinusoidal encoding")
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2, dtype=torch.float) * -(math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Add the encoding of positions 0..seq-1 to embeddings x of shape (batch, seq, embed)."""
        seq = x.size(1)
        output = x + self.pe[:, :seq]
        return output


class DecoderLayer(nn.Module):
    """Masked self-attention then feed-forward, each followed by dropout, a residual and LayerNorm."""

    def __init__(self, embed_size: int, num_heads: int, ff_size: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_size, num_heads)
        self.feed_forward = PositionWiseFeedForward(embed_size, ff_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Attend over preceding positions given the causal mask, then transform each position, keeping (batch, seq, embed)."""
        attention_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout(ff_output))
        return output


class Transformer(nn.Module):
    """Decoder-only Transformer language model with causal self-attention and trainable input embeddings."""

    def __init__(self, vocab_size: int, embed_size: int, num_heads: int, num_layers: int, ff_size: int, max_len: int, dropout: float, pad_id: int):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
        self.pos = PositionalEncoding(embed_size, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(embed_size, num_heads, ff_size, dropout) for _ in range(num_layers)])
        self.out = nn.Linear(embed_size, vocab_size)

    def causal_mask(self, seq: int, dev: torch.device) -> Tensor:
        """Return a (seq, seq) mask that is True above the diagonal, blocking future positions."""
        ones = torch.ones(seq, seq, dtype=torch.bool, device=dev)
        return torch.triu(ones, diagonal=1)

    def forward(self, x: Tensor) -> Tensor:
        """Return next-token logits (batch, seq, vocab) using only preceding tokens."""
        if x.size(1) > self.max_len:
            raise ValueError(f"sequence length {x.size(1)} exceeds max_len {self.max_len}")
        emb = self.emb(x)
        h = self.pos(emb)
        h = self.dropout(h)
        mask = self.causal_mask(x.size(1), x.device)
        for layer in self.layers:
            h = layer(h, mask)
        logits = self.out(h)
        return logits

    @torch.no_grad()
    def generate(self, prompt: list[int], eos_id: int, max_len: int = 40, temp: float = 1.0) -> list[int]:
        """Sample up to max_len tokens after the prompt, stopping at <eos>; context is the last self.max_len tokens."""
        dev = next(self.parameters()).device
        out = list(prompt)
        for _ in range(max_len):
            ctx = torch.tensor([out[-self.max_len:]], device=dev)
            logits = self(ctx)
            probs = torch.softmax(logits[0, -1] / temp, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt == eos_id:
                break
            out.append(nxt)
        return out
