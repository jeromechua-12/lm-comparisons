import torch
from torch import Tensor, nn


class RNN(nn.Module):
    """Single-layer tanh Elman RNN language model with trainable input embeddings."""

    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.RNN(emb_dim, hid_dim, num_layers=1, nonlinearity="tanh", batch_first=True)
        self.out = nn.Linear(hid_dim, vocab_size)

    def forward(self, x: Tensor, h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return next-token logits (B, T, V) and the final hidden state."""
        emb = self.emb(x)
        states, h = self.rnn(emb, h)
        logits = self.out(states)
        return logits, h

    @torch.no_grad()
    def generate(self, prompt: list[int], eos_id: int, max_len: int = 40, temp: float = 1.0) -> list[int]:
        """Sample up to max_len tokens after the prompt, stopping at <eos>."""
        dev = next(self.parameters()).device
        out = list(prompt)
        logits, h = self(torch.tensor([prompt], device=dev))
        for _ in range(max_len):
            probs = torch.softmax(logits[0, -1] / temp, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt == eos_id:
                break
            out.append(nxt)
            logits, h = self(torch.tensor([[nxt]], device=dev), h)
        return out
