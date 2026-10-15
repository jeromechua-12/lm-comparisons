import torch
from torch import Tensor, nn

type State = tuple[Tensor, Tensor]


class LSTM(nn.Module):
    """Single-layer LSTM language model with trainable input embeddings."""

    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=1, batch_first=True)
        self.out = nn.Linear(hid_dim, vocab_size)

    def forward(self, x: Tensor, state: State | None = None) -> tuple[Tensor, State]:
        """Return next-token logits (B, T, V) and the final (hidden, cell) state."""
        emb = self.emb(x)
        states, state = self.lstm(emb, state)
        logits = self.out(states)
        return logits, state

    @torch.no_grad()
    def generate(self, prompt: list[int], eos_id: int, max_len: int = 40, temp: float = 1.0) -> list[int]:
        """Sample up to max_len tokens after the prompt, stopping at <eos>."""
        dev = next(self.parameters()).device
        out = list(prompt)
        logits, state = self(torch.tensor([prompt], device=dev))
        for _ in range(max_len):
            probs = torch.softmax(logits[0, -1] / temp, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt == eos_id:
                break
            out.append(nxt)
            logits, state = self(torch.tensor([[nxt]], device=dev), state)
        return out
