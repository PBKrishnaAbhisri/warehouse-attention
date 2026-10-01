"""Toy task: associative recall. 8 (key, value) pairs + a query key -> predict its value."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attention import SelfAttention

N_KEYS, N_VALS, N_PAIRS = 16, 10, 8


def make_batch(batch, gen):
    keys = torch.rand(batch, N_KEYS, generator=gen).argsort(dim=1)[:, :N_PAIRS]  # 8 distinct keys
    vals = torch.randint(0, N_VALS, (batch, N_PAIRS), generator=gen)             # random values
    pick = torch.randint(0, N_PAIRS, (batch,), generator=gen)                     # which pair is asked
    idx = torch.arange(batch)
    q_key, target = keys[idx, pick], vals[idx, pick]

    X = torch.zeros(batch, N_PAIRS + 1, N_KEYS + N_VALS)          # one-hot rows
    rows = torch.arange(N_PAIRS)[None, :]
    X[idx[:, None], rows, keys] = 1.0                             # key part of each pair row
    X[idx[:, None], rows, N_KEYS + vals] = 1.0                    # value part of each pair row
    X[idx, N_PAIRS, q_key] = 1.0                                  # query row: key only, value blank
    return X, target

class RecallModel(nn.Module):
    def __init__(self, d_k=16, scale=True, init_std=None):
        super().__init__()
        self.attn = SelfAttention(d_in=N_KEYS + N_VALS, d_k=d_k, scale=scale, init_std=init_std)
        self.head = nn.Linear(d_k, N_VALS)

    def forward(self, X):
        Y, A = self.attn(X)                    # Y: (B, 9, d_k), A: (B, 9, 9)
        return self.head(Y[:, -1, :]), A       # read the query row (last row)


def train(d_k=16, scale=True, lr=1e-2, steps=1500, batch=64, seed=0, log_every=50, init_std=None):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = RecallModel(d_k, scale, init_std)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for step in range(steps + 1):
        X, y = make_batch(batch, gen)
        logits, A = model(X)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        if step % log_every == 0:
            gq = model.attn.W_Q.grad.norm().item()
            gk = model.attn.W_K.grad.norm().item()
            a = A[:, -1, :]                                        # query row weights (B, 9)
            entropy = -(a * (a + 1e-12).log()).sum(-1).mean().item()
            acc = (logits.argmax(-1) == y).float().mean().item()
            history.append(dict(step=step, loss=loss.item(), acc=acc,
                                grad_qk=(gq**2 + gk**2) ** 0.5, entropy=entropy))
        opt.step()
    return model, history


def evaluate(model, n=2000, seed=999):
    gen = torch.Generator().manual_seed(seed)       # fresh data the model never trained on
    X, y = make_batch(n, gen)
    with torch.no_grad():
        logits, _ = model(X)
    return (logits.argmax(-1) == y).float().mean().item()


if __name__ == "__main__":
    model, hist = train()
    print(f"{'step':>5} {'loss':>8} {'acc':>6} {'grad_qk':>9} {'entropy':>8}")
    for h in hist[::3] + [hist[-1]]:
        print(f"{h['step']:5d} {h['loss']:8.4f} {h['acc']:6.2f} {h['grad_qk']:9.4f} {h['entropy']:8.3f}")
    print(f"Test accuracy on fresh data: {evaluate(model):.3f}  (chance = 0.100)")