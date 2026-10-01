"""Gradient verification: autograd vs finite differences for W_Q, W_K, W_V."""
import torch
from src.attention import attention

torch.manual_seed(0)
dtype = torch.float64   # float32 is too imprecise for finite differences


def loss_fn(X, W_Q, W_K, W_V, R):
    # A scalar "loss": run attention, then weight the output by a fixed random matrix R.
    Y, _ = attention(X, W_Q, W_K, W_V)
    return (Y * R).sum()


def numerical_grad(f, W, eps=1e-6):
    """Nudge each entry of W by +eps and -eps, measure how the loss changes."""
    W = W.detach().clone()
    grad = torch.zeros_like(W)
    for i in range(W.numel()):
        orig = W.view(-1)[i].item()
        W.view(-1)[i] = orig + eps
        f_plus = f(W).item()
        W.view(-1)[i] = orig - eps
        f_minus = f(W).item()
        W.view(-1)[i] = orig                      # restore
        grad.view(-1)[i] = (f_plus - f_minus) / (2 * eps)
    return grad


B, T, d_in, d_k, d_v = 2, 6, 5, 4, 3
X = torch.randn(B, T, d_in, dtype=dtype)
R = torch.randn(B, T, d_v, dtype=dtype)
params = {
    "W_Q": torch.randn(d_in, d_k, dtype=dtype) / d_in ** 0.5,
    "W_K": torch.randn(d_in, d_k, dtype=dtype) / d_in ** 0.5,
    "W_V": torch.randn(d_in, d_v, dtype=dtype) / d_in ** 0.5,
}

# 1) Analytical gradient: autograd (chain rule backwards through attention)
leaf = {n: p.clone().requires_grad_(True) for n, p in params.items()}
loss = loss_fn(X, leaf["W_Q"], leaf["W_K"], leaf["W_V"], R)
loss.backward()

# 2) Numerical gradient: finite differences, compared per matrix
print(f"{'param':6} {'analytic[0,0]':>15} {'numeric[0,0]':>15} {'max abs diff':>14} {'rel diff':>10}")
for name in params:
    def f(W, name=name):
        a = {n: (W if n == name else params[n]) for n in params}
        return loss_fn(X, a["W_Q"], a["W_K"], a["W_V"], R)

    num = numerical_grad(f, params[name])
    ana = leaf[name].grad
    abs_diff = (ana - num).abs().max().item()
    rel_diff = ((ana - num).norm() / (ana.norm() + num.norm())).item()
    print(f"{name:6} {ana[0,0].item():15.8f} {num[0,0].item():15.8f} {abs_diff:14.2e} {rel_diff:10.2e}")
    assert rel_diff < 1e-6, f"gradient mismatch for {name}"

print("Gradient check PASSED")