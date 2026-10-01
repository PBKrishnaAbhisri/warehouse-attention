import math
import torch
import torch.nn.functional as F
from src.attention import attention, stable_softmax, naive_softmax, SelfAttention


def make_inputs(B=2, T=24, d_in=5, d_k=8, d_v=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, T, d_in, generator=g)
    W_Q = torch.randn(d_in, d_k, generator=g) / math.sqrt(d_in)
    W_K = torch.randn(d_in, d_k, generator=g) / math.sqrt(d_in)
    W_V = torch.randn(d_in, d_v, generator=g) / math.sqrt(d_in)
    return X, W_Q, W_K, W_V


def test_shapes():
    X, W_Q, W_K, W_V = make_inputs()
    Y, A = attention(X, W_Q, W_K, W_V)
    assert A.shape == (2, 24, 24)
    assert Y.shape == (2, 24, 6)


def test_attention_rows_sum_to_one_and_nonnegative():
    X, W_Q, W_K, W_V = make_inputs()
    _, A = attention(X, W_Q, W_K, W_V)
    assert torch.allclose(A.sum(dim=-1), torch.ones(2, 24), atol=1e-6)
    assert (A >= 0).all()


def test_stable_softmax_matches_torch():
    s = torch.randn(4, 10)
    assert torch.allclose(stable_softmax(s), torch.softmax(s, dim=-1), atol=1e-6)


def test_matches_independent_reference():
    # Independent check: PyTorch's built-in attention is used ONLY here, never in src/.
    X, W_Q, W_K, W_V = make_inputs()
    Y, _ = attention(X, W_Q, W_K, W_V)
    ref = F.scaled_dot_product_attention(X @ W_Q, X @ W_K, X @ W_V)
    assert torch.allclose(Y, ref, atol=1e-5)


def test_stable_softmax_survives_large_logits():
    s = torch.tensor([[1000.0, 1001.0, 999.0]])
    out = stable_softmax(s)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(), torch.tensor(1.0))


def test_naive_softmax_breaks_on_large_logits():
    s = torch.tensor([[1000.0, 1001.0, 999.0]])
    assert torch.isnan(naive_softmax(s)).any()  # exp(1000) = inf, inf/inf = nan


def test_module_output_dimensions():
    layer = SelfAttention(d_in=5, d_k=8, d_v=6)
    Y, A = layer(torch.randn(3, 24, 5))
    assert Y.shape == (3, 24, 6) and A.shape == (3, 24, 24)

def test_gradcheck_against_finite_differences():
    X = torch.randn(1, 4, 3, dtype=torch.float64)
    W = [torch.randn(3, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    assert torch.autograd.gradcheck(
        lambda a, b, c: attention(X, a, b, c)[0], W, eps=1e-6, atol=1e-5)