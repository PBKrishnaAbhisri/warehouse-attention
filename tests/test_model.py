import torch
from src.models import WarehouseModel


def test_warehouse_model_shapes():
    m = WarehouseModel(d_k=8)
    out, A = m(torch.randn(3, 24, 5), torch.randn(3, 4))
    assert out.shape == (3,) and A.shape == (3, 24, 24)
    assert torch.allclose(A.sum(-1), torch.ones(3, 24), atol=1e-5)

def test_uniform_control_has_equal_weights():
    m = WarehouseModel(d_k=8, uniform=True)
    out, A = m(torch.randn(2, 24, 5), torch.randn(2, 4))
    assert out.shape == (2,)
    assert torch.allclose(A, torch.full((2, 24, 24), 1 / 24))