import numpy as np
import torch

from src.utils.seed import set_seed


def test_seed_reproducible_numpy():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_seed_reproducible_torch():
    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.allclose(a, b)
