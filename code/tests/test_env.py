import pytest
import torch


def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA no detectado"


def test_gpu_tensor_roundtrip():
    x = torch.randn(2, 3, 16, 56, 56, device="cuda")
    y = x * 2 + 1
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_amp_works():
    x = torch.randn(2, 3, device="cuda")
    with torch.cuda.amp.autocast():
        y = x @ x.T
    assert y.dtype in (torch.float16, torch.float32, torch.bfloat16)
