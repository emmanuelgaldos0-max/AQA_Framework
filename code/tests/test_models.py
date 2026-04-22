"""Tests de shapes, gradientes y features intermedias de los 3 modelos."""
import pytest
import torch

from src.models import (
    build_i3d,
    build_mobilenetv3,
    build_tsm_mobilenetv2,
)


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _input(T=8, device="cpu"):
    # clips cortos para los tests (8 frames en vez de 64) para ahorrar memoria
    return torch.randn(1, 3, T, 224, 224, device=device)


def test_i3d_forward(device):
    model = build_i3d(pretrained=False).to(device).eval()
    x = _input(T=8, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1,)
    assert 0.0 <= y.item() <= 1.0
    assert model.mid_feat.dim() == 5
    assert model.mid_feat.shape[1] == model.mid_feat_channels


def test_tsm_mbv2_forward(device):
    T = 8
    model = build_tsm_mobilenetv2(clip_length=T, pretrained=False).to(device).eval()
    x = _input(T=T, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1,)
    assert model.mid_feat.shape[1] == model.mid_feat_channels
    assert model.final_feat.shape[1] == model.final_feat_channels


def test_mbv3_forward(device):
    T = 8
    model = build_mobilenetv3(clip_length=T, pretrained=False).to(device).eval()
    x = _input(T=T, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1,)
    assert model.mid_feat.shape[1] == model.mid_feat_channels


def test_tsm_gradient_flow():
    T = 4
    model = build_tsm_mobilenetv2(clip_length=T, pretrained=False).train()
    x = torch.randn(2, 3, T, 112, 112, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    # verificar que al menos un parámetro de la cabeza recibió gradiente
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.head.parameters())


def test_student_much_smaller_than_teacher():
    i3d = build_i3d(pretrained=False)
    tsm = build_tsm_mobilenetv2(clip_length=8, pretrained=False)
    mbv3 = build_mobilenetv3(clip_length=8, pretrained=False)
    n_teacher = sum(p.numel() for p in i3d.parameters())
    n_tsm = sum(p.numel() for p in tsm.parameters())
    n_mbv3 = sum(p.numel() for p in mbv3.parameters())
    # Students < 20% del tamaño del Teacher
    assert n_tsm < 0.2 * n_teacher
    assert n_mbv3 < 0.2 * n_teacher
