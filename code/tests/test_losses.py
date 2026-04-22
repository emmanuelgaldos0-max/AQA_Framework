"""Tests de las 3 pérdidas KD y flujo de gradientes Student-only."""
import torch

from src.losses import (
    FeatureAlignLoss,
    attention_kd_loss,
    attention_map,
    regression_loss,
)


def test_regression_loss_zero():
    x = torch.tensor([0.5, 0.3, 0.8])
    assert regression_loss(x, x).item() == 0.0


def test_attention_map_shape():
    feat = torch.randn(2, 64, 8, 14, 14)
    a = attention_map(feat)
    assert a.shape == (2, 8, 14, 14)


def test_attention_kd_identity_kl():
    feat = torch.randn(2, 64, 8, 14, 14)
    loss = attention_kd_loss(feat, feat, mode="kl")
    # KL(p||p) = 0
    assert loss.item() < 1e-5


def test_attention_kd_identity_mse():
    feat = torch.randn(2, 64, 8, 14, 14)
    loss = attention_kd_loss(feat, feat, mode="mse")
    assert loss.item() < 1e-5


def test_attention_kd_interpolates_different_shapes():
    feat_s = torch.randn(2, 64, 8, 14, 14)
    feat_t = torch.randn(2, 128, 4, 7, 7)
    loss = attention_kd_loss(feat_s, feat_t, mode="kl")
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)


def test_feature_align_loss_identity_zero_ignoring_proj():
    # Con pesos aleatorios en las proyecciones no sale cero, pero sí finito.
    loss_fn = FeatureAlignLoss(s_channels=64, t_channels=128)
    feat_s = torch.randn(2, 64, 8, 14, 14)
    feat_t = torch.randn(2, 128, 8, 7, 7)
    loss = loss_fn(feat_s, feat_t)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_gradient_student_only_through_attention():
    feat_s = torch.randn(2, 64, 8, 14, 14, requires_grad=True)
    feat_t = torch.randn(2, 64, 8, 14, 14, requires_grad=False)
    loss = attention_kd_loss(feat_s, feat_t, mode="kl")
    loss.backward()
    assert feat_s.grad is not None
    assert feat_t.grad is None


def test_gradient_student_only_through_feature_align():
    loss_fn = FeatureAlignLoss(s_channels=64, t_channels=128)
    feat_s = torch.randn(2, 64, 8, 14, 14, requires_grad=True)
    feat_t = torch.randn(2, 128, 8, 7, 7, requires_grad=False)
    loss = loss_fn(feat_s, feat_t)
    loss.backward()
    assert feat_s.grad is not None
    assert feat_t.grad is None
