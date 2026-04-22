import numpy as np

from src.utils.metrics import mae, plcc, srcc


def test_srcc_identity():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(srcc(x, x) - 1.0) < 1e-9


def test_srcc_reversed():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x[::-1].copy()
    assert abs(srcc(x, y) - (-1.0)) < 1e-9


def test_plcc_identity():
    x = np.linspace(0, 1, 10)
    assert abs(plcc(x, x) - 1.0) < 1e-9


def test_mae_zero():
    x = np.array([0.1, 0.2, 0.3])
    assert mae(x, x) == 0.0


def test_mae_scale():
    pred = np.array([0.0])
    gt = np.array([1.0])
    assert abs(mae(pred, gt, scale=100.0) - 100.0) < 1e-9
