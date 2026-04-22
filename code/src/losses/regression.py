"""Pérdida de regresión sobre el score AQA (MSE en escala [0,1])."""
import torch.nn.functional as F


def regression_loss(pred, target):
    return F.mse_loss(pred.view(-1), target.view(-1))
