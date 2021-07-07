import torch
import torch.nn as nn


def mse_loss(x_, x):
    mse = nn.MSELoss(reduction='sum')(x_, x)
    return mse
