import torch
import torch.nn as nn


def mse_loss(x_, x, reduction='sum'):
    mse = nn.MSELoss(reduction=reduction)(x_, x)
    return mse


def kl_div(mu1, logvar1, mu2=None, logvar2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if logvar2 is None:
        logvar2 = torch.zeros_like(mu1)

    return 0.5 * (
        logvar2 - logvar1 + (
            torch.exp(logvar1) + (mu1 - mu2).pow(2)
        ) / torch.exp(logvar2) - 1)


def dmm_loss(y, y_p, y_q, mu1, logvar1, mu2, logvar2, kl_annealing_factor=1, r1=1, r2=0):
    kl_raw = kl_div(mu1, logvar1, mu2, logvar2)
    B, T = y.shape[0], y.shape[-1]
    nll_raw_q = mse_loss(y_q, y[:, :, :T], 'none')
    nll_raw_p = mse_loss(y_p, y[:, :, :T], 'none')

    kl_m = kl_raw.sum() / (B * T)
    nll_m_q = nll_raw_q.sum() / (B * T)
    nll_m_p = nll_raw_p.sum() / (B * T)

    loss = kl_m * kl_annealing_factor + r1 * nll_m_q + r2 * nll_m_p

    return kl_m, nll_m_q, nll_m_p, loss
