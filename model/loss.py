import torch
import torch.nn as nn


def mse_loss(x_, x, reduction='sum'):
    mse = nn.MSELoss(reduction=reduction)(x_, x)
    return mse


def kl_div(mu1, var1, mu2=None, var2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if var2 is None:
        var2 = torch.zeros_like(mu1)

    # return 0.5 * (
    #     var2.log() - var1.log() + (
    #         var1 + (mu1 - mu2).pow(2)
    #     ) / var2 - 1)
    return 0.5 * (
        var2 - var1 + (
            torch.exp(var1) + (mu1 - mu2).pow(2)
        ) / torch.exp(var2) - 1)


def data_driven_loss(x, x_):
    B, T = x.shape[0], x.shape[-1]
    loss = mse_loss(x_, x, 'none').sum() / B
    return loss


def dmm_loss(y, y_p, y_q, mu1, var1, mu2, var2, LX_p, LX_q, kl_annealing_factor=1, r1=1, r2=0, smooth=5e-3):
    kl_raw = kl_div(mu1, var1, mu2, var2)
    B, T = y.shape[0], y.shape[-1]
    nll_raw_q = mse_loss(y_q, y[:, :, :T], 'none')
    nll_raw_p = mse_loss(y_p, y[:, :, :T], 'none')
    reg_raw_q = mse_loss(LX_q, torch.zeros_like(LX_q), 'none')
    reg_raw_p = mse_loss(LX_p, torch.zeros_like(LX_p), 'none')

    kl_m = kl_raw.sum() / (B * T)
    nll_m_q = nll_raw_q.sum() / (B * T)
    nll_m_p = nll_raw_p.sum() / (B * T)
    reg_m_q = reg_raw_q.sum() / (B * T)
    reg_m_p = reg_raw_p.sum() / (B * T)

    loss = kl_m * kl_annealing_factor + r1 * (nll_m_q + smooth * reg_m_q) + r2 * (nll_m_p + smooth * reg_m_p)

    return kl_m, nll_m_q, nll_m_p, reg_m_q, reg_m_p, loss


def stochastic_ddr_loss(x, x_q, x_p, LX_q, LX_p, mu_p_seq=None, var_p_seq=None, mu_q_seq=None, var_q_seq=None, kl_annealing_factor=1, r1=1, r2=0):
    B, T = x.shape[0], x.shape[-1]
    nll_raw_q = mse_loss(x_q, x[:, :, :T], 'none')

    if x_p is not None:
        nll_raw_p = mse_loss(x_p, x[:, :, :T], 'none')
    else:
        nll_raw_p = torch.zeros_like(nll_raw_q)

    nll_m_p = nll_raw_p.sum() / B
    nll_m_q = nll_raw_q.sum() / B
    reg_m_p = torch.zeros_like(LX_p).sum()
    reg_m_q = torch.zeros_like(LX_q).sum()

    if mu_q_seq is not None:
        kl_raw = kl_div(mu_q_seq, var_q_seq, mu_p_seq, var_p_seq)
        kl_m = kl_raw.sum() / B
    else:
        kl_m = torch.zeros_like(y).sum()

    loss = kl_m * kl_annealing_factor + r1 * nll_m_q + r2 * nll_m_p

    return kl_m, nll_m_q, nll_m_p, reg_m_q, reg_m_p, loss


def physics_loss(y, y_q, y_p, LX_q, LX_p, mu_p_seq=None, var_p_seq=None, mu_q_seq=None, var_q_seq=None, kl_annealing_factor=1, r1=1, r2=0, smooth=5e-3):
    B, T = y.shape[0], y.shape[-1]
    nll_raw_q = mse_loss(y_q, y[:, :, :T], 'none')
    reg_raw_q = mse_loss(LX_q, torch.zeros_like(LX_q), 'none')

    if y_p is not None:
        nll_raw_p = mse_loss(y_p, y[:, :, :T], 'none')
        reg_raw_p = mse_loss(LX_p, torch.zeros_like(LX_p), 'none')
    else:
        nll_raw_p = torch.zeros_like(nll_raw_q)
        reg_raw_p = torch.zeros_like(nll_raw_q)

    nll_m_q = nll_raw_q.sum() / B
    reg_m_q = reg_raw_q.sum() / B

    nll_m_p = nll_raw_p.sum() / B
    reg_m_p = reg_raw_p.sum() / B

    if mu_q_seq is not None:
        kl_raw = kl_div(mu_q_seq, var_q_seq, mu_p_seq, var_p_seq)
        kl_m = kl_raw.sum() / B
    else:
        kl_m = torch.zeros_like(y).sum() / B

    loss = kl_m * kl_annealing_factor + r1 * (nll_m_q + smooth * reg_m_q) + r2 * (nll_m_p + smooth * reg_m_p)

    return kl_m, nll_m_q, nll_m_p, reg_m_q, reg_m_p, loss


def baseline_loss(x, mu_theta, logvar_theta, mu, logvar, kl_annealing_factor):
    B, T = x.shape[0], x.shape[-1]
    diff_sq = (x - mu_theta).pow(2)
    precis = torch.exp(-logvar_theta)

    nll_m_q = 0.5 * torch.sum(logvar_theta + torch.mul(diff_sq, precis))
    nll_m_p = torch.zeros_like(nll_m_q)

    reg_m_q = torch.zeros_like(nll_m_q)
    reg_m_p = torch.zeros_like(nll_m_q)

    kl_m = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    nll_m_q /= B
    kl_m /= B
    
    loss = kl_m * kl_annealing_factor + nll_m_q
    return kl_m, nll_m_q, nll_m_p, reg_m_q, reg_m_p, loss
