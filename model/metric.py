import scipy.stats as stats
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from skimage.filters import threshold_otsu


def mse(output, target):
    mse = F.mse_loss(output, target, reduction='none')
    return mse

def tcc(u, x):
    m, n, w = u.shape
    res = []
    for i in range(m):
        correlation_sum = 0
        count = 0
        for j in range(n):
            a = u[i, j, :]
            b = x[i, j, :]
            if (a == a[0]).all() or (b == b[0]).all():
                count += 1
                continue
            correlation_sum = correlation_sum + stats.pearsonr(a, b)[0]
        correlation_sum = correlation_sum / (n - count)
        res.append(correlation_sum)
    res = np.array(res)
    return res


def scc(u, x):
    m, n, w = u.shape
    res = []
    for i in range(m):
        correlation_sum = 0
        count = 0
        for j in range(w):
            a = u[i, :, j]
            b = x[i, :, j]
            if (a == a[0]).all() or (b == b[0]).all():
                count += 1
                continue
            correlation_sum = correlation_sum + stats.pearsonr(a, b)[0]
        correlation_sum = correlation_sum / (w - count)
        res.append(correlation_sum)
    res = np.array(res)
    return res


def dcc_fix_th(u, x):
    m, n, w = u.shape
    dice_cc = []

    u_apd = np.sum(u > 0.02, axis=2)
    u_scar = u_apd > 0.25 * w

    x_apd = np.sum(x > 0.5 * 0.1, axis=2)
    x_scar = x_apd > 0.25 * w

    for i in range(m):
        u_row = u_scar[i, :]
        x_row = x_scar[i, :]

        u_scar_idx = np.where(u_row == 1)[0]
        x_scar_idx = np.where(x_row == 1)[0]

        intersect = set(u_scar_idx) & set(x_scar_idx)
        dice_cc.append(2 * len(intersect) / float(len(set(u_scar_idx)) + len(set(x_scar_idx))))

    dice_cc = np.array(dice_cc)
    return dice_cc


def dcc(u, x):
    m, n, w = u.shape
    dice_cc = []

    u_threshold = 0.02
    x_threshold = 0.035

    for i in range(m):
        u_row = u[i, :, 50]
        x_row = x[i, :, 50]

        u_scar_idx = np.where(u_row > u_threshold)[0]
        x_scar_idx = np.where(x_row > x_threshold)[0]

        intersect = set(u_scar_idx) & set(x_scar_idx)
        dice_cc.append(2 * len(intersect) / float(len(set(u_scar_idx)) + len(set(x_scar_idx))))

    dice_cc = np.array(dice_cc)
    return dice_cc


def scar_tcc(u, x):
    m, n, w = u.shape
    res = []

    u_threshold = 0.02
    x_threshold = 0.035

    for i in range(m):
        u_row = u[i, :, 50]
        x_row = x[i, :, 50]

        u_scar_idx = np.where(u_row > u_threshold)[0]
        x_scar_idx = np.where(x_row > x_threshold)[0]

        intersect = set(u_scar_idx) & set(x_scar_idx)
        n_sub = len(intersect)

        correlation_sum = 0
        count = 0
        for j in intersect:
            a = u[i, j, :]
            b = x[i, j, :]
            if (a == a[0]).all() or (b == b[0]).all():
                count += 1
                continue
            correlation_sum = correlation_sum + stats.pearsonr(a, b)[0]
        if n_sub != 0 and n_sub - count != 0:
            correlation_sum = correlation_sum / (n_sub - count)
        res.append(correlation_sum)
    res = np.array(res)
    return res


def save_data(u, x, u_threshold, x_threshold):
    import scipy.io as sio
    u_sample = np.zeros_like(u[0, :, 50])
    x_sample = np.zeros_like(x[0, :, 50])
    u_sample[u[0, :, 50] > u_threshold] = 1
    x_sample[x[0, :, 50] > x_threshold] = 1
    sio.savemat('./experiments/multi_seg/01/data/u_fixed.mat', {'u': u_sample, 'x': x_sample})
