import os
import numpy as np
import scipy.io as sio
from skimage.filters import threshold_otsu
import argparse


def parse_args():
    """
    Args:
        config: json file with hyperparams and exp settings
        seed: random seed value
        stage: 1 for traing VAE, 2 for optimization,  and 12 for both
        logging: 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--exps', type=str, default='mixed_loss', help='experiment name')
    parser.add_argument('--idx', type=str, default='03', help='experiment id')
    parser.add_argument('--heart', type=str, default='EC', help='heart id')
    parser.add_argument('--th', type=float, default=0.15, help='scar threshold')

    args = parser.parse_args()
    return args


def scar(u, x, threshold):
    N, T = u.shape

    u_abs = np.abs(u)
    u_peak = np.max(u_abs, axis=1)

    x_scar = x < 1.5
    x_scar_idx = np.where(x_scar == True)[0]

    # u_scar = u_peak > threshold_otsu(u_peak)
    u_scar = u_peak < threshold
    u_scar_idx = np.where(u_scar == True)[0]

    intersect = set(u_scar_idx) & set(x_scar_idx)
    dice_cc = 2 * len(intersect) / float(len(set(u_scar_idx)) + len(set(x_scar_idx)))

    return u_scar.astype(float), dice_cc


args = parse_args()

path = 'experiments/{}/{}/data/{}_real_test.mat'.format(args.exps, args.idx, args.heart)
# path = 'experiments/{}/{}_real/{}_test.mat'.format(args.exps, args.heart, args.heart)
data = sio.loadmat(path)
recons = data['recons']

path = './data/structure/{}/inf_idx.mat'.format(args.heart)
data = sio.loadmat(path)
inf_idx = data['inf_idx']
inf_idx = np.squeeze(inf_idx)

# recons = np.transpose(recons, [2, 0, 1])
# recons = recons[:, :, :65]
recons = recons[:, inf_idx == 3, :]

path = '/data/Halifax/Real/{}/bipolar.mat'.format(args.heart)
data = sio.loadmat(path)
grnths = data['bipolar']
grnths = np.squeeze(grnths)

u_scars = np.zeros([4, recons.shape[1]])
dice = np.zeros(4)
for i in range(4):
    u_scars[i], dice[i] = scar(recons[i], grnths, args.th)
    print("Dice coefficient = {}".format(dice[i]))
print("Dice coefficient avg = {}, std = {}".format(dice.mean(), dice.std()))
sio.savemat('experiments/{}/{}/data/{}_scar.mat'.format(args.exps, args.idx, args.heart), {'recons': recons, 'scar': u_scars})
# sio.savemat('experiments/{}/{}_real/{}_scar.mat'.format(args.exps, args.heart, args.heart), {'recons': recons, 'scar': u_scars})
