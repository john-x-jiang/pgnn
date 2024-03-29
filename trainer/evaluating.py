import os
import sys
import random
import numpy as np

import scipy.io as sio
import scipy.stats as stats
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_driver(model, data_loaders, metrics, hparams, exp_dir, data_tag):
    eval_config = hparams.evaluating
    loss_type = hparams.loss

    evaluate_epoch(model, data_loaders, metrics, exp_dir, hparams, data_tag, eval_config, loss_type=loss_type)


def evaluate_epoch(model, data_loaders, metrics, exp_dir, hparams, data_tag, eval_config, loss_type=None):
    torso_len = eval_config['torso_len']
    signal_scaler = eval_config.get('signal_scaler')
    window = eval_config.get('window')
    time_resolution = eval_config.get('time_resolution')
    
    model.eval()
    n_steps = 0
    mses = {}
    tccs = {}
    sccs = {}
    dccs = {}
    scar_tccs = {}

    q_recons = {}
    all_xs = {}
    all_labels = {}

    with torch.no_grad():
        data_names = list(data_loaders.keys())
        for data_name in data_names:
            data_loader = data_loaders[data_name]
            len_epoch = len(data_loader)
            for idx, data in enumerate(data_loader):
                signal, label = data.x, data.y
                signal = signal.to(device)
                x = signal[:, :-torso_len]
                y = signal[:, -torso_len:]

                is_real = True if -1 in label[:, 1].int() else False

                if signal_scaler is not None:
                    y = y * signal_scaler
                    x = x * signal_scaler
                
                if time_resolution is not None:
                    tr = label[:, 2][0].numpy()
                else:
                    tr = None
                
                if window is not None:
                    y = y[:, :, :window]
                    x = x[:, :, :window]

                physics_vars, _ = model(y, data_name, tr)
                if loss_type == 'data_driven_loss':
                    x_ = physics_vars
                elif loss_type == 'baseline_loss':
                    x_, _ = physics_vars
                elif loss_type == 'physics_loss' or loss_type == 'mixed_loss':
                    x_q, LX_q, y_q, x_p, LX_p, y_p = physics_vars
                    x_ = x_q
                    # x_ = x_p
                else:
                    raise NotImplemented

                if idx == 0:
                    q_recons[data_name] = tensor2np(x_)
                    all_xs[data_name] = tensor2np(x)
                    all_labels[data_name] = tensor2np(label)                    
                else:
                    q_recons[data_name] = np.concatenate((q_recons[data_name], tensor2np(x_)), axis=0)
                    all_xs[data_name] = np.concatenate((all_xs[data_name], tensor2np(x)), axis=0)
                    all_labels[data_name] = np.concatenate((all_labels[data_name], tensor2np(label)), axis=0)

                if not is_real:
                    for met in metrics:
                        if met.__name__ == 'mse':
                            mse = met(x_, x)
                            mse = tensor2np(mse)
                            res = mse.mean((1, 2))
                            if idx == 0:
                                mses[data_name] = res
                            else:
                                mses[data_name] = np.concatenate((mses[data_name], res), axis=0)
                        if met.__name__ == 'tcc':
                            if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                                x = tensor2np(x)
                                x_ = tensor2np(x_)
                            tcc = met(x_, x)
                            if idx == 0:
                                tccs[data_name] = tcc
                            else:
                                tccs[data_name] = np.concatenate((tccs[data_name], tcc), axis=0)
                        if met.__name__ == 'scc':
                            if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                                x = tensor2np(x)
                                x_ = tensor2np(x_)
                            scc = met(x_, x)
                            if idx == 0:
                                sccs[data_name] = scc
                            else:
                                sccs[data_name] = np.concatenate((sccs[data_name], scc), axis=0)
                        if met.__name__ == 'dcc':
                            if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                                x = tensor2np(x)
                                x_ = tensor2np(x_)
                            dcc = met(x_, x)
                            if idx == 0:
                                dccs[data_name] = dcc
                            else:
                                dccs[data_name] = np.concatenate((dccs[data_name], dcc), axis=0)
                        if met.__name__ == 'dcc_fix_th':
                            if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                                x = tensor2np(x)
                                x_ = tensor2np(x_)
                            dcc = met(x_, x)
                            if idx == 0:
                                dccs[data_name] = dcc
                            else:
                                dccs[data_name] = np.concatenate((dccs[data_name], dcc), axis=0)
                        if met.__name__ == 'scar_tcc':
                            if type(x) == torch.Tensor or type(x_) == torch.Tensor:
                                x = tensor2np(x)
                                x_ = tensor2np(x_)
                            scar_tcc = met(x_, x)
                            if idx == 0:
                                scar_tccs[data_name] = scar_tcc
                            else:
                                scar_tccs[data_name] = np.concatenate((scar_tccs[data_name], scar_tcc), axis=0)
                else:
                    for met in metrics:
                        if met.__name__ == 'mse':
                            mses[data_name] = None
                        if met.__name__ == 'tcc':
                            tccs[data_name] = None
                        if met.__name__ == 'scc':
                            sccs[data_name] = None
                        if met.__name__ == 'dcc':
                            dccs[data_name] = None
                        if met.__name__ == 'dcc_fix_th':
                            dccs[data_name] = None
                        if met.__name__ == 'scar_tcc':
                            scar_tccs[data_name] = None

    for met in metrics:
        if met.__name__ == 'mse':
            print_results(exp_dir, 'mse', mses)
        if met.__name__ == 'tcc':
            print_results(exp_dir, 'tcc', tccs)
        if met.__name__ == 'scc':
            print_results(exp_dir, 'scc', sccs)
        if met.__name__ == 'dcc':
            print_results(exp_dir, 'dcc', dccs)
        if met.__name__ == 'dcc_fix_th':
            print_results(exp_dir, 'dcc', dccs)
        if met.__name__ == 'scar_tcc':
            print_results(exp_dir, 'scar_tcc', scar_tccs)

    save_result(exp_dir, q_recons, all_xs, all_labels, data_tag)


def print_results(exp_dir, met_name, mets):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')
    
    data_names = list(mets.keys())
    met = []
    for data_name in data_names:
        if mets[data_name] is None:
            continue
        met.append(mets[data_name])
        print('{}: {} for full seq avg = {:05.5f}, std = {:05.5f}'.format(data_name, met_name, mets[data_name].mean(), mets[data_name].std()))
        with open(os.path.join(exp_dir, 'data/metric.txt'), 'a+') as f:
            f.write('{}: {} for full seq avg = {}, std = {}\n'.format(data_name, met_name, mets[data_name].mean(), mets[data_name].std()))
    if len(met) == 0:
        return
    met = np.hstack(met)
    print('total: {} for full seq avg = {:05.5f}, std = {:05.5f}'.format(met_name, met.mean(), met.std()))
    with open(os.path.join(exp_dir, 'data/metric.txt'), 'a+') as f:
        f.write('total: {} for full seq avg = {}, std = {}\n'.format(met_name, met.mean(), met.std()))


def save_result(exp_dir, recons, all_xs, all_labels, data_tag, obs_len=None):
    if not os.path.exists(exp_dir + '/data'):
        os.makedirs(exp_dir + '/data')
    
    data_names = list(recons.keys())
    for data_name in data_names:
        sio.savemat(
            os.path.join(exp_dir, 'data/{}_{}.mat'.format(data_name, data_tag)), 
            {'recons': recons[data_name], 'inps': all_xs[data_name], 'label': all_labels[data_name]}
        )


def tensor2np(t):
    return t.cpu().detach().numpy()
