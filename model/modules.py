import os
import torch
import numpy as np
import scipy.io
import pickle
import numbers
import itertools
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchdiffeq

from torch_geometric.nn.inits import uniform
from torch_geometric.data import DataLoader
from data_loader.heart_data import HeartEmptyGraphDataset
from torch_spline_conv import spline_basis, spline_weighting

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Spline(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = degree
        self.norm = norm

        kernel_size = torch.tensor(repeat(kernel_size, dim), dtype=torch.long)
        self.register_buffer('kernel_size', kernel_size)

        is_open_spline = repeat(is_open_spline, dim)
        is_open_spline = torch.tensor(is_open_spline, dtype=torch.uint8)
        self.register_buffer('is_open_spline', is_open_spline)

        K = kernel_size.prod().item()
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, pseudo):
        if edge_index.numel() == 0:
            out = torch.mm(x, self.root)
            out = out + self.bias
            return out

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        row, col = edge_index
        n, m_out = x.size(0), self.weight.size(2)

        # Weight each node.
        basis, weight_index = spline_basis(pseudo, self._buffers['kernel_size'],
                                                self._buffers['is_open_spline'], self.degree)
        weight_index = weight_index.detach()
        out = spline_weighting(x[col], self.weight, basis, weight_index)

        # Convert e x m_out to n x m_out features.
        row_expand = row.unsqueeze(-1).expand_as(out)
        out = x.new_zeros((n, m_out)).scatter_add_(0, row_expand, out)

        # Normalize out by node degree (if wished).
        if self.norm:
            deg = node_degree(row, n, out.dtype, out.device)
            out = out / deg.unsqueeze(-1).clamp(min=1)

        # Weight root node separately (if wished).
        if self.root is not None:
            out = out + torch.mm(x, self.root)

        # Add bias (if wished).
        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



class ST_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_seq,
                 out_seq,
                 dim,
                 kernel_size,
                 process,
                 stride=1,
                 padding=0,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True,
                 sample_rate=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_rate = sample_rate

        self.gcn = Spline(in_channels=in_channels, out_channels=out_channels, dim=dim, kernel_size=kernel_size[0], norm=False)

        if process == 'e':
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    in_seq,
                    out_seq,
                    kernel_size[1],
                    stride,
                    padding
                ),
                nn.ELU(inplace=True)
            )
        elif process == 'd':
            self.tcn = nn.Sequential(
                nn.ConvTranspose2d(
                    in_seq,
                    out_seq,
                    kernel_size[1],
                    stride,
                    padding
                ),
                nn.ELU(inplace=True)
            )
        else:
            raise NotImplementedError

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride, 1)
            ),
            nn.ELU(inplace=True)
        )

    def forward(self, x, edge_index, edge_attr):
        N, V, C, T = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        res = self.residual(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = x.permute(3, 0, 1, 2).contiguous()
        x = x.view(-1, C)
        edge_index, edge_attr = expand(N, V, T, edge_index, edge_attr, self.sample_rate)
        x = F.elu(self.gcn(x, edge_index, edge_attr), inplace=True)
        x = x.view(T, N, V, -1)
        x = x.permute(1, 3, 0, 2).contiguous()

        x = x + res
        x = F.elu(x, inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = self.tcn(x)
        return x.permute(0, 3, 2, 1).contiguous()


def expand(batch_size, num_nodes, T, edge_index, edge_attr, sample_rate=None):
    # edge_attr = edge_attr.repeat(T, 1)
    num_edges = int(edge_index.shape[1] / batch_size)
    edge_index = edge_index[:, 0:num_edges]
    edge_attr = edge_attr[0:num_edges, :]


    sample_number = int(sample_rate * num_edges) if sample_rate is not None else num_edges
    selected_edges = torch.zeros(edge_index.shape[0], batch_size * T * sample_number).to(device)
    selected_attrs = torch.zeros(batch_size * T * sample_number, edge_attr.shape[1]).to(device)

    for i in range(batch_size * T):
        chunk = edge_index + num_nodes * i
        if sample_rate is not None:
            index = np.random.choice(num_edges, sample_number, replace=False)
            index = np.sort(index)
        else:
            index = np.arange(num_edges)
        selected_edges[:, sample_number * i:sample_number * (i + 1)] = chunk[:, index]
        selected_attrs[sample_number * i:sample_number * (i + 1), :] = edge_attr[index, :]

    selected_edges = selected_edges.long()
    return selected_edges, selected_attrs


def repeat(src, length):
    if isinstance(src, numbers.Number):
        src = list(itertools.repeat(src, length))
    return src


def node_degree(index, num_nodes=None, dtype=None, device=None):
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = torch.zeros((num_nodes), dtype=dtype, device=device)
    return out.scatter_add_(0, index, out.new_ones((index.size(0))))


def load_graph(filename, ecgi=0, graph_method=None):
    with open(filename + '.pickle', 'rb') as f:
        g = pickle.load(f)
        g1 = pickle.load(f)
        g2 = pickle.load(f)
        g3 = pickle.load(f)
        g4 = pickle.load(f)

        P10 = pickle.load(f)
        P21 = pickle.load(f)
        P32 = pickle.load(f)
        P43 = pickle.load(f)

        if ecgi == 1:
            t_g = pickle.load(f)
            t_g1 = pickle.load(f)
            t_g2 = pickle.load(f)
            t_g3 = pickle.load(f)

            t_P10 = pickle.load(f)
            t_P21 = pickle.load(f)
            t_P32 = pickle.load(f)

            if graph_method == 'bipartite':
                Hs = pickle.load(f)
                Ps = pickle.load(f)
            else:
                raise NotImplementedError

    if ecgi == 0:
        P01 = P10 / P10.sum(axis=0)
        P12 = P21 / P21.sum(axis=0)
        P23 = P32 / P32.sum(axis=0)
        P34 = P43 / P43.sum(axis=0)

        P01 = torch.from_numpy(np.transpose(P01)).float()
        P12 = torch.from_numpy(np.transpose(P12)).float()
        P23 = torch.from_numpy(np.transpose(P23)).float()
        P34 = torch.from_numpy(np.transpose(P34)).float()

        P10 = torch.from_numpy(P10).float()
        P21 = torch.from_numpy(P21).float()
        P32 = torch.from_numpy(P32).float()
        P43 = torch.from_numpy(P43).float()

        return g, g1, g2, g3, g4, P10, P21, P32, P43, P01, P12, P23, P34
    elif ecgi == 1:
        t_P01 = t_P10 / t_P10.sum(axis=0)
        t_P12 = t_P21 / t_P21.sum(axis=0)
        t_P23 = t_P32 / t_P32.sum(axis=0)

        t_P01 = torch.from_numpy(np.transpose(t_P01)).float()
        t_P12 = torch.from_numpy(np.transpose(t_P12)).float()
        t_P23 = torch.from_numpy(np.transpose(t_P23)).float()

        if graph_method == 'bipartite':
            Ps = torch.from_numpy(Ps).float()
        else:
            raise NotImplementedError

        P10 = torch.from_numpy(P10).float()
        P21 = torch.from_numpy(P21).float()
        P32 = torch.from_numpy(P32).float()
        P43 = torch.from_numpy(P43).float()

        return g, g1, g2, g3, g4, P10, P21, P32, P43,\
            t_g, t_g1, t_g2, t_g3, t_P01, t_P12, t_P23, Hs, Ps


def get_params(data_path, heart_name, batch_size, ecgi=0, graph_method=None):
    # Load physics parameters
    physics_dir = os.path.join(data_path, 'physics/{}/'.format(heart_name))
    mat_files = scipy.io.loadmat(os.path.join(physics_dir, 'h_L.mat'), squeeze_me=True, struct_as_record=False)
    L = mat_files['h_L']

    mat_files = scipy.io.loadmat(os.path.join(physics_dir, 'H.mat'), squeeze_me=True, struct_as_record=False)
    H = mat_files['H']

    L = torch.from_numpy(L).float().to(device)
    print('Load Laplacian: {} x {}'.format(L.shape[0], L.shape[1]))

    H = torch.from_numpy(H).float().to(device)
    print('Load H matrix: {} x {}'.format(H.shape[0], H.shape[1]))

    # Load geometrical parameters
    graph_file = os.path.join(data_path, 'signal/{}/{}_{}'.format(heart_name, heart_name, graph_method))
    if ecgi == 0:
        g, g1, g2, g3, g4, P10, P21, P32, P43, P01, P12, P23, P34 = \
            load_graph(graph_file, ecgi, graph_method)
    else:
        g, g1, g2, g3, g4, P10, P21, P32, P43,\
        t_g, t_g1, t_g2, t_g3, t_P01, t_P12, t_P23, Hs, Ps = load_graph(graph_file, ecgi, graph_method)

    num_nodes = [g.pos.shape[0], g1.pos.shape[0], g2.pos.shape[0], g3.pos.shape[0],
                 g4.pos.shape[0]]
    # print(g)
    # print(g1)
    # print(g2)
    # print(g3)
    # print('P21 requires_grad:', P21.requires_grad)
    print('number of nodes:', num_nodes)

    g_dataset = HeartEmptyGraphDataset(mesh_graph=g)
    g_loader = DataLoader(g_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg = next(iter(g_loader))

    g1_dataset = HeartEmptyGraphDataset(mesh_graph=g1)
    g1_loader = DataLoader(g1_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg1 = next(iter(g1_loader))

    g2_dataset = HeartEmptyGraphDataset(mesh_graph=g2)
    g2_loader = DataLoader(g2_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg2 = next(iter(g2_loader))

    g3_dataset = HeartEmptyGraphDataset(mesh_graph=g3)
    g3_loader = DataLoader(g3_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg3 = next(iter(g3_loader))

    g4_dataset = HeartEmptyGraphDataset(mesh_graph=g4)
    g4_loader = DataLoader(g4_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg4 = next(iter(g4_loader))

    P10 = P10.to(device)
    P21 = P21.to(device)
    P32 = P32.to(device)
    P43 = P43.to(device)

    bg1 = bg1.to(device)
    bg2 = bg2.to(device)
    bg3 = bg3.to(device)
    bg4 = bg4.to(device)

    bg = bg.to(device)

    if ecgi == 0:
        P01 = P01.to(device)
        P12 = P12.to(device)
        P23 = P23.to(device)
        P34 = P34.to(device)

        P1n = np.ones((num_nodes[1], 1))
        Pn1 = P1n / P1n.sum(axis=0)
        Pn1 = torch.from_numpy(np.transpose(Pn1)).float()
        P1n = torch.from_numpy(P1n).float()
        P1n = P1n.to(device)
        Pn1 = Pn1.to(device)

        params = {
            "bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4,
            "P01": P01, "P12": P12, "P23": P23, "P34": P34,
            "P10": P10, "P21": P21, "P32": P32, "P43": P43,
            "P1n": P1n, "Pn1": Pn1, "num_nodes": num_nodes, "g": g, "bg": bg
        }
    elif ecgi == 1:
        t_num_nodes = [t_g.pos.shape[0], t_g1.pos.shape[0], t_g2.pos.shape[0], t_g3.pos.shape[0]]
        # print(t_g)
        # print(t_g1)
        # print(t_g2)
        # print('t_P12 requires_grad:', t_P12.requires_grad)
        print('number of nodes on torso:', t_num_nodes)
        t_g_dataset = HeartEmptyGraphDataset(mesh_graph=t_g)
        t_g_loader = DataLoader(t_g_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg = next(iter(t_g_loader))

        t_g1_dataset = HeartEmptyGraphDataset(mesh_graph=t_g1)
        t_g1_loader = DataLoader(t_g1_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg1 = next(iter(t_g1_loader))

        t_g2_dataset = HeartEmptyGraphDataset(mesh_graph=t_g2)
        t_g2_loader = DataLoader(t_g2_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg2 = next(iter(t_g2_loader))

        t_g3_dataset = HeartEmptyGraphDataset(mesh_graph=t_g3)
        t_g3_loader = DataLoader(t_g3_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg3 = next(iter(t_g3_loader))

        t_P01 = t_P01.to(device)
        t_P12 = t_P12.to(device)
        t_P23 = t_P23.to(device)

        t_bg1 = t_bg1.to(device)
        t_bg2 = t_bg2.to(device)
        t_bg3 = t_bg3.to(device)
        t_bg = t_bg.to(device)

        if graph_method == 'bipartite':
            H_dataset = HeartEmptyGraphDataset(mesh_graph=Hs)
            H_loader = DataLoader(H_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            H_inv = next(iter(H_loader))

            H_inv = H_inv.to(device)
            Ps = Ps.to(device)

            params = {
                "bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4,
                "P10": P10, "P21": P21, "P32": P32, "P43": P43,
                "num_nodes": num_nodes, "g": g, "bg": bg,
                "t_bg1": t_bg1, "t_bg2": t_bg2, "t_bg3": t_bg3,
                "t_P01": t_P01, "t_P12": t_P12, "t_P23": t_P23,
                "t_num_nodes": t_num_nodes, "t_g": t_g, "t_bg": t_bg,
                "H_inv": H_inv, "P": Ps,
                "H": H, "L": L
            }
        else:
            raise NotImplementedError

    return params
