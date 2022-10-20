import torch
import torch.nn as nn
import numpy as np
from model.modules import *
from abc import abstractmethod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError
    
    def setup(self, *inputs):
        pass

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class Euclidean(BaseModel):
    def __init__(self, seq_len, in_dim, out_dim, mid_dim_i, mid_dim_o, latent_dim, v_latent, v_mid):
        super().__init__()
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.mid_dim_i = mid_dim_i
        self.mid_dim_o = mid_dim_o
        self.v_mid = v_mid
        self.v_latent = v_latent

        # encoder
        self.fc1 = nn.LSTM(in_dim, mid_dim_i)
        self.fc21 = nn.LSTM(mid_dim_i, latent_dim)
        self.fc22 = nn.LSTM(mid_dim_i, latent_dim)
        self.lin1 =nn.Linear(latent_dim * seq_len, v_mid)
        self.lin2 =nn.Linear(v_mid, v_latent)

        # decoder
        self.lin3 = nn.Linear(v_latent, v_mid)
        self.lin4 = nn.Linear(v_mid, latent_dim * seq_len)
        self.fc3 = nn.LSTM(latent_dim, mid_dim_o)
        self.fc41 = nn.LSTM(mid_dim_o, out_dim)
        self.fc42 = nn.LSTM(mid_dim_o, out_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x, heart_name):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.seq_len)
        x = x.permute(2, 0, 1).contiguous()
        _, B, _ = x.shape
        out, hidden = self.fc1(x)
        h1 = self.relu(out)
        out21, hidden21 = self.fc21(h1)
        outMean = out21.permute(1, 2, 0).contiguous().view(B,-1)
        outMean = self.relu(self.lin1(outMean))
        outMean = self.relu(self.lin2(outMean))
        out22, hidden22 = self.fc22(h1)
        outVar = out22.permute(1, 2, 0).contiguous().view(B, -1)
        outVar = self.relu(self.lin1(outVar))
        outVar = self.relu(self.lin2(outVar))
        return outMean, outVar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z, heart_name):
        B, _ = z.shape
        z1 = self.relu(self.lin3(z))
        z2 = self.relu(self.lin4(z1))
        z = z2.view(B, self.latent_dim,-1).permute(2, 0, 1)

        out3, hidden3 = self.fc3(z)
        h3 = self.relu(out3)
        out1,hidden1 = self.fc41(h3)
        out2, hidden2 = self.fc42(h3)
        out1 = out1.permute(1, 2, 0).contiguous()
        out2 = out2.permute(1, 2, 0).contiguous()
        return out1, out2
    
    def forward(self, x, heart_name):
        mu, logvar = self.encode(x, heart_name)
        z = self.reparameterize(mu, logvar)
        mu_theta, logvar_theta = self.decode(z, heart_name)
        return (mu_theta, logvar_theta), (mu, logvar)


class ST_GCNN(BaseModel):
    def __init__(self, num_channel, num_sequence, latent_feature, latent_sequence):
        super().__init__()
        self.nf = num_channel
        self.ns = num_sequence
        self.latent_feature = latent_feature
        self.latent_sequence = latent_sequence

        self.conv1 = ST_Block(self.nf[0], self.nf[2], self.ns[0], self.ns[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = ST_Block(self.nf[2], self.nf[3], self.ns[2], self.ns[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = ST_Block(self.nf[3], self.nf[4], self.ns[3], self.ns[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[-1], 1)
        self.fce21 = nn.Conv2d(self.nf[-1], self.latent_feature, 1)

        self.trans = Spline(self.latent_feature, self.latent_feature, dim=3, kernel_size=3, norm=False, degree=2, root_weight=False, bias=False)
        
        self.fcd3 = nn.Conv2d(self.latent_feature, self.nf[-1], 1)
        self.fcd4 = nn.Conv2d(self.nf[-1], self.nf[5], 1)

        self.deconv4 = ST_Block(self.nf[5], self.nf[3], self.ns[4], self.ns[3], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv3 = ST_Block(self.nf[3], self.nf[2], self.ns[3], self.ns[2], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv2 = ST_Block(self.nf[2], self.nf[1], self.ns[2], self.ns[1], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv1 = ST_Block(self.nf[1], self.nf[0], self.ns[1], self.ns[0], dim=3, kernel_size=(3, 1), process='d', norm=False)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()

        self.P10 = dict()
        self.P21 = dict()
        self.P32 = dict()
        self.P43 = dict()

        self.tg = dict()
        self.tg1 = dict()
        self.tg2 = dict()

        self.t_P01 = dict()
        self.t_P12 = dict()
        self.t_P23 = dict()

        self.H_inv = dict()
        self.P = dict()

        self.L = dict()
        self.H = dict()
    
    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        
        self.P10[heart_name] = params["P10"]
        self.P21[heart_name] = params["P21"]
        self.P32[heart_name] = params["P32"]
        self.P43[heart_name] = params["P43"]

        self.tg[heart_name] = params["t_bg"]
        self.tg1[heart_name] = params["t_bg1"]
        self.tg2[heart_name] = params["t_bg2"]

        self.t_P01[heart_name] = params["t_P01"]
        self.t_P12[heart_name] = params["t_P12"]
        self.t_P23[heart_name] = params["t_P23"]

        self.H_inv[heart_name] = params["H_inv"]
        self.P[heart_name] = params["P"]

        self.H[heart_name] = params["H"]  # forward matrix
        self.L[heart_name] = params["L"]  # Laplacian matrix
    
    def encode(self, data, heart_name):
        batch_size = data.shape[0]
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            data.view(batch_size, -1, self.nf[0], self.ns[0]), self.tg[heart_name].edge_index, self.tg[heart_name].edge_attr  # (1230*bs) X f[0]
        x = self.conv1(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[2] * self.ns[2])
        x = torch.matmul(self.t_P01[heart_name], x)
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], self.ns[2]), self.tg1[heart_name].edge_index, self.tg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[3] * self.ns[3])
        x = torch.matmul(self.t_P12[heart_name], x)
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], self.ns[3]), self.tg2[heart_name].edge_index, self.tg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[4] * self.ns[4])
        x = torch.matmul(self.t_P23[heart_name], x)
        x = x.view(batch_size, -1, self.nf[4], self.ns[4])

        # latent
        x = x.permute(0, 2, 1, 3).contiguous()
        x = F.elu(self.fce1(x), inplace=True)

        z = self.fce21(x)
        return z
    
    def inverse(self, z, heart_name):
        batch_size = z.shape[0]
        x = z.view(batch_size, self.latent_feature, -1, self.latent_sequence)

        x = x.permute(0, 2, 1, 3).contiguous()
        edge_index, edge_attr = self.H_inv[heart_name].edge_index, self.H_inv[heart_name].edge_attr
        
        num_heart, num_torso = self.P[heart_name].shape[0], self.P[heart_name].shape[1]
        
        x_bin = torch.zeros(batch_size, num_heart, self.latent_feature, self.latent_sequence).to(device)
        x_bin = torch.cat((x_bin, x), 1)
        
        x_bin = x_bin.permute(3, 0, 1, 2).contiguous()
        x_bin = x_bin.view(-1, self.latent_feature)
        edge_index, edge_attr = expand(batch_size, num_heart + num_torso, self.latent_sequence, edge_index, edge_attr)

        x_bin = self.trans(x_bin, edge_index, edge_attr)
        x_bin = x_bin.view(self.latent_sequence, batch_size, -1, self.latent_feature)
        x_bin = x_bin.permute(1, 2, 3, 0).contiguous()
        
        x_bin = x_bin[:, 0:-num_torso, :, :]
        x_bin = x_bin.permute(0, 2, 1, 3).contiguous()
        return x_bin
    
    def decode(self, z, heart_name):
        """ graph  convolutional decoder
        """
        batch_size = z.shape[0]
        x = F.elu(self.fcd3(z), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.nf[5] * self.ns[4])
        x = torch.matmul(self.P43[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[5], self.ns[4]), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.deconv4(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[3] * self.ns[3])
        x = torch.matmul(self.P32[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], self.ns[3]), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.deconv3(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[2] * self.ns[2])
        x = torch.matmul(self.P21[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], self.ns[2]), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.deconv2(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[1] * self.ns[1])
        x = torch.matmul(self.P10[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], self.ns[1]), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.deconv1(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.ns[0])
        return x
    
    def forward(self, y, heart_name):
        z = self.encode(y, heart_name)
        z_h = self.inverse(z, heart_name)
        x = self.decode(z_h, heart_name)
        return x, None


class BayesianFilter(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_dim,
                 ode_func_type,
                 ode_num_layers,
                 ode_method,
                 rnn_type):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.ode_func_type = ode_func_type
        self.ode_num_layers = ode_num_layers
        self.ode_method = ode_method
        self.rnn_type = rnn_type

        # encoder + inverse
        self.conv1 = Spatial_Block(self.nf[0], self.nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = Spatial_Block(self.nf[2], self.nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = Spatial_Block(self.nf[3], self.nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[6], 1)
        self.fce2 = nn.Conv2d(self.nf[6], latent_dim, 1)

        self.trans = Spline(latent_dim, latent_dim, dim=3, kernel_size=3, norm=False, degree=2, root_weight=False, bias=False)
        
        # Bayesian filter
        self.propagation = Propagation(latent_dim, fxn_type=ode_func_type, num_layers=ode_num_layers, method=ode_method, rtol=1e-5, atol=1e-7)
        self.correction = Correction(latent_dim, rnn_type=rnn_type, dim=3, kernel_size=3, norm=False)

        # decoder
        self.fcd3 = nn.Conv2d(latent_dim, self.nf[5], 1)
        self.fcd4 = nn.Conv2d(self.nf[5], self.nf[4], 1)

        self.deconv4 = Spatial_Block(self.nf[4], self.nf[3], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv3 = Spatial_Block(self.nf[3], self.nf[2], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv2 = Spatial_Block(self.nf[2], self.nf[1], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv1 = Spatial_Block(self.nf[1], self.nf[0], dim=3, kernel_size=(3, 1), process='d', norm=False)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

        self.P10 = dict()
        self.P21 = dict()
        self.P32 = dict()
        self.P43 = dict()

        self.tg = dict()
        self.tg1 = dict()
        self.tg2 = dict()

        self.t_P01 = dict()
        self.t_P12 = dict()
        self.t_P23 = dict()

        self.H_inv = dict()
        self.P = dict()

        self.L = dict()
        self.H = dict()

    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]
        
        self.P10[heart_name] = params["P10"]
        self.P21[heart_name] = params["P21"]
        self.P32[heart_name] = params["P32"]
        self.P43[heart_name] = params["P43"]

        self.tg[heart_name] = params["t_bg"]
        self.tg1[heart_name] = params["t_bg1"]
        self.tg2[heart_name] = params["t_bg2"]

        self.t_P01[heart_name] = params["t_P01"]
        self.t_P12[heart_name] = params["t_P12"]
        self.t_P23[heart_name] = params["t_P23"]

        self.H_inv[heart_name] = params["H_inv"]
        self.P[heart_name] = params["P"]

        self.H[heart_name] = params["H"]  # forward matrix
        self.L[heart_name] = params["L"]  # Laplacian matrix

    def embedding(self, data, heart_name):
        batch_size, seq_len = data.shape[0], data.shape[-1]
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            data.view(batch_size, -1, self.nf[0], seq_len), self.tg[heart_name].edge_index, self.tg[heart_name].edge_attr  # (1230*bs) X f[0]
        x = self.conv1(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.t_P01[heart_name], x)
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.tg1[heart_name].edge_index, self.tg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.t_P12[heart_name], x)
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.tg2[heart_name].edge_index, self.tg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.t_P23[heart_name], x)

        # latent
        x = x.view(batch_size, -1, self.nf[4], seq_len)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = F.elu(self.fce1(x), inplace=True)
        x = torch.tanh(self.fce2(x))

        x = x.permute(0, 2, 1, 3).contiguous()

        # inverse
        edge_index, edge_attr = self.H_inv[heart_name].edge_index, self.H_inv[heart_name].edge_attr
        num_right, num_left = self.P[heart_name].shape[0], self.P[heart_name].shape[1]
        
        x_bin = torch.zeros(batch_size, num_right, self.latent_dim, seq_len).to(device)
        x_bin = torch.cat((x_bin, x), 1)

        x_bin = x_bin.permute(3, 0, 1, 2).contiguous()
        x_bin = x_bin.view(-1, self.latent_dim)
        edge_index, edge_attr = expand(batch_size, num_right + num_left, seq_len, edge_index, edge_attr)

        x_bin = self.trans(x_bin, edge_index, edge_attr)
        x_bin = x_bin.view(seq_len, batch_size, num_left + num_right, -1)
        x_bin = x_bin.permute(1, 2, 3, 0).contiguous()
        
        x_bin = x_bin[:, 0:-num_left, :, :]
        return x_bin
    
    def time_modeling(self, x, heart_name):
        N, V, C, T = x.shape
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr

        x = x.permute(3, 0, 1, 2).contiguous()
        last_h = x[0]

        outputs = []
        outputs.append(last_h.view(1, N, V, C))

        x = x.view(T, N * V, C)
        for t in range(1, T):
            last_h = last_h.view(N, V, -1)

            # Propagation
            last_h = self.propagation(last_h, 1, steps=1)
            # Corrrection
            last_h = last_h.view(N * V, -1)
            h = self.correction(x[t], last_h, edge_index, edge_attr)

            last_h = h
            outputs.append(h.view(1, N, V, C))
        
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.permute(1, 2, 3, 0).contiguous()
        return outputs
    
    def decoder(self, x, heart_name):
        batch_size, seq_len = x.shape[0], x.shape[-1]
        x = x.permute(0, 2, 1, 3).contiguous()

        x = F.elu(self.fcd3(x), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.P43[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[4], seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.deconv4(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.P32[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.deconv3(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.P21[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.deconv2(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[1] * seq_len)
        x = torch.matmul(self.P10[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.deconv1(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, seq_len)
        return x
    
    def physics(self, x_, heart_name):
        LX = torch.matmul(self.L[heart_name], x_)
        y_ = torch.matmul(self.H[heart_name], x_)
        return LX, y_

    def forward(self, y, heart_name):
        embed = self.embedding(y, heart_name)
        z = self.time_modeling(embed, heart_name)
        x = self.decoder(z, heart_name)
        LX, y = self.physics(x, heart_name)
        return (x, LX, y, None, None, None), (None, None, None, None)


class VariationalBF(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_dim,
                 ode_func_type,
                 ode_num_layers,
                 ode_method,
                 rnn_type):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.ode_func_type = ode_func_type
        self.ode_num_layers = ode_num_layers
        self.ode_method = ode_method
        self.rnn_type = rnn_type

        # encoder + inverse
        self.conv1 = Spatial_Block(self.nf[0], self.nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = Spatial_Block(self.nf[2], self.nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = Spatial_Block(self.nf[3], self.nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[6], 1)
        self.fce2 = nn.Conv2d(self.nf[6], latent_dim, 1)

        self.trans = Spline(latent_dim, latent_dim, dim=3, kernel_size=3, norm=False, degree=2, root_weight=False, bias=False)
        
        self.propagation = Propagation(latent_dim, fxn_type=ode_func_type, num_layers=ode_num_layers, method=ode_method, rtol=1e-5, atol=1e-7, stochastic=True)
        self.correction = Correction(latent_dim, rnn_type=rnn_type, dim=3, kernel_size=3, norm=False, stochastic=True)

        # decoder
        self.fcd3 = nn.Conv2d(latent_dim, self.nf[5], 1)
        self.fcd4 = nn.Conv2d(self.nf[5], self.nf[4], 1)

        self.deconv4 = Spatial_Block(self.nf[4], self.nf[3], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv3 = Spatial_Block(self.nf[3], self.nf[2], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv2 = Spatial_Block(self.nf[2], self.nf[1], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv1 = Spatial_Block(self.nf[1], self.nf[0], dim=3, kernel_size=(3, 1), process='d', norm=False)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

        self.P10 = dict()
        self.P21 = dict()
        self.P32 = dict()
        self.P43 = dict()

        self.tg = dict()
        self.tg1 = dict()
        self.tg2 = dict()

        self.t_P01 = dict()
        self.t_P12 = dict()
        self.t_P23 = dict()

        self.H_inv = dict()
        self.P = dict()

        self.L = dict()
        self.H = dict()

    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]
        
        self.P10[heart_name] = params["P10"]
        self.P21[heart_name] = params["P21"]
        self.P32[heart_name] = params["P32"]
        self.P43[heart_name] = params["P43"]

        self.tg[heart_name] = params["t_bg"]
        self.tg1[heart_name] = params["t_bg1"]
        self.tg2[heart_name] = params["t_bg2"]

        self.t_P01[heart_name] = params["t_P01"]
        self.t_P12[heart_name] = params["t_P12"]
        self.t_P23[heart_name] = params["t_P23"]

        self.H_inv[heart_name] = params["H_inv"]
        self.P[heart_name] = params["P"]

        self.H[heart_name] = params["H"]  # forward matrix
        self.L[heart_name] = params["L"]  # Laplacian matrix

    def embedding(self, data, heart_name):
        batch_size, seq_len = data.shape[0], data.shape[-1]
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            data.view(batch_size, -1, self.nf[0], seq_len), self.tg[heart_name].edge_index, self.tg[heart_name].edge_attr  # (1230*bs) X f[0]
        x = self.conv1(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.t_P01[heart_name], x)
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.tg1[heart_name].edge_index, self.tg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.t_P12[heart_name], x)
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.tg2[heart_name].edge_index, self.tg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.t_P23[heart_name], x)

        # latent
        x = x.view(batch_size, -1, self.nf[4], seq_len)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = F.elu(self.fce1(x), inplace=True)
        x = torch.tanh(self.fce2(x))

        x = x.permute(0, 2, 1, 3).contiguous()

        # inverse
        edge_index, edge_attr = self.H_inv[heart_name].edge_index, self.H_inv[heart_name].edge_attr
        num_right, num_left = self.P[heart_name].shape[0], self.P[heart_name].shape[1]
        
        x_bin = torch.zeros(batch_size, num_right, self.latent_dim, seq_len).to(device)
        x_bin = torch.cat((x_bin, x), 1)

        x_bin = x_bin.permute(3, 0, 1, 2).contiguous()
        x_bin = x_bin.view(-1, self.latent_dim)
        edge_index, edge_attr = expand(batch_size, num_right + num_left, seq_len, edge_index, edge_attr)

        x_bin = self.trans(x_bin, edge_index, edge_attr)
        x_bin = x_bin.view(seq_len, batch_size, num_left + num_right, -1)
        x_bin = x_bin.permute(1, 2, 3, 0).contiguous()
        
        x_bin = x_bin[:, 0:-num_left, :, :]
        return x_bin
    
    def reparameterization(self, mu, logvar):
        # std = torch.sqrt(var)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def time_modeling(self, x, heart_name):
        N, V, C, T = x.shape
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr

        x = x.permute(3, 0, 1, 2).contiguous()
        # last_h = x[0]

        mu_p_seq, logvar_p_seq = [], []
        mu_q_seq, logvar_q_seq = [], []
        z_q_seq, z_p_seq = [], []
        # outputs = []
        # outputs.append(last_h.view(1, N, V, C))

        x = x.view(T, N * V, C)
        for t in range(T):
            # Propagation
            if t == 0:
                mu_p, logvar_p = self.propagation.init()
                mu_p = mu_p.expand(N, V, C)
                logvar_p = logvar_p.expand(N, V, C)
                last_h = x[0]
                last_h = last_h.view(N, V, -1)
            else:
                last_h = last_h.view(N, V, -1)
                mu_p, logvar_p = self.propagation(last_h, 1, steps=1)

            # mu_p, var_p = self.propagation(last_h, 1, steps=1)
            mu_p = torch.clamp(mu_p, min=-100, max=85)
            logvar_p = torch.clamp(logvar_p, min=-100, max=85)
            z_p = self.reparameterization(mu_p, logvar_p)

            # Corrrection
            z_p = z_p.view(N * V, -1)
            mu_q, logvar_q = self.correction(x[t], z_p, edge_index, edge_attr)
            mu_q = torch.clamp(mu_q, min=-100, max=85)
            logvar_q = torch.clamp(logvar_q, min=-100, max=85)

            # mu_p = mu_p.view(N * V, -1)
            # logvar_p = logvar_p.view(N * V, -1)
            # mu_q = self.correction(x[t], mu_p, edge_index, edge_attr)
            # logvar_q = self.correction(x[t], logvar_p, edge_index, edge_attr)

            z_q = self.reparameterization(mu_q, logvar_q)

            last_h = z_q
            # outputs.append(z_q.view(1, N, V, C))
            z_q_seq.append(z_q.view(1, N, V, C))
            z_p_seq.append(z_p.view(1, N, V, C))
            mu_p_seq.append(mu_p.view(1, N, V, C))
            logvar_p_seq.append(logvar_p.view(1, N, V, C))
            mu_q_seq.append(mu_q.view(1, N, V, C))
            logvar_q_seq.append(logvar_q.view(1, N, V, C))
        
        # outputs = torch.cat(outputs, dim=0)
        z_q_seq = torch.cat(z_q_seq, dim=0)
        z_p_seq = torch.cat(z_p_seq, dim=0)
        mu_p_seq = torch.cat(mu_p_seq, dim=0)
        logvar_p_seq = torch.cat(logvar_p_seq, dim=0)
        mu_q_seq = torch.cat(mu_q_seq, dim=0)
        logvar_q_seq = torch.cat(logvar_q_seq, dim=0)

        # outputs = outputs.permute(1, 2, 3, 0).contiguous()
        z_q_seq = z_q_seq.permute(1, 2, 3, 0).contiguous()
        z_p_seq = z_p_seq.permute(1, 2, 3, 0).contiguous()
        mu_p_seq = mu_p_seq.permute(1, 2, 3, 0).contiguous()
        logvar_p_seq = logvar_p_seq.permute(1, 2, 3, 0).contiguous()
        mu_q_seq = mu_q_seq.permute(1, 2, 3, 0).contiguous()
        logvar_q_seq = logvar_q_seq.permute(1, 2, 3, 0).contiguous()

        return z_p_seq, z_q_seq, mu_p_seq, logvar_p_seq, mu_q_seq, logvar_q_seq
    
    def decoder(self, x, heart_name):
        batch_size, seq_len = x.shape[0], x.shape[-1]
        x = x.permute(0, 2, 1, 3).contiguous()

        x = F.elu(self.fcd3(x), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.P43[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[4], seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.deconv4(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.P32[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.deconv3(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.P21[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.deconv2(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[1] * seq_len)
        x = torch.matmul(self.P10[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.deconv1(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, seq_len)
        return x
    
    def physics(self, x_, heart_name):
        LX = torch.matmul(self.L[heart_name], x_)
        y_ = torch.matmul(self.H[heart_name], x_)
        return LX, y_

    def forward(self, y, heart_name):
        embed = self.embedding(y, heart_name)
        z_p_seq, z_q_seq, mu_p_seq, logvar_p_seq, mu_q_seq, logvar_q_seq = self.time_modeling(embed, heart_name)
        # decode from p
        x_p = self.decoder(z_p_seq, heart_name)
        LX_p, y_p = self.physics(x_p, heart_name)
        # decode from q
        x_q = self.decoder(z_q_seq, heart_name)
        LX_q, y_q = self.physics(x_q, heart_name)
        return (x_q, LX_q, y_q, x_p, LX_p, y_p), (mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq)
