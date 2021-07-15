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

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


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


class DKF(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_feature,
                 rnn_type,
                 orthogonal_init,
                 rnn_bidirection,
                 reverse_input,
                 rnn_dim,
                 train_init,
                 z_dim,
                 transition_dim,
                 sample):
        super().__init__()
        self.nf = num_channel
        self.latent_feature = latent_feature
        self.rnn_type = rnn_type
        self.orthogonal_init = orthogonal_init
        self.rnn_bidirection = rnn_bidirection
        self.reverse_input = reverse_input
        self.rnn_dim = rnn_dim
        self.train_init = train_init
        self.z_dim = z_dim
        self.transition_dim = transition_dim
        self.sample = sample

        # encoder + inverse
        self.conv1 = Spatial_Block(self.nf[0], self.nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = Spatial_Block(self.nf[2], self.nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = Spatial_Block(self.nf[3], self.nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[5], 1)
        self.fce2 = nn.Conv2d(self.nf[5], latent_feature, 1)

        self.trans = Spline(latent_feature, rnn_dim, dim=3, kernel_size=3, norm=False, degree=2, root_weight=False, bias=False)
        
        # RNN encoder
        self.rnn_encoder = RnnEncoder(rnn_dim, rnn_dim, dim=3, kernel_size=3, norm=False,
                                      n_layer=1, nonlin='relu', rnn_type=rnn_type, bd=rnn_bidirection,
                                      reverse_input=reverse_input, orthogonal_init=orthogonal_init)
        # combiner
        self.combiner = Combiner(z_dim, rnn_dim)
        # transition
        self.transition = Transition(z_dim, transition_dim, identity_init=True)
        # initialize hidden states
        self.mu_p_0, self.logvar_p_0 = self.transition.init_z_0(trainable=train_init)
        self.z_q_0 = self.combiner.init_z_q_0(trainable=train_init)

        # decoder
        self.fcd3 = nn.Conv2d(z_dim, self.nf[5], 1)
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
        
        x_bin = torch.zeros(batch_size, num_right, self.latent_feature, seq_len).to(device)
        x_bin = torch.cat((x_bin, x), 1)

        x_bin = x_bin.permute(3, 0, 1, 2).contiguous()
        x_bin = x_bin.view(-1, self.latent_feature)
        edge_index, edge_attr = expand(batch_size, num_right + num_left, seq_len, edge_index, edge_attr)

        x_bin = self.trans(x_bin, edge_index, edge_attr)
        x_bin = x_bin.view(seq_len, batch_size, num_left + num_right, -1)
        x_bin = x_bin.permute(1, 2, 3, 0).contiguous()
        
        x_bin = x_bin[:, 0:-num_left, :, :]
        return x_bin
    
    def reparameterization(self, mu, logvar):
        if not self.sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def time_modeling(self, x, heart_name):
        batch_size, V, T = x.shape[0], x.shape[1], x.shape[-1]
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr
        h_rnn = self.rnn_encoder(x, edge_index, edge_attr)

        # initial condition
        z_q_0 = self.z_q_0.expand(batch_size, V, self.z_dim)  # q(z_0)
        mu_p_0 = self.mu_p_0.expand(batch_size, V, self.z_dim)
        logvar_p_0 = self.logvar_p_0.expand(batch_size, V, self.z_dim)
        z_prev = z_q_0

        mu_q_seq = torch.zeros([batch_size, V, self.z_dim, T]).to(x.device)
        logvar_q_seq = torch.zeros([batch_size, V, self.z_dim, T]).to(x.device)
        mu_p_seq = torch.zeros([batch_size, V, self.z_dim, T]).to(x.device)
        logvar_p_seq = torch.zeros([batch_size, V, self.z_dim, T]).to(x.device)
        z_q_seq = torch.zeros([batch_size, V, self.z_dim, T]).to(x.device)
        z_p_seq = torch.zeros([batch_size, V, self.z_dim, T]).to(x.device)

        for t in range(T):
            # q(z_t | z_{t-1}, x_{t:T})
            mu_q, logvar_q = self.combiner(h_rnn[:, :, :, t], z_prev, rnn_bidirection=self.rnn_bidirection) 
            # print('Reconstruction q: {}, mean max: {}, var max: {} \n mean avg: {}, var avg: {}'.format(t, torch.max(mu_q), torch.max(logvar_q), torch.mean(mu_q), torch.mean(logvar_q)))
            zt_q = self.reparameterization(mu_q, logvar_q)
            z_prev = zt_q

            # p(z_{t+1} | z_t)
            mu_p, logvar_p = self.transition(z_prev)
            # print('Reconstruction p: {}, mean max: {}, var max: {} \n mean avg: {}, var avg: {}'.format(t, torch.max(mu_p), torch.max(logvar_p), torch.mean(mu_p), torch.mean(logvar_p)))
            zt_p = self.reparameterization(mu_p, logvar_p)

            mu_q_seq[:, :, :, t] = mu_q
            logvar_q_seq[:, :, :, t] = logvar_q
            z_q_seq[:, :, :, t] = zt_q
            mu_p_seq[:, :, :, t] = mu_p
            logvar_p_seq[:, :, :, t] = logvar_p
            z_p_seq[:, :, :, t] = zt_p
        
        # move one step ahead for p(z_t | z_{t-1})
        mu_p_0 = mu_p_0.reshape(batch_size, V, self.z_dim, 1)
        mu_p_seq = torch.cat([mu_p_0, mu_p_seq[:, :, :, :-1]], dim=-1)
        logvar_p_0 = logvar_p_0.reshape(batch_size, V, self.z_dim, 1)
        logvar_p_seq = torch.cat([logvar_p_0, logvar_p_seq[:, :, :, :-1]], dim=-1)
        z_p_0 = self.reparameterization(mu_p_0, logvar_p_0)
        z_p_seq = torch.cat([z_p_0, z_p_seq[:, :, :, :-1]], dim=-1)

        return z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq

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
        z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq = self.time_modeling(embed, heart_name)
        x_q = self.decoder(z_q_seq, heart_name)
        LX_q, y_q = self.physics(x_q, heart_name)

        x_p = self.decoder(z_p_seq, heart_name)
        LX_p, y_p = self.physics(x_p, heart_name)
        return (x_q, LX_q, y_q, x_p, LX_p, y_p), (mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq)
