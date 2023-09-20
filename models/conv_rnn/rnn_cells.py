import torch as th
import torch.nn as nn
from torch import cat, tanh, Tensor, sigmoid

class LSTMCell2d(nn.Module):
    '''LSTMCell for 2d inputs'''
    def __init__(self, 
                 x_channels: int, 
                 h_channels: int,
                 kwargs: dict = {'kernel_size' : 1},
                 rnn_act: callable = None, 
                 norm: str = 'none'):
        '''
        :param x_channels: Input channels
        :param h_channels: Latent state channels
        :param tk_kwargs: (Optional) Kwargs of the convolution kernel. Default to pointwise.
        :param rnn_act: (Optional) Activation function to use inside the LSTM. Default to tanh.
        '''
        super().__init__()
            
        self.phi = rnn_act or tanh    
        self.pk = nn.Conv2d(
            in_channels = x_channels + h_channels, 
            out_channels = 4 * h_channels,
            **kwargs)
        if norm == 'layer':
            self.norm = nn.GroupNorm(num_groups = 1, num_channels = x_channels)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(num_features = x_channels)
        else:
            self.norm = nn.Identity()
    def forward(self, x, h, c) -> Tensor:
        '''
        LSTM forward pass
        :param x: Input
        :param h: Hidden state
        :param c: Cell state
        '''
        if x is not None:
            x = self.norm(x)
            z = th.cat((x, h), dim = 1) 
        else:
            z = h
        i, f, o, g = self.pk(z).chunk(chunks = 4, axis = 1)
        c = sigmoid(f) * c + sigmoid(i) * self.phi(g)
        h = sigmoid(o) * self.phi(c)
        return h, c
    
class GRUCell2d(nn.Module):
    '''GRUCell for 2d inputs'''
    def __init__(self, 
                 x_channels: int, 
                 h_channels: int,
                 kwargs: dict = {'kernel_size' : 1},
                 rnn_act: callable = None, 
                 norm: str = None):
        '''
        :param x_channels: Input channels
        :param h_channels: Latent state channels
        :param tk_kwargs: (Optional) Kwargs of the convolution kernel. Default to pointwise.
        :param rnn_act: (Optional) Activation function to use inside the LSTM. Default to tanh.
        '''
        super().__init__()
        self.phi = rnn_act or tanh    
        self.pk_c = nn.Conv2d(
            in_channels = x_channels + h_channels, 
            out_channels = h_channels,
            **kwargs)
        self.pk_g = nn.Conv2d(
            in_channels = x_channels + h_channels, 
            out_channels = 2 * h_channels, 
            **kwargs)
        if norm == 'layer':
            self.norm = nn.GroupNorm(num_groups = 1, num_channels = x_channels)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(num_features = x_channels)
        else:
            self.norm = nn.Identity()
            
    def forward(self, x, h) -> Tensor:
        '''
        GRU forward pass
        :param x: Input
        :param h: Hidden state
        '''
        if x is not None:
            x = self.norm(x)
            z = th.cat((x, h), dim = 1) 
        else:
            z = h
        r, u = sigmoid(self.pk_g(z)).chunk(chunks = 2, axis = 1)
        h_ = r * h
        z_ = cat((x, h_), axis = 1) if x is not None else h_
        c = self.phi(self.pk_c(z_))
        h = (1 - u) * c + u * h
        return h