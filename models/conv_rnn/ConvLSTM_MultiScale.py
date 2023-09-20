import torch as th
from einops import rearrange, repeat
from torch import nn

   
class DownConv2d(nn.Module):
    '''
    Strided convolution down-sampling layer
    '''
    def __init__(self, in_channels: int, scale: int, k: int = 3, act_layer = nn.ReLU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels = in_channels * scale, 
                      kernel_size = k, 
                      stride = scale, 
                      padding = k // 2),
            act_layer()
        )
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.conv(x)
    
class UpConv2d(nn.Module):
    '''
    Strided convolution up-sampling layer
    '''
    def __init__(self, in_channels: int, scale: int, k: int = 3, act_layer = nn.ReLU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 
                               out_channels = in_channels.div(scale, rounding_mode = 'trunc'), 
                               kernel_size = k, 
                               stride = scale, 
                               padding = k // 2, 
                               output_padding = k % 2 if scale > 1 else 0),
            act_layer()
        )
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.conv(x)
    
class LSTMCell2d(nn.Module):
    '''LSTMCell for 2d inputs'''
    def __init__(self, 
                 x_channels: int, 
                 h_channels: int,
                 k: int = 3, 
                 T_max: int = -1):
        '''
        :param x_channels: Input channels
        :param h_channels: Latent state channels
        :param k: Size of the convolution kernel
        :param T_max: Bias initialisation parameter
        '''
        super().__init__()
            
        self.conv = nn.Sequential(
            nn.Conv2d(
            in_channels = x_channels + h_channels, 
            out_channels = 4 * h_channels,
            kernel_size = k,
            padding = 'same'),
            nn.GroupNorm(num_channels = 4 * h_channels, num_groups = 4)
        )
        
        if isinstance(T_max, int) and T_max > 1:
            self._chrono_init_(T_max)
        
    def forward(self, x: th.Tensor, h: th.Tensor, c: th.Tensor) -> th.Tensor:
        '''
        LSTM forward pass
        :param x: Input
        :param h: Hidden state
        :param c: Cell state
        '''
        z = th.cat((x, h), dim = 1) if x is not None else h
        z = self.conv(z)
        i, f, o, g = z.chunk(chunks = 4, axis = 1)
        c = th.sigmoid(f) * c + th.sigmoid(i) * th.tanh(g)
        h = th.sigmoid(o) * th.tanh(c)
        return h, c
    
    
    def _chrono_init_(self, T_max):
        '''
        Bias initialisation based on: https://arxiv.org/pdf/1804.11188.pdf
        :param T_max: The largest time scale we want to capture
        '''
        b = self.conv[0].bias
        h = len(b) // 4
        b.data.fill_(0)
        b.data[h:2*h] = th.log(nn.init.uniform_(b.data[h:2*h], 1, T_max - 1))
        b.data[:h] = - b.data[h:2*h]
        

class MultiScaleForecaster_LSTM(nn.Module):
    '''
    A multi-scale ConvLSTM forecasting architecture. 
    '''
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int,
                 scale: int = 1, 
                 stages: int = 0,
                 act_layer = nn.ReLU):
        '''
        :param in_channels: Number of dynamic input channels
        :param hidden_channels: Number of hidden channels
        :param out_channels: Number of dynamic output channels
        :param scale: Scaling applied to the resolution per block
        :param stages: Number of stages that scale the resolution
        :param act_layer: Activation function for conv blocks. Has to be a nn.Module.
        '''
        super().__init__()
        assert scale % 2 == 0 or scale == 1, 'Odd scales lead to sizing issues'
        self.stages = stages
        self.scale = scale
        self.h_channels = hidden_channels * scale ** th.arange(stages + 1)
        
        self.input_embedding = nn.Sequential(
            nn.Conv2d(in_channels, self.h_channels[0], 3, padding = 'same'),
            act_layer()
        )
        self.encoder_layers = nn.ModuleDict()
        self.decoder_layers = nn.ModuleDict()
        for i in range(stages):
            self.encoder_layers.add_module(f'process_{i}', 
                                           LSTMCell2d(self.h_channels[i], 
                                                      self.h_channels[i]))
            self.encoder_layers.add_module(f'scale_{i}', 
                                           DownConv2d(self.h_channels[i], 
                                                      scale, 
                                                      act_layer = act_layer))
            self.decoder_layers.add_module(f'process_{i}',
                                           LSTMCell2d(self.h_channels[i], 
                                                      self.h_channels[i]))
            self.decoder_layers.add_module(f'scale_{i}', 
                                           UpConv2d(self.h_channels[i+1], 
                                                    scale, 
                                                    act_layer = act_layer))
        self.encoder_layers.add_module(f'process_{stages}', 
                                       LSTMCell2d(self.h_channels[stages], 
                                                  self.h_channels[stages]))
        self.decoder_layers.add_module(f'process_{stages}', 
                                       LSTMCell2d(0, 
                                                  self.h_channels[stages]))
        self.output_embedding = nn.Sequential(
            nn.Conv2d(self.h_channels[0], self.h_channels[0], 3, padding = 'same'),
            act_layer(),
            nn.Conv2d(self.h_channels[0], out_channels, 1)
        )
        
        
    def forward(self, 
                x_dynamic: th.Tensor, 
                x_static: th.Tensor = None,
                horizon: int = 1) -> th.Tensor:
        '''
        :param x: Input tensor
        :param horizon: Number of frames to generate
        '''
        #shape of x is expected to be:
        bs, channels, context, height, width = x_dynamic.shape
        
        h = [th.zeros((bs, self.h_channels[i], height // self.scale**i, width // self.scale**i), device = x_dynamic.device) 
             for i in range(self.stages + 1)]
        c = [th.zeros((bs, self.h_channels[i], height // self.scale**i, width // self.scale**i), device = x_dynamic.device) 
             for i in range(self.stages + 1)]
        
        x_static = repeat(x_static, 'd h w -> b d h w', b = bs) if x_static is not None else None
        
        for t in range(context):
            x = x_dynamic[:, :, t]
            x = th.cat((x, x_static), dim = 1) if x_static is not None else x
            z = self.input_embedding(x)
            for i in range(self.stages):
                h[i], c[i] = self.encoder_layers[f'process_{i}'](z, h[i], c[i])
                z = self.encoder_layers[f'scale_{i}'](h[i])
            h[-1], c[-1] = self.encoder_layers[f'process_{self.stages}'](z, h[-1], c[-1])
        out = []
        for t in range(horizon):
            h[-1], c[-1] = self.decoder_layers[f'process_{self.stages}'](None, h[-1], c[-1])
            for i in reversed(range(self.stages)):
                z = self.decoder_layers[f'scale_{i}'](h[i + 1])
                h[i], c[i] = self.decoder_layers[f'process_{i}'](z, h[i], c[i])
            out.append(self.output_embedding(z))
        #stack to [bs, channels, horizon, height, width] format
        output = th.stack(out, dim = 2) 
        return output
    
