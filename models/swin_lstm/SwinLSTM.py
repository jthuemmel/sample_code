import einops
import torch as th
from torch import nn


class SwinLSTM(nn.Module):
    def __init__(self, 
                 h_channels: int, 
                 k_conv: int = 7, 
                 output_activation: nn.Module = nn.GELU(),
                 dropout: float = 0.):
        '''
        :param x_channels: Input channels   
        :param h_channels: Latent state channels
        :param kernel_size: Convolution kernel size
        :param activation_fn: Output activation function
        '''
        super().__init__()
        self.spatial_mixing = nn.Conv2d(h_channels, h_channels, kernel_size = k_conv, padding='same', groups= h_channels)
        self.norm = nn.GroupNorm(4, h_channels, affine = False)
        self.channel_mixing = nn.Conv2d(h_channels, 4 * h_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.to_output = output_activation
        
        nn.init.dirac_(self.spatial_mixing.weight)
        nn.init.dirac_(self.channel_mixing.weight)
        nn.init.ones_(self.channel_mixing.bias[:h_channels]) * 2

    def forward(self, x, hidden, cell, context = None):
        '''
        LSTM forward pass
        :param x: Input
        :param hidden: Hidden state
        :param cell: Cell state
        :param context: Context information
        '''
        z = x + hidden if x is not None else hidden
        #Spatial mixing
        z = self.spatial_mixing(z)
        #Feature-wise Linear Modulation
        scale, bias = context.chunk(2, dim = 1) if context is not None else (0, 0)
        #Apply modulation
        z = self.norm(z) * (1 + scale) + bias
        #Channel mixing
        z = self.channel_mixing(z)
        #Gate splitting
        f, i, g, o = einops.rearrange(z, 'b (gates c) h w -> gates b c h w', gates = 4) #forget gate, input gate, proposal state, output gate
        #calculate new cell state
        cell = th.sigmoid(f) * cell + th.sigmoid(i) * self.dropout(th.tanh(g))
        #calculate new hidden state
        hidden = th.sigmoid(o) * self.to_output(cell)
        return hidden, cell
    

class Net(nn.Module):
    def __init__(self,
                 data_dim: int,
                 latent_dim: int,
                 patch_size: tuple = (4,4),
                 num_layers: int = 2,
                 conditioning_num: int = 12,
                 k_conv: int = 7,
                 dropout: float = 0.1,
                 predict_var: bool = True
                 ) -> None:
        '''
        ElNet: The ElNet model.
        Args:
            data_dim: The number of input channels
            latent_dim: The dimension of the latent space.
            patch_size: The size of the latent patches.
            num_layers: The number of LSTM layers in the model.
            k_conv: The kernel size of the convolutional layers.
            conditioning_num: The number of conditioning classes.'''
        
        super().__init__()
        #define model attributes
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        #define model layers
        self.processor = nn.ModuleDict()
        self.processor['to_latent'] = nn.Conv3d(data_dim, latent_dim, kernel_size= (1 , *patch_size), stride = (1, *patch_size))
        
        for i in range(num_layers):
            self.processor[f'encoder_lstm_{i}'] = SwinLSTM(latent_dim, k_conv = k_conv, dropout=dropout)
            self.processor[f'decoder_lstm_{i}'] = SwinLSTM(latent_dim, k_conv = k_conv, dropout=dropout)

        self.processor['to_mean'] = nn.ConvTranspose3d(latent_dim, data_dim, kernel_size= (1 , *patch_size), stride = (1, *patch_size))
        self.processor['to_logvar'] = nn.ConvTranspose3d(latent_dim, data_dim, kernel_size= (1 , *patch_size), stride = (1, *patch_size)) if predict_var else None
        
        self.embedding = nn.Embedding(conditioning_num, 2 * latent_dim)
        #

    def forward(self, x, context):
        '''
        Forward pass of the model.
        Args:
            x: Input tensor.
            context: Context tensor.
            '''
        batch, _, history, height, width = x.shape
        h, w = height // self.patch_size[0], width // self.patch_size[1]
        horizon = context.shape[1] - history 
        assert horizon > 0, 'Context length must be greater than history length'
        #initialize hidden and cell states  
        hidden = [th.zeros((batch, self.latent_dim, h, w), device = x.device) for _ in range(self.num_layers)]
        cell = [th.zeros((batch, self.latent_dim, h, w), device = x.device) for _ in range(self.num_layers)]
        #get conditioning 
        conditioning = self.embedding(context)
        #encoder
        patches = self.processor['to_latent'](x)
        for t in range(history):
            z = patches[:, :, t]
            for i in reversed(range(self.num_layers)):
                hidden[i], cell[i] = self.processor[f'encoder_lstm_{i}'](z, hidden[i], cell[i], context = conditioning[:, t, :, None, None]) 
                z = z + hidden[i]
        #decoder
        out = th.zeros((batch, self.latent_dim, horizon, h, w), device = x.device)
        for t in range(horizon):
            z = None
            for i in range(self.num_layers):
                hidden[i], cell[i] = self.processor[f'decoder_lstm_{i}'](z, hidden[i], cell[i], context = conditioning[:, t + history, :, None, None])    
                z = z + hidden[i] if z is not None else hidden[i]
            out[:, :, t] = z
        frcst = (self.processor['to_mean'](out), self.processor['to_logvar'](out)) if self.processor['to_logvar'] is not None else (self.processor['to_mean'](out), None)
        return frcst
