import torch as th
from torch import nn
import numpy as np
import einops

from .pos_emb import *
    
class FFN_conv(nn.Module):
    '''
    Feed-forward module
    :param channels: In- and output channels
    :param r: Channel expansion factor
    '''
    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        self.l1 = nn.Conv2d(channels, channels * r, 1)
        self.l2 = nn.Conv2d(channels * r, channels, 1)
        self.act = nn.ReLU()
        
    def forward(self, x: th.Tensor):
        return self.l2(self.act(self.l1(x)))       
    
class AFNO(nn.Module):
    '''
    Adaptive-Fourier-Neural Operator module
    :param channels: In- and output channels
    :param num_heads: Number of projection sub-spaces
    :param r: Channel expansion factor
    :param scale: Re-scale the initial weight distribution of the BlockMLP
    :param keep_modes: Fraction of Fourier modes to keep
    :param lmbd: Lambda parameter for the soft-shrinkage operation
    '''
    def __init__(self, 
                 channels: int, 
                 num_heads: int, 
                 r: int = 4, 
                 lmbd: float = 1e-2, 
                 keep_modes: float = 1., 
                 scale: float = 2e-2):
        super().__init__()
        size_heads = channels // num_heads
        self.num_heads = num_heads
        self.keep_modes = keep_modes
        self.r = r
        #first fully connected block has dimensions: (real, img), num_heads, size_heads, size_latent
        self.w_1 = nn.Parameter(scale * th.randn(2, num_heads, size_heads, r * size_heads))
        self.b_1 = nn.Parameter(scale * th.randn(2, num_heads, r * size_heads))
        #first fully connected block has dimensions: (real, img), num_heads, size_latent, size_heads
        self.w_2 = nn.Parameter(scale * th.randn(2, num_heads, r * size_heads, size_heads))
        self.b_2 = nn.Parameter(scale * th.randn(2, num_heads, size_heads))
        #relu
        self.act = nn.ReLU()
        #softshrink
        self.softshrink = nn.Softshrink(lmbd)
        
    def forward(self, x, grid_size = None):
        if grid_size:
            height, width = grid_size 
        else:
            height = width = th.sqrt(x.shape[1])
        #tokens to grid
        #x = einops.rearrange(x, 'b (h w) c -> b h w c', h = height, w = width)
        #to fourier space
        x = th.fft.rfft2(x, norm="ortho")
        #split heads
        x = einops.rearrange(x, 'b (nh sh) h w -> b h w nh sh', nh = self.num_heads)
        #mix it!
        x = self._block_mlp(x)
        #apply soft shrinkage
        x = self.softshrink(x)
        x = th.view_as_complex(x)
        #merge heads
        x = einops.rearrange(x, 'b h w nh sh -> b (nh sh) h w')
        #to data space
        x = th.fft.irfft2(x, s = (height, width), norm="ortho")
        #back to sequence
        #x = einops.rearrange(x, 'b h w c -> b (h w) c')
        return x
    
    def _block_mlp(self, x: th.Tensor):
        b, h, w, nh, sh = x.shape
        #hard fourier-mode thresholding
        total_modes = (h*w)//2 + 1
        kept_modes = int(total_modes * self.keep_modes)
        #intermediate matrices for real and imaginary parts
        x_1_real = th.zeros((b, h, w, nh, self.r * sh), device = x.device)
        x_1_imag = th.zeros((b, h, w, nh, self.r * sh), device = x.device)
        x_2_real = th.zeros(x.shape, device = x.device)
        x_2_imag = th.zeros(x.shape, device = x.device)
        #mlp projection for real and imaginary parts
        x_1_real[:, :, :kept_modes] = self.act(
            th.einsum('...s, ...sc -> ...c', x[:, :, :kept_modes].real, self.w_1[0]) - \
            th.einsum('...s, ...sc -> ...c', x[:, :, :kept_modes].imag, self.w_1[1]) + \
            self.b_1[0])
        x_1_imag[:, :, :kept_modes] = self.act(
            th.einsum('...s, ...sc -> ...c', x[:, :, :kept_modes].imag, self.w_1[1]) + \
            th.einsum('...s, ...sc -> ...c', x[:, :, :kept_modes].real, self.w_1[0]) + \
            self.b_1[1])
        x_2_real[:, :, :kept_modes] = \
            th.einsum('...c, ...cs -> ...s', x_1_real[:, :, :kept_modes], self.w_2[0]) - \
            th.einsum('...c, ...cs -> ...s', x_1_imag[:, :, :kept_modes], self.w_2[1]) + \
            self.b_2[0]
        x_2_imag[:, :, :kept_modes] = \
            th.einsum('...c, ...cs -> ...s', x_1_imag[:, :, :kept_modes], self.w_2[1]) + \
            th.einsum('...c, ...cs -> ...s', x_1_real[:, :, :kept_modes], self.w_2[0]) + \
            self.b_2[1]
        return th.stack([x_2_real, x_2_imag], dim = -1)


class EncoderBlock(nn.Module):
    '''
    Wrapper for the transformer block structure. Spatial mixing -> channel mixing.
    Additional kwargs can be handed to the spatial mixing module.
    '''
    def __init__(self, 
                 channels: int, 
                 num_heads: int,
                 r: int, 
                 SpatialMixer: nn.Module,
                 ChannelMixer: nn.Module,
                 NormLayer: nn.Module = nn.LayerNorm,
                 **kwargs: dict,
                ):
        super().__init__()
        
        self.spatial_mixing = SpatialMixer(channels, num_heads, **kwargs)
        self.channel_mixing = ChannelMixer(channels, r = r)
        self.norm1 = NormLayer(channels)
        self.norm2 = NormLayer(channels)
        
    def forward(self, x: th.Tensor, **kwargs):
        x = x + self.spatial_mixing(self.norm1(x), **kwargs)
        x = x + self.channel_mixing(self.norm2(x))
        return x

class ViT_AFNO(nn.Module):

    def __init__(self, 
                 channels: int, 
                 input_channels: int, 
                 output_channels: int,
                 num_heads: int, 
                 patch_size: int,
                 img_size: (int, int), 
                 network_depth: int, 
                 r: int = 4, 
                 NormLayer: nn.Module = nn.BatchNorm2d,
                 **kwargs):
        '''
        :param channels: Internal channels
        :param input_channels: Input channels
        :param output_channels: Output channels
        :param num_heads: How many independent sub-spaces to use for spatial mixing
        :param patch_size: Size of the (non-overlapping) patches.
        :param img_size: Size of the expected image
        :param network_depth: How many Transformer blocks to use
        :param r: Channel expansion factor. Defaults to 4
        :param kwargs: Additional kwargs to hand to the spatial_mixer.
        '''
        super().__init__()
        self.k = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        
        pos_embed = get_2d_sincos_pos_embed(embed_dim = channels, grid_size = self.grid_size)
        self.position_embedding = nn.Parameter(
            th.from_numpy(pos_embed).float().view(1, channels, self.grid_size[0], self.grid_size[1]), 
            requires_grad = False)
                
        self.input_embedding = nn.Conv2d(input_channels, 
                                         channels, 
                                         bias = False,
                                         kernel_size = patch_size, 
                                         stride = patch_size)
        self.input_embedding.weight.requires_grad = False
        self.norm = NormLayer(channels)
        
        self.blocks = nn.ModuleList([EncoderBlock(channels, 
                                                  num_heads, 
                                                  r = r, 
                                                  NormLayer = nn.BatchNorm2d,
                                                  ChannelMixer = FFN_conv,
                                                  SpatialMixer = AFNO,
                                                  **kwargs) 
                                     for i in range(network_depth)])
        
        self.output_embedding = nn.ConvTranspose2d(channels, output_channels, kernel_size = patch_size, stride = patch_size)
        
    def _forward(self, x: th.Tensor):
        '''
        Input x is assumed to be a batch of images shaped (batch, height, width, data_channels)
        '''
        bs,cs, height, width = x.shape
        if (height // self.k, width // self.k) != self.grid_size:
            self.resize_grid((height, width))
            print('Resizing grid...')
        x = self.input_embedding(x)
        x = self.norm(x) + self.position_embedding
        for block in self.blocks:
            x = block(x, grid_size = self.grid_size)
        x = self.output_embedding(x)
        return x
    
    def forward(self, x_hat: th.Tensor, x_aux: th.Tensor = None, horizon: int = 1):
        '''
        :param x_hat: Input x is assumed to be a batch of images shaped (batch, data_channels, time, height, width)
        :param x_aux: Static input x is assumed to be a batch of static information to add at each time step
        :param horizon: How many steps of closed-loop to perform
        '''
        context, height, width = x_hat.shape[-3:]
        if (height // self.k, width // self.k) != self.grid_size:
            self.resize_grid((height, width))
            print('Resizing grid...')
        y = []
        for t in range(0, horizon, context):
            #add static information
            x = th.cat((x_hat, 
                        x_aux[:, :, t: t + context]), dim = 1) if x_aux is not None else x_hat
            x = einops.rearrange(x, 'b c t h w -> b (c t) h w') #time2depth
            x = self._forward(x)
            x_hat = einops.rearrange(x, 'b (c t) h w -> b c t h w', t = context) #depth2time
            y.append(x_hat)
        return th.cat(y, dim = 2)
        
    def resize_grid(self, new_img_size: (int, int)):
        '''
        Can be used to adapt the position embedding and internal grid to a new image size
        '''
        self.grid_size = (new_img_size[0] // self.k, new_img_size[1] // self.k)
        self.position_embedding = nn.Parameter(nn.functional.interpolate(
            self.position_embedding, self.grid_size, mode = 'bilinear', align_corners = False),
                                              requires_grad = False)

