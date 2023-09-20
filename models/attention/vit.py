import einops
import numpy as np
import torch as th
from torch import nn

from .pos_emb import *


class FFN(nn.Module):
  '''
  Feed-forward module
  :param channels: In- and output channels
  :param r: Channel expansion factor
  '''

  def __init__(self, channels: int, r: int = 4, **kwargs):
    super().__init__()
    self.l1 = nn.Linear(channels, channels * r)
    self.l2 = nn.Linear(channels * r, channels)
    self.act = nn.ReLU()

  def forward(self, x: th.Tensor):
    return self.l2(self.act(self.l1(x)))


class MHSA(nn.Module):
  '''
  Multi-head self-attention module
  :param channels: In- and output channels
  :param num_heads: Number of projection sub-spaces
  '''

  def __init__(self, channels: int, num_heads: int, **kwargs):
    super().__init__()
    self.scale = (channels // num_heads) ** -0.5
    self.num_heads = num_heads
    self.to_qkv = nn.Linear(channels, channels * 3, bias=False)
    self.to_out = nn.Linear(channels, channels)

  def forward(self, x: th.Tensor, **kwargs):
    # map to qkv
    qkv = self.to_qkv(x).chunk(chunks=3, dim=-1)
    # split qkv into heads
    q, k, v = map(lambda t: einops.rearrange(t, 'b n (h c) -> b h n c', h=self.num_heads), (qkv))
    # apply temperature scaling to q
    q = q * self.scale
    # compute the attention scores per head
    scores = th.einsum('b h n c, b h m c -> b h n m', q, k)
    # apply the softmax
    attn = scores.softmax(dim=-1)
    # weigh the values by the attention matrix
    v_hat = th.einsum('b h n m, b h m c -> b h n c', attn, v)
    # merge heads
    x = einops.rearrange(v_hat, 'b h n c -> b n (h c)')
    # map to output
    x = self.to_out(x)
    return x


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
    self.channel_mixing = ChannelMixer(channels, r=r)
    self.norm1 = NormLayer(channels)
    self.norm2 = NormLayer(channels)

  def forward(self, x: th.Tensor, **kwargs):
    x = x + self.spatial_mixing(self.norm1(x), **kwargs)
    x = x + self.channel_mixing(self.norm2(x))
    return x


class ViT_MHSA(nn.Module):

  def __init__(self,
               channels: int,
               input_channels: int,
               output_channels: int,
               num_heads: int,
               patch_size: int,
               img_size: (int, int),
               network_depth: int,
               r: int = 4,
               NormLayer: nn.Module = nn.LayerNorm,
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

    pos_embed = get_2d_sincos_pos_embed(embed_dim=channels, grid_size=self.grid_size)
    self.position_embedding = nn.Parameter(th.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False)

    self.input_embedding = nn.Conv2d(input_channels,
                                     channels,
                                     bias=False,
                                     kernel_size=patch_size,
                                     stride=patch_size)
    self.input_embedding.weight.requires_grad = False  # freeze the input layer

    self.norm = NormLayer(channels)
    self.blocks = nn.ModuleList([EncoderBlock(channels,
                                              num_heads,
                                              r=r,
                                              NormLayer=nn.LayerNorm,
                                              ChannelMixer=FFN,
                                              SpatialMixer=MHSA,
                                              **kwargs)
                                 for i in range(network_depth)])

    self.output_embedding = nn.ConvTranspose2d(channels, output_channels, kernel_size=patch_size, stride=patch_size)

  def _forward(self, x: th.Tensor):
    '''
    Input x is assumed to be a batch of images shaped (batch, data_channels, height, width)
    '''
    height, width = x.shape[-2:]
    if (height // self.k, width // self.k) != self.grid_size:
      self.resize_grid((height, width))
      print('Resizing grid...')
    x = self.input_embedding(x)
    x = einops.rearrange(x, 'b c h w -> b (h w) c')
    x = self.norm(x) + self.position_embedding
    for block in self.blocks:
      x = block(x)
    x = einops.rearrange(x, 'b (h w) c -> b c h w', h=self.grid_size[0], w=self.grid_size[1])
    x = self.output_embedding(x)
    return x

  def forward(self, x_hat: th.Tensor, x_aux: th.Tensor = None, horizon: int = 1):
    '''
    Input x is assumed to be a batch of images shaped (batch, data_channels, time, height, width)
    :param x_hat: Input
    :param x_aux: Additional prescribed input
    :param horizon: Number of closed-loop steps
    '''
    context, height, width = x_hat.shape[-3:]
    if (height // self.k, width // self.k) != self.grid_size:
      self.resize_grid((height, width))
      print('Resizing grid...')
    y = []
    for t in range(0, horizon, context):
      # add static information
      x = th.cat((x_hat,
                  x_aux[:, :, t: t + context]), dim=1) if x_aux is not None else x_hat
      # time2depth
      x = einops.rearrange(x, 'b c t h w -> b (c t) h w')
      x = self._forward(x)
      # depth2time
      x_hat = einops.rearrange(x, 'b (c t) h w -> b c t h w', t=context)
      y.append(x_hat)
    out = th.cat(y, dim=2)
    return out

  def resize_grid(self, new_img_size: (int, int)):
    '''
    Can be used to adapt the position embedding and internal grid to a new image size
    '''
    self.grid_size = (new_img_size[0] // self.k, new_img_size[1] // self.k)
    self.position_embedding = nn.Parameter(nn.functional.interpolate(
        self.position_embedding, self.grid_size, mode='bilinear', align_corners=False),
        requires_grad=False)


