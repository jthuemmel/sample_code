import einops
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from components import MLP, DropPath, RMSNorm

class Attention(nn.Module):
    """
    Implements scaled dot-product attention with support for windowed computation and flexible in/out dims.
    Inspiration taken from: https://github.com/lucidrains/x-transformers/tree/main
    """
    def __init__(self, dim_q: int, dim_kv: int = None, dim_out: int = None, dim_head: int = 64, num_heads: int = None, **kwargs):
        super().__init__()
        dim_kv = dim_kv or dim_q
        dim_out = dim_out or dim_q
        self.nhead = num_heads or dim_q // dim_head # support more heads than dim_q allows
        dim_inner = dim_head * self.nhead
        self.to_q = nn.Linear(dim_q, dim_inner, bias = False)
        self.to_k = nn.Linear(dim_kv, dim_inner, bias = False)
        self.to_v = nn.Linear(dim_kv, dim_inner, bias = False)
        self.out_proj = nn.Linear(dim_inner, dim_out, bias = False)

    def forward(self, q, k, v, is_causal = False, **kwargs):
        # q: (B, *, N, D), k: (B, *, M, D), v: (B, *, M, D)
        B, ndim = q.size(0), q.ndim # needed to handle windowed computation

        # prepare queries, keys, values
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v) # (B, *, N, D) -> (B, *, N, H * d)
        q, k, v = map(lambda x: einops.rearrange(x, '... n (h d) -> (...) h n d', h = self.nhead), (q, k, v)) # (B, *, N, H * d) -> (B*, H, N, d)

        # scaled dot-product attention
        attn = nn.functional.scaled_dot_product_attention(q, k, v, is_causal = is_causal)
        
        # rearrange back to original shape
        out = einops.rearrange(attn, 'b h n d ->  b n (h d)') if ndim == 3 else einops.rearrange(attn, '(b g) h n d -> b g n (h d)', b = B)

        # project back to original dimension
        out = self.out_proj(out)
        return out

class SwinBlock(nn.Module):
    """
    Implements shifted window attention and feed-forward block for Swin Transformer.
    Inspiration taken from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
    """
    def __init__(self, dim: int, dim_head: int = 64, drop_path: float = 0., num_windows: int = 1, shift_window: int = 0):
        super().__init__()
        self.norm_attn = RMSNorm(dim)
        self.norm_ff = RMSNorm(dim)
        self.drop_path = DropPath(drop_path)
        self.attn = Attention(dim, dim_head = dim_head)
        self.ff = MLP(dim)

        self.split_windows = Rearrange("b (w n) d -> b w n d", w = num_windows) if num_windows > 1 else nn.Identity()
        self.merge_windows = Rearrange("b w n d -> b (w n) d") if num_windows > 1 else nn.Identity()

        self.shift_window = shift_window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # attention block
        skip = x
        x = self.norm_attn(x)

        # shift window for SW-MSA
        shifted_x = torch.roll(x, shifts = -self.shift_window, dims = 1) if self.shift_window > 0 else x

        # split windows, calculate attention, merge windows
        shifted_x = self.split_windows(shifted_x)
        shifted_x = self.attn(shifted_x, shifted_x, shifted_x)
        shifted_x = self.merge_windows(shifted_x)

        # reverse shift window
        x = torch.roll(shifted_x, shifts = -self.shift_window, dims = 1) if self.shift_window > 0 else shifted_x

        # drop-path and add skip
        x = self.drop_path(x) + skip
        
        # feed-forward
        skip = x
        x = self.norm_ff(x)
        x = self.ff(x)
        return self.drop_path(x) + skip

class PerceiverBlock(nn.Module):
    """
    Implements cross-attention and feed-forward block for Perceiver IO.
    Inspiration taken from: https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py
    """
    def __init__(self, dim: int, dim_tgt: int = None, dim_src: int = None, dim_head: int = 64, num_heads: int = None, **kwargs):
        super().__init__()
        dim_tgt = dim_tgt or dim
        dim_src = dim_src or dim
        self.norm_tgt = RMSNorm(dim_tgt) # this may be unnecessary if the input is already normalized
        self.norm_src = RMSNorm(dim_src)
        self.norm_ff = RMSNorm(dim)
        self.attn = Attention(dim_q = dim_tgt, dim_kv = dim_src, dim_head = dim_head, dim_out = dim, num_heads = num_heads) 
        self.ff = MLP(dim)

    def forward(self, src, tgt):
        # src: (B, *, N, D_src), tgt: (B, *, M, D_tgt) -> (B, *, M, D_tgt)

        # skip connection is omitted for now

        # cross-attention
        tgt = self.norm_tgt(tgt)
        src = self.norm_src(src)
        tgt = self.attn(tgt, src, src) # (B, *, M, D) x (B, *, N, D) -> (B, *, M, D)

        # feed-forward
        skip = tgt
        tgt = self.norm_ff(tgt)
        tgt = self.ff(tgt)
        return tgt + skip

class TransformerBlock(nn.Module):
    """
    Implements self-attention and feed-forward block for Transformer. Optionally supports causal attention.
    """
    def __init__(self, dim: int, dim_head: int = 64, drop_path: float = 0., is_causal = False, **kwargs):
        super().__init__()
        self.is_causal = is_causal
        self.norm_attn = RMSNorm(dim)
        self.norm_ff = RMSNorm(dim)
        self.drop_path = DropPath(drop_path)
        self.attn = Attention(dim, dim_head = dim_head)
        self.ff = MLP(dim)

    def forward(self, x):
        # x: (B, *, N, D)
        # attention
        skip = x
        x = self.norm_attn(x)
        x = self.attn(x, x, x, is_causal = self.is_causal)
        x = self.drop_path(x) + skip
        
        # feed-forward
        skip = x
        x = self.norm_ff(x)
        x = self.ff(x)
        return self.drop_path(x) + skip