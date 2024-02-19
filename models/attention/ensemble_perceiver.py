import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange

def structured_dropout(tokens: torch.Tensor, p: float = 0.) -> torch.Tensor:
        """
        Randomly drops a fraction p of tokens.
        Args: 
            tokens: tensor of tokens shape (B, N, C)
            p: float [0,1] indicating fraction of tokens to drop
        """
        B, N, C = tokens.shape #batch, sequence length, channels
        M = max(1, int((1 - p) * N)) #number of tokens to keep
        noise = torch.rand(B, N, 1, device = tokens.device) #random noise
        idcs = noise.topk(M, dim=1, sorted=False).indices #indices of tokens to keep
        tokens = tokens.gather(dim = 1, index = idcs.expand(-1, -1, C))
        return tokens

class MLP(nn.Module):
    """
    Inverted bottleneck feed-forward network with dropout and pre-LayerNorm
    """
    def __init__(self, dim, hidden_dim, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.mlp(x)

class CrossAttention(nn.Module):
    """
    Multihead Cross Attention wrapper with pre-norm.
    """
    def __init__(self, q_dim, kv_dim, query_residual = False, **attn_kwargs):
        super().__init__()
        self.q_norm = nn.LayerNorm(q_dim)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.attn = nn.MultiheadAttention(q_dim, kdim = kv_dim, vdim = kv_dim, 
                                          num_heads = attn_kwargs['nhead'], 
                                          dropout = attn_kwargs['dropout'], 
                                          batch_first = attn_kwargs['batch_first'])
        self.mlp = MLP(q_dim, attn_kwargs['dim_feedforward'], attn_kwargs['dropout'])
        self.query_residual = query_residual
        
    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        """
        q: query tensor of shape (B, N, C_q)
        kv: context tensor of shape (B, M, C_k)
        """
        q = self.q_norm(q)
        kv = self.kv_norm(kv) 
        attn = self.attn(q, kv, kv, need_weights = False)[0] + q if self.query_residual else self.attn(q, kv, kv, need_weights = False)[0]
        out = self.mlp(attn) + attn
        return out

class EnsemblePerceiver(nn.Module):
    def __init__(self, 
                 domain_size: tuple, # size of the model domain (time, height, width)
                 token_size: tuple, # size of the tokens (time, height, width)
                 num_variables: int, # number of input variables
                 num_tails: int, # number of prediction tails
                 num_latents: int, # number of latent variables
                 num_layers: int, # number of transformer layers
                 num_channels: int, # number of channels in the transformer
                 num_registers: int = 1, #number of register tokens
                 sigma: float = 0.02, # standard deviation of the initial parameters
                 ensemble_mode: str = 'query', # 'query' or 'tail'
                 attn_kwargs: dict = {  # keyword arguments for the attention layers
                     "dim_feedforward": 2048, # dimension of the feedforward network model
                     "batch_first": True, # input and output tensors are (batch, seq, feature)
                     "norm_first": True, # normalize before attention
                     "nhead": 16, # number of attention heads
                     "activation": "gelu", # activation function
                     "dropout": 0.1, # dropout probability
                 }
                ):
        super().__init__()
        assert len(domain_size) == len(token_size), "world and tokens must have the same number of dimensions"
        #handling token size and sequence length
        token_dim = math.prod(token_size) * num_variables #dimension of tokens after folding
        max_seq_len = math.prod(domain_size) // math.prod(token_size) #maximum sequence length
        pt, ph, pw = token_size #patch sizes
        T, H, W = (d // t for d, t in zip(domain_size, token_size)) #internal grid size
        #attributes
        self.ensemble_mode = ensemble_mode
        self.num_registers = num_registers
        self.num_latents = num_latents
        self.num_tails = num_tails
        #grid to patches and patches to grid
        self.patchify = Rearrange('b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)', pt = pt, ph = ph, pw = pw)
        #learnable parameters
        self.position_codes = nn.Parameter(torch.zeros(1, max_seq_len, num_channels))
        self.latent_codes = nn.Parameter(torch.zeros(1, num_latents + num_registers, num_channels))
        #ensemble
        if self.ensemble_mode == 'query':
            self.query_codes = nn.Parameter(torch.zeros(1, max_seq_len * num_tails, num_channels))
            self.tail = nn.Linear(num_channels, token_dim) 
            self.depatchify = Rearrange('b (e t h w) (c pt ph pw) -> b e c (t pt) (h ph) (w pw)', pt = pt, ph = ph, pw = pw, t = T, h = H, w = W, e = num_tails)
        else:
            self.query_codes = nn.Parameter(torch.zeros(1, max_seq_len, num_channels))
            self.tail = nn.ModuleList([nn.Linear(num_channels, token_dim) for _ in range(num_tails)]) 
            self.depatchify = Rearrange('b e (t h w) (c pt ph pw) -> b e c (t pt) (h ph) (w pw)', pt = pt, ph = ph, pw = pw, t = T, h = H, w = W)   
        #encoder module (token -> latent)
        self.encoder = nn.Linear(token_dim, num_channels)
        self.encoder_to_latent = CrossAttention(num_channels, num_channels, query_residual= True, **attn_kwargs)
        #transformer module (latent -> latent)
        self.transformer = nn.Sequential(*[nn.TransformerEncoderLayer(num_channels, **attn_kwargs) for _ in range(num_layers)])
        #decoder module (latent -> token)
        self.latent_to_decoder = CrossAttention(num_channels, num_channels, query_residual= False, **attn_kwargs)
        #weights initialization
        self.sigma = sigma #affects the effective learning rate of the learnable parameters(?)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std = self.sigma)

    def forward(self, data: torch.Tensor, dropout: float = 0.):
        B = data.shape[0]
        tokens = self.patchify(data)
        tokens = self.encoder(tokens) + self.position_codes #TODO: make positional codes indexable
        if self.training and dropout > 0:
            tokens = structured_dropout(tokens, dropout)
        latents = self.latent_codes.expand(B, -1, -1)
        latents = self.encoder_to_latent(latents, tokens)
        latents, registers = self.transformer(latents).split([self.num_latents, self.num_registers], dim = 1)
        query = self.query_codes.expand(B, -1, -1)
        out = self.latent_to_decoder(query, latents)
        out = self.tail(out) if self.ensemble_mode == 'query' else torch.stack([linear(out) for linear in self.tail], dim = 1)
        out = self.depatchify(out)
        return out, registers
    

