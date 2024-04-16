import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange


class MLP(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 4, dropout: float =0.):
        """
        dim: dimension of the input
        expansion_factor: expansion factor of the MLP
        dropout: dropout rate
        """
        super(MLP, self).__init__()
        hidden_dim = expansion_factor * dim # inverted bottleneck MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)
    
class AdaptivePerceiverBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, expansion_factor: int = 4, dropout: float = 0., attn_residual: bool = True):
        """
        dim: dimension of the input
        num_heads: number of heads in the multihead attention
        expansion_factor: expansion factor of the MLP
        dropout: dropout rate
        attn_residual: whether to use residual connection in the attention layer (important to avoid for query-decoding)
        """
        super().__init__()
        # modulation network, Rearrange covers chunking and unsqueezing
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6), Rearrange("b (h d) -> h b () d", h = 6))
        # attention
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        # mlp
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim, expansion_factor, dropout)
        # switch for residual connection in the attention layer
        self.attn_residual = attn_residual

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor, context: torch.Tensor = None):
        """
        x: [batch, x_seq_len, dim]
        conditioning: [batch, dim]
        context: [batch, context_seq_len, dim]
        """
        # apply modulation network to produce (shift, scale, gate) for the attention and mlp
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning)        
        # skip connection
        residual = x
        # adaptive LayerNorm 
        x = self.norm_attn(x) * (1 + scale_attn) + shift_attn
        z = self.norm_attn(context) * (1 + scale_attn) + shift_attn if context is not None else x
        # apply attention
        x = self.attn(x, z, z, need_weights = False)[0]
        # apply residual connection
        x = gate_attn * x + residual if self.attn_residual else x
        # skip connection
        residual = x
        # adaptive LayerNorm
        x = self.norm_mlp(x) * (1 + scale_mlp) + shift_mlp
        # apply mlp and residual connection
        x = gate_mlp * self.mlp(x) + residual
        return x
    
class AdaptiveTails(nn.Module):
    def __init__(self, dim: int, output_dim, num_tails: int = 1):
        """ Adaptive tails module learns a set of tails for the input sequence.
        dim: dimension of the input
        output_dim: dimension of the output
        num_tails: number of tails to learn        
        """
        super().__init__()
        self.norm_tails = nn.LayerNorm(dim, elementwise_affine=False)
        self.tails = nn.ModuleList([nn.Linear(dim, output_dim) for _ in range(num_tails)])
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2), Rearrange("b (h d) -> h b () d", h = 2))

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        """
        x: [batch, x_seq_len, dim]
        conditioning: [batch, 1, dim]
        """
        shift, scale = self.adaLN_modulation(conditioning)
        x = self.norm_tails(x) * (1 + scale) + shift
        x = torch.stack([tail(x) for tail in self.tails], dim = 1)
        return x

class TimeStepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256, max_period: float = 1e4, learnable: bool = True):
        """ Time step embedding module learns a set of sinusoidal embeddings for the input sequence.
        dim: dimension of the input
        freq_dim: number of frequencies to use
        max_period: maximum period to use
        learnable: whether to learn the time step embeddings
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)) if learnable else nn.Identity()
        self.freq_dim = freq_dim
        self.max_period = max_period

    @staticmethod
    def timestep_embedding(t: torch.Tensor, num_freqs: int, max_period: float):
        """ 
        t: [N] tensor of time steps to embed
        num_freqs: number of frequencies to use
        max_period: maximum period to use
        returns: [N, num_freqs] tensor of embedded time steps
        """
        half = num_freqs // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args =  t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if num_freqs % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1) # zero pad
        return embedding
    
    def forward(self, t: torch.Tensor):
        """ 
        t: [N] tensor of time steps to embed
        returns: [N, dim] tensor of embedded time steps
        """
        t_freqs = self.timestep_embedding(t, self.freq_dim, self.max_period)
        return self.mlp(t_freqs)

class Tokenization(nn.Module):
    """
    Tokenization module that converts arbitrarily sized grids of values into a sequence of equally spaced tokens of specified size.
    """
    def __init__(self, shape_grid: tuple, shape_tokens: tuple):
        """
        shape_grid: shape of the input grid
        size_shape_tokenstoken: shape of the tokens
        """
        super().__init__()
        #check if shape_grid is evenly divisible by size_token
        assert len(shape_grid) == len(shape_tokens), "shape_grid and shape_tokens must have the same length" 
        assert all([shape_grid[i] % shape_tokens[i] == 0 for i in range(len(shape_grid))]), "shape_grid must be evenly divisible by shape_tokens"
        #patterns for rearrange
        pattern = {f"p{i}": shape_tokens[i] for i in range(len(shape_tokens))}
        pattern.update({f"n{i}": shape_grid[i] // shape_tokens[i]  for i in range(len(shape_grid))})
        #recipe for rearrange
        recipe_grid = "b " + " ".join([f"(n{i} p{i})" for i in range(len(shape_grid))])
        recipe_tokens = "b (" + " ".join([f"n{i}" for i in range(len(shape_grid))])+ ") (" + " ".join([f"p{i}" for i in range(len(shape_grid))]) + ")"
        #rearrange layers
        self.grid_to_token = Rearrange(recipe_grid + " -> " + recipe_tokens, **pattern)
        self.token_to_grid = Rearrange(recipe_tokens + " -> " + recipe_grid, **pattern)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        """
        x: input tensor
        reverse: whether to convert from grid to patch or patch to grid
        """
        return self.grid_to_token(x) if not reverse else self.token_to_grid(x)
    
class AdaptivePerceiver(nn.Module):
    def __init__(self,
                 shape_grid: tuple,
                 shape_tokens: tuple,
                 num_channels: int,
                 num_layers: int = 6,
                 num_latents: int = 256,
                 num_tails: int = 1,
                 num_heads: int = 8,
                 num_freqs: int = 256,
                 sigma: float = 0.02, 
                 dropout: float = 0.1
                 ):
        """
        shape_grid: shape of the input grid
        shape_tokens: shape of the tokens
        num_channels: number of channels in the model
        num_layers: number of transformer layers
        num_latents: number of latent embeddings
        num_tails: number of tails to learn
        num_heads: number of heads in the multihead attention
        num_freqs: number of frequencies to use in the time step embedding
        sigma: standard deviation of the weight initialization
        dropout: dropout rate for the model
        """
        super().__init__()
        #patch dimensions
        patch_dim = math.prod(shape_tokens)
        num_tokens = math.prod(shape_grid) // patch_dim
        # learnable position embeddings
        self.position_code = nn.Parameter(torch.zeros((1, num_tokens, num_channels)))
        # learnable queries
        self.query_code = nn.Parameter(torch.zeros((1, num_tokens, num_channels)))
        # learnable latents
        self.latent_code = nn.Parameter(torch.zeros((1, num_latents, num_channels)))
        # learnable conditioning 
        self.conditioning_code = TimeStepEmbedding(num_channels, num_freqs, learnable = True)
        #patching
        self.tokenize = Tokenization(shape_grid, shape_tokens)
        #input projection
        self.input_projection = nn.Linear(patch_dim, num_channels)
        #perceiver blocks
        self.encoder = AdaptivePerceiverBlock(num_channels, num_heads, dropout = dropout, attn_residual = False)
        self.processor = nn.ModuleList([AdaptivePerceiverBlock(num_channels, num_heads, dropout = dropout) 
                                          for _ in range(num_layers)])
        self.decoder = AdaptivePerceiverBlock(num_channels, num_heads, dropout = dropout, attn_residual = False)
        #adaptive tail ensemble
        self.tails = AdaptiveTails(num_channels, patch_dim, num_tails)
        #weight initialization
        self.sigma = sigma
        self.initialize_weights()

    def initialize_weights(self):
        # standard Transformer initialization:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize position embeddings:
        nn.init.normal_(self.position_code, std=self.sigma)
        nn.init.normal_(self.query_code, std=self.sigma)
        nn.init.normal_(self.latent_code, std=self.sigma)
        nn.init.normal_(self.conditioning_code.mlp[0].weight, std=self.sigma)
        nn.init.normal_(self.conditioning_code.mlp[2].weight, std=self.sigma)
        # Zero-out adaLN modulation layers in AdaptivePerceiver blocks:
        nn.init.constant_(self.encoder.adaLN_modulation[-2].weight, 0)
        nn.init.constant_(self.encoder.adaLN_modulation[-2].bias, 0)
        for block in self.processor:
            nn.init.constant_(block.adaLN_modulation[-2].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-2].bias, 0)
        nn.init.constant_(self.decoder.adaLN_modulation[-2].weight, 0)
        nn.init.constant_(self.decoder.adaLN_modulation[-2].bias, 0)
        # Zero-out output layers:
        nn.init.constant_(self.tails.adaLN_modulation[-2].weight, 0)
        nn.init.constant_(self.tails.adaLN_modulation[-2].bias, 0)

    def forward(self, data: torch.Tensor, conditioning: torch.Tensor = None, masks: tuple = (None, None)):
        #tokenization
        tokens = self.tokenize(data)
        #input embedding
        x = self.input_projection(tokens)
        # dimensions
        B, N, C, D = *x.shape, tokens.shape[-1]
        #conditional embedding
        c = self.conditioning_code(conditioning) if conditioning is not None else torch.zeros((B, C), device=data.device)
        #indexing
        encoder_index = masks[0].view(-1).nonzero() if masks[0] is not None else torch.arange(N, device=data.device)[:, None]
        decoder_index = masks[1].view(-1).nonzero() if masks[1] is not None else torch.arange(N, device=data.device)[:, None]
        #apply position code
        x = x + self.position_code.expand(B, -1, -1)
        #masking
        x = x.gather(dim = 1, index = encoder_index.expand(B, -1, C))
        # encoder to latent 
        latents = self.latent_code.expand(B, -1, -1)
        latents = self.encoder(latents, conditioning = c, context = x)
        #latent transformer
        for block in self.processor:
            latents = block(latents, conditioning = c)
        #decoder from latent 
        queries = self.query_code.expand(B, -1, -1).gather(dim = 1, index = decoder_index.expand(B, -1, C))
        queries = self.decoder(queries, conditioning = c, context = latents)
        # projection to output 
        y = self.tails(queries, conditioning = c)
        # gather outputs
        mu = torch.zeros_like(tokens).scatter(dim = 1, index = decoder_index.expand(B, -1, D), 
                src = y.mean(dim = 1))
        sigma = torch.zeros_like(tokens).scatter(dim = 1, index = decoder_index.expand(B, -1, D), 
                src = y.std(dim = 1))
        # de-tokenization
        outputs = self.tokenize(mu, reverse = True), self.tokenize(sigma, reverse = True)
        return outputs