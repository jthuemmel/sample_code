import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange

class ConvNext(nn.Module):
    """
    Convolutional block based on "A ConvNet for the 2020s"
    """
    def __init__(self, channels: int, channels_c: int, k: int = 7, r: int = 4):
        """
        channels: number of input channels
        channels_c: number of conditioning channels
        k: kernel size
        r: expansion ratio
        """
        super().__init__()
        self.channels_last = Rearrange('b c h w -> b h w c')
        self.channels_first = Rearrange('b h w c -> b c h w')
        
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=k, stride=1, padding="same", groups=channels)
        self.norm = nn.LayerNorm(channels, elementwise_affine= False)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * r),
            nn.SiLU(),
            nn.Linear(channels * r, channels)
        )
        self.affine_map = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels_c, channels * 3),
            Rearrange('b (chunk c) -> chunk b () () c', chunk = 3)
        )
        self.channels = channels_c

    def forward(self, x: torch.Tensor, c: torch.Tensor = None):
        """
        x: input tensor [N, C, H, W]
        c: conditioning tensor [N, C_c]
        returns: output tensor [N, C, H, W]
        """
        res = x
        scale, shift, gate = self.affine_map(c if c is not None else torch.zeros((x.shape[0], self.channels), device=x.device))
        x = self.dwconv(x)
        x = self.channels_last(x)
        x = self.norm(x)
        x = (1 + scale) * x + shift
        x = self.mlp(x)
        x = x * gate
        x = self.channels_first(x)
        x = x + res
        return x

class Resize(nn.Module):
    """
    Resize block based on "A ConvNet for the 2020s"
    """
    def __init__(self, channels_in: int, channels_out: int, channels_c: int, scale: int = 2, transpose: bool = False):
        """
        channels_in: number of input channels
        channels_out: number of output channels
        channels_c: number of conditioning channels
        scale: scaling factor
        transpose: whether to use transposed convolution (i.e. upsampling) or not (i.e. downsampling)
        """
        super().__init__()      
        self.channels = channels_c

        if transpose:
            self.conv = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=scale, stride=scale)
        else:
            self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=scale, stride=scale)

        self.channels_last = Rearrange('b c h w -> b h w c')
        self.channels_first = Rearrange('b h w c -> b c h w')
        self.norm = nn.LayerNorm(channels_in, elementwise_affine= False)
        self.affine_map = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels_c, channels_in * 2),
            Rearrange('b (chunk c) -> chunk b () () c', chunk = 2)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor = None):
        """
        x: input tensor [N, C, H, W]
        c: conditioning tensor [N, C_c]
        returns: output tensor [N, C, H', W']
        """
        shift, scale = self.affine_map(c if c is not None else torch.zeros((x.shape[0], self.channels), device=x.device))
        x = self.channels_last(x)
        x = self.norm(x)
        x = x * (1 + scale) + shift
        x = self.channels_first(x)
        x = self.conv(x)
        return x
    
class FrequencyEmbedding(nn.Module):
    """
    Frequency embedding for conditional information.
    """
    def __init__(self, channels: int, freq_dim: int = 256, max_period: float = 1e4, learnable: bool = True):
        """ Time step embedding module learns a set of sinusoidal embeddings for the input sequence.
        dim: dimension of the input
        freq_dim: number of frequencies to use
        max_period: maximum period to use
        learnable: whether to learn the time step embeddings
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, channels), nn.SiLU(), nn.Linear(channels, channels)) if learnable else nn.Identity()
        self.freq_dim = freq_dim
        self.max_period = max_period
        
    @staticmethod
    def frequency_embedding(t: torch.Tensor, num_freqs: int, max_period: float):
        """ 
        t: [N] tensor of time steps to embed
        num_freqs: number of frequencies to use
        max_period: maximum period to use
        returns: [N, num_freqs] tensor of embedded time steps
        """
        assert num_freqs % 2 == 0, "please use an even number of frequencies"
        half = num_freqs // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args =  t[..., None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
    
    def forward(self, t: torch.Tensor):
        """ 
        t: [N] tensor of time steps to embed
        returns: [N, dim] tensor of embedded time steps
        """
        t_freqs = self.frequency_embedding(t, self.freq_dim, self.max_period)
        return self.mlp(t_freqs)

#
class ConvNextLSTM(nn.Module):
    """
    LSTM variant of the ConvNext block.
    """
    def __init__(self, channels: int, channels_c: int, k: int = 7, r: int = 4):
            """
            channels: number of input channels
            channels_c: number of conditioning channels
            k: kernel size
            r: expansion ratio
            """
            super().__init__()
            self.channels_last = Rearrange('b c h w -> b h w c')
            self.channels_first = Rearrange('b h w c -> b c h w')
            
            self.dwconv = nn.Conv2d(channels, channels, kernel_size=k, stride=1, padding="same", groups=channels)
            self.norm = nn.LayerNorm(channels, elementwise_affine= False)

            self.lstm = nn.Sequential(
                nn.Linear(channels, channels * 4),
                Rearrange('b h w (chunk c) -> chunk b h w c', chunk = 4)
            )
            self.affine_map = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels_c, channels * 2),
                Rearrange('b (chunk c) -> chunk b () () c', chunk = 2)
            )
            self.channels = channels_c

    def forward(self, hidden: torch.Tensor, cell: torch.Tensor, c: torch.Tensor = None):
        """
        x: input tensor [N, C, H, W]
        c: conditioning tensor [N, C_c]
        returns: output tensor [N, C, H, W]
        """
        scale, shift = self.affine_map(c if c is not None else torch.zeros((hidden.shape[0], self.channels), device=hidden.device))
        hidden = self.dwconv(hidden)
        hidden = self.channels_last(hidden)
        hidden = self.norm(hidden)
        hidden = (1 + scale) * hidden + shift
        f, i, g, o = self.lstm(hidden)
        cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(g)
        hidden = torch.tanh(cell) * torch.sigmoid(o) 
        hidden = self.channels_first(hidden)
        return hidden, cell

    
class U_Net(nn.Module):
    """
    U-Net architecture with ConvNext blocks and conditioning.
    """
    def __init__(self, 
                 channels_in: int,
                 channels_out: int,
                 num_channels: list = [32, 128, 256], 
                 num_encoder_blocks: list = [1, 2], 
                 num_decoder_blocks: list = [1, 2], 
                 scales: list = [4, 2], 
                 num_tails: int = 1,
                 k: int = 7,
                 r: int = 4,
                 num_freqs: int = 64, 
                 channels_c: int = 64,
                 learnable_conditioning: bool = True
                 ):
        """
        channels_in: number of input channels
        channels_out: number of output channels
        num_channels: number of channels in each stage
        num_encoder_blocks: number of ConvNext blocks in the encoder
        num_decoder_blocks: number of ConvNext blocks in the decoder
        scales: scaling factors in each stage
        num_tails: number of tails to ensemble
        k: kernel size
        r: expansion ratio
        num_freqs: number of frequencies to embed the conditioning
        channels_c: number of conditioning channels
        learnable_conditioning: whether the conditioning has learnable weights
        """
        super().__init__()
        self.num_stages = len(num_channels)
        
        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()
       
        # conditioning
        self.global_conditioning = FrequencyEmbedding(channels_c, num_freqs, learnable= learnable_conditioning, max_period=100)

        # add linear blocks
        self.encoder["linear"] = nn.Conv2d(channels_in, num_channels[0], kernel_size=1)
        self.decoder["linear"] = nn.ModuleList([nn.Conv2d(num_channels[0], channels_out, 1) for _ in range(num_tails)])
        # multi-stage U-Net:
        for i in range(1, self.num_stages):
            # add Resize blocks
            self.encoder[f"resize_{i}"] = Resize(num_channels[i-1], num_channels[i], channels_c, scale = scales[i-1])
            self.decoder[f"resize_{i}"] = Resize(num_channels[i], num_channels[i-1], channels_c, scale = scales[i-1], transpose=True)
            # add ConvNext blocks
            self.encoder[f"blocks_{i}"] = nn.ModuleList([ConvNext(num_channels[i], channels_c, k = k, r = r) for _ in range(num_encoder_blocks[i-1])])
            self.decoder[f"blocks_{i}"] = nn.ModuleList([ConvNext(num_channels[i], channels_c, k = k, r = r) for _ in range(num_decoder_blocks[i-1])])
        
        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            #initialise the weights of the convolutions
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_basic_init)

        def _ada_init(module):
            #initialise the affine map to zero
            if "affine_map" in module._modules:
                nn.init.zeros_(module.affine_map[1].weight)
                if module.affine_map[1].bias is not None:
                    nn.init.zeros_(module.affine_map[1].bias)

        self.apply(_ada_init)
        
    def forward(self, x: torch.Tensor, c: torch.Tensor = None):
        """
        x: input tensor [N, C, H, W]
        c: conditioning tensor [N, *]
        returns: output tensor [N, N_tails, C_out, H, W]
        """
        c = self.global_conditioning(c) if c is not None else None
        skips = [None]
        x = self.encoder["linear"](x)
        for i in range(1, self.num_stages):
            x = self.encoder[f"resize_{i}"](x, c = c)
            for block in self.encoder[f"blocks_{i}"]:
                x = block(x, c)
            skips.append(x)
        for i in reversed(range(1, self.num_stages)):
            x = skips[i] if i == self.num_stages else x + skips[i] # no skip connection for the deepest stage
            for block in self.decoder[f"blocks_{i}"]:
                x = block(x, c)
            x = self.decoder[f"resize_{i}"](x, c = c)
        x = torch.stack([tail(x) for tail in self.decoder["linear"]], dim = 1)
        return x


class U_LSTM(nn.Module):
    """
    U-Net Encoder-Decoder architecture with ConvNextLSTM blocks and conditioning.
    """
    def __init__(self, 
                 channels_in: int,
                 channels_out: int,
                 num_channels: list = [32, 128, 256], 
                 num_encoder_blocks: list = [1, 2], 
                 num_decoder_blocks: list = [1, 2], 
                 scales: list = [4, 2], 
                 num_tails: int = 1,
                 k: int = 7,
                 r: int = 4,
                 num_freqs: int = 64, 
                 channels_c: int = 64,
                 learnable_conditioning: bool = True
                 ):
        """
        channels_in: number of input channels
        channels_out: number of output channels
        num_channels: number of channels in each stage
        num_encoder_blocks: number of ConvNext blocks in the encoder
        num_decoder_blocks: number of ConvNext blocks in the decoder
        scales: scaling factors in each stage
        num_tails: number of tails to ensemble
        k: kernel size
        r: expansion ratio
        num_freqs: number of frequencies to embed the conditioning
        channels_c: number of conditioning channels
        learnable_conditioning: whether the conditioning has learnable weights
        """
        super().__init__()
        self.num_stages = len(num_channels)
        self.num_channels = num_channels
        self.num_tails = num_tails
        self.channels_out = channels_out
        self.num_encoder_blocks = [1] + num_encoder_blocks # add a dummy block to the beginning
        self.num_decoder_blocks = [1] + num_decoder_blocks # add a dummy block to the beginning
        # 
        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()
       
        # conditioning
        self.global_conditioning = FrequencyEmbedding(channels_c, num_freqs, learnable= learnable_conditioning, max_period=100)

        # add linear blocks
        self.encoder["linear"] = nn.Conv2d(channels_in, num_channels[0], kernel_size=1)
        self.decoder["linear"] = nn.ModuleList([nn.Conv2d(num_channels[0], channels_out, 1) for _ in range(num_tails)])


        # multi-stage U-Net:
        for i in range(1, self.num_stages):
            # add Resize blocks
            self.encoder[f"resize_{i}"] = Resize(num_channels[i-1], num_channels[i], channels_c, scale = scales[i-1])
            self.decoder[f"resize_{i}"] = Resize(num_channels[i], num_channels[i-1], channels_c, scale = scales[i-1], transpose=True)
            # add ConvNext blocks
            self.encoder[f"blocks_{i}"] = nn.ModuleList([ConvNextLSTM(num_channels[i], channels_c, k = k, r = r) for _ in range(num_encoder_blocks[i-1])])
            self.decoder[f"blocks_{i}"] = nn.ModuleList([ConvNextLSTM(num_channels[i], channels_c, k = k, r = r) for _ in range(num_decoder_blocks[i-1])])
        
        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            #initialise the weights of the convolutions
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_basic_init)

        def _ada_init(module):
            #initialise the affine map to zero
            if "affine_map" in module._modules:
                nn.init.zeros_(module.affine_map[1].weight)
                if module.affine_map[1].bias is not None:
                    nn.init.zeros_(module.affine_map[1].bias)

        self.apply(_ada_init)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor = None, horizon: int = 1):
        T = x.shape[2]
        states = self.forward_encoder(x, context[:, :T] if context is not None else None)
        predictions = self.forward_decoder(states, context[:, T:] if context is not None else None, horizon = horizon)
        return predictions
        
    def forward_encoder(self, x: torch.Tensor, context: torch.Tensor = None):
        # initialize hidden states and cell states for the encoder in each stage and block
        hidden_encoder = [[torch.zeros((1, self.num_channels[i], 1, 1), device=x.device) for _ in range(self.num_encoder_blocks[i])]
                            for i in range(self.num_stages)]
        cell_encoder = [[torch.zeros((1, 1, 1, self.num_channels[i]), device=x.device) for _ in range(self.num_encoder_blocks[i])]
                            for i in range(self.num_stages)]
        # forward pass through the encoder
        for tau in range(x.shape[2]):
            # conditioning in each time step
            context = self.global_conditioning(context[:, tau]) if context is not None else None
            # linear block in each step
            hidden_encoder[0][-1] = self.encoder["linear"](x[:, :, tau])
            # LSTM layers
            for i in range(1, self.num_stages):
                # resize last hidden state of the previous stage and add to the first hidden state of the current stage
                hidden_encoder[i][0] = hidden_encoder[i][0] + self.encoder[f"resize_{i}"](hidden_encoder[i-1][-1], c = context)
                # encoder blocks in each stage
                for j,block in enumerate(self.encoder[f"blocks_{i}"]):
                    # add the hidden state of the previous block to the current hidden state
                    z = hidden_encoder[i][j] + hidden_encoder[i][j-1] if j > 0 else hidden_encoder[i][j]
                    # propagate through the LSTM block
                    hidden_encoder[i][j], cell_encoder[i][j] = block(z, cell_encoder[i][j], context)
        # return hidden states and cell states
        return hidden_encoder, cell_encoder
    
    def forward_decoder(self, states: tuple, context: torch.Tensor = None, horizon: int = 1):
        # unpack states
        hidden_encoder, cell_encoder = states
        # initialize hidden states and cell states for the decoder in each stage and block
        hidden_decoder = [[torch.zeros_like(hidden_encoder[i][-1]) for _ in range(self.num_decoder_blocks[i])] 
                          for i in range(self.num_stages)]
        cell_decoder = [[torch.zeros_like(cell_encoder[i][-1]) for _ in range(self.num_decoder_blocks[i])]
                        for i in range(self.num_stages)]        
        # copy the last hidden state of the encoder to the first hidden state of the decoder
        for i in range(self.num_stages):
            hidden_decoder[i][0] = hidden_encoder[i][-1]
            cell_decoder[i][0] = cell_encoder[i][-1]
        # forward pass through the decoder
        predictions = []
        for tau in range(horizon):
            # conditioning in each time step
            context = self.global_conditioning(context[:, tau]) if context is not None else None
            # LSTM layers in reverse order to maintain the skip connections
            for i in reversed(range(1, self.num_stages)):
                for j,block in enumerate(self.decoder[f"blocks_{i}"]):
                    # add the hidden state of the previous block to the current hidden state
                    z = hidden_decoder[i][j] + hidden_decoder[i][j-1] if j > 0 else hidden_decoder[i][j]
                    # propagate through the LSTM block
                    hidden_decoder[i][j], cell_decoder[i][j] = block(z, cell_decoder[i][j], context)
                # resize last hidden state of the current stage and add to the first hidden state of the next stage
                hidden_decoder[i-1][0] = hidden_decoder[i-1][0] + self.decoder[f"resize_{i}"](hidden_decoder[i][-1], c = context)
            # linear tails in each step
            predictions.append(torch.stack([tail(hidden_decoder[0][-1]) for tail in self.decoder["linear"]], dim = 1))
        # return predictions
        predictions = torch.stack(predictions, dim = 2)
        return predictions
