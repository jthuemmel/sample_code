import torch as th
from torch import nn
from einops.layers.torch import Rearrange


class MHA(nn.Module):
    '''
    Multi-head attention module.
    Implements pre-normalization and layer scale.
    '''
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0., layer_scale:float = None, use_residual: bool = True):
        '''
        Args:
            dim: dimension of input
            heads: number of heads
            dropout: dropout rate
            layer_scale: initial per-layer scale factor
        '''
        super().__init__()
        #query projection
        self.to_q = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )
        #key and value projection
        self.to_kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            Rearrange('b n (kv d) -> kv b n d', kv = 2)
        )
        #multi-head attention
        self.attn = nn.MultiheadAttention(dim, num_heads= heads, dropout = dropout, batch_first= True)
        #layer scale
        self.layer_scale = nn.Parameter(th.ones(1, 1, dim) * layer_scale) if layer_scale is not None else 1
       #residual connection
        self.use_residual = use_residual
        
    def forward(self, x: th.Tensor, context: th.Tensor = None):
        '''
        Args:
            x: input tensor for query or query, key and value for self-attention
            context (optional): input for cross-attention
        '''
        q = self.to_q(x)
        k, v = self.to_kv(context) if context is not None else self.to_kv(x)
        out = self.attn(q, k, v)[0]
        out = x + out * self.layer_scale if self.use_residual else out
        return out
    
class FFN(nn.Module):
    '''
    Feed-forward network module.
    Implements pre-normalization and layer scale.
    '''
    def __init__(self, dim: int, expansion_factor: int = 4, dropout: float = 0., layer_scale: int = 1e-4, activation = nn.GELU):
        ''' 
        Args:
            dim: dimension of input
            expansion_factor: expansion factor of hidden dimension
            dropout: dropout rate
            layer_scale: initial per-layer scale factor
            activation: activation function
        '''
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.layer_scale = nn.Parameter(th.ones(1, 1, dim) * layer_scale)  if layer_scale is not None else 1
    def forward(self, x):
        return x + self.net(x) * self.layer_scale

class ContextCast(nn.Module):
    '''
    ContextCast implements a masked autoencoder for pretrained context embeddings.
    '''
    def __init__(self, data_dim: int, 
                 encoder_dim: int, 
                 decoder_dim: int,
                 grid_size: tuple = (6, 64, 160), 
                 num_latents = 1, 
                 patch_size: tuple = (1, 8, 8), 
                 encoder_depth: int = 4, 
                 decoder_depth: int = 2,
                 dropout: float = 0., 
                 predict_std: bool = False,
                 layer_scale: float = None,
                 **kwargs):
        """
        Args:
            data_dim: dimension of input data
            encoder_dim: dimension of latent space in encoder
            decoder_dim: dimension of latent space in decoder
            grid_size: size of the grid in the latent space
            num_latents: number of latent vectors to use
            patch_size: size of the patch to use
            encoder_depth: number of layers in the encoder
            decoder_depth: number of layers in the decoder
            dropout: dropout rate
            learnable_position_code: whether to use learnable position code or not
            predict_std: whether to predict std or not
            layer_scale: initial per-layer scale factor
        """
        super().__init__()
        #set helper parameters
        self.patch_size = patch_size if len(patch_size) == 3 else (1, *patch_size) #add time dimension if not present
        self.grid_size = (grid_size[0] // self.patch_size[0], grid_size[1] // self.patch_size[1], grid_size[2] // self.patch_size[2]) #grid size in latent space
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2] #number of tokens in latent space
        self.patch_dim = self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * data_dim #patch dimension
        self.encoder_dim = encoder_dim #encoder dimension
        self.decoder_dim = decoder_dim #decoder dimension
        #patch embedding
        self.to_patch = Rearrange('b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)', 
                                  pt = self.patch_size[0], ph = self.patch_size[1], pw = self.patch_size[2])  
        #inverse patch embedding
        self.from_patch = Rearrange('b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)', 
                                 pt = self.patch_size[0], ph = self.patch_size[1], pw = self.patch_size[2], 
                                 t = self.grid_size[0], h = self.grid_size[1], w = self.grid_size[2]) 
        
        ###ENCODER###
        #encoder input projection
        self.encoder_projection = nn.Sequential(
             nn.Linear(self.patch_dim, encoder_dim),
             nn.LayerNorm(encoder_dim))
        #encoder stack
        self.encoder_stack = nn.ModuleList([nn.Sequential(
            MHA(encoder_dim, dropout=dropout, layer_scale= layer_scale), 
            FFN(encoder_dim, dropout=dropout, layer_scale= layer_scale)) 
            for _ in range(encoder_depth)])
        
        ###DECODER###
        #decoder input projection
        self.decoder_projection = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, decoder_dim))
        #decoder stack
        self.decoder_stack = nn.ModuleList([nn.Sequential(
            MHA(decoder_dim, dropout=dropout, layer_scale= layer_scale),
            FFN(decoder_dim, dropout=dropout, layer_scale= layer_scale),)
            for _ in range(decoder_depth)])
        
        ###OUTPUT###
        #predict mean
        self.to_mean = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, self.patch_dim),
            self.from_patch)
        #predict std if desired
        self.to_std = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, self.patch_dim),
            self.from_patch) if predict_std else None

        ###TOKENS###
        #Initialize mask token
        self.mask_token = th.nn.init.normal_(nn.Parameter(th.zeros((1, 1, decoder_dim))), std = 0.02)
        #Initialize class token
        self.class_token = th.nn.init.normal_(nn.Parameter(th.zeros((1, 1, encoder_dim))), std = 0.02)
        #Initialize position code
        self.encoder_position_code = th.nn.init.normal_(nn.Parameter(th.zeros(1, self.num_patches, encoder_dim), requires_grad = True), std = 0.2)
        self.decoder_position_code = th.nn.init.normal_(nn.Parameter(th.zeros(1, self.num_patches, decoder_dim), requires_grad = True), std = 0.2)
        #Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            th.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, observation: th.Tensor, keep_idcs: th.Tensor) -> th.Tensor:
        B = observation.shape[0] #batch size
        ###ENCODER###
        #calculate position code
        s_enc = self.encoder_position_code.expand(B, -1, -1)
        #calculate patch embedding
        patches = self.to_patch(observation)
        #apply position code and linear projection
        z = s_enc + self.encoder_projection(patches)
        #mask patches <=> select only non-masked patches
        z = z.gather(dim = 1, index = keep_idcs.expand(-1, -1, self.encoder_dim))
        #add class token
        class_token = self.class_token.expand(B, -1, -1)
        z = th.cat((class_token, z), dim = 1)
        #process
        for encoder in self.encoder_stack:
            z = encoder(z)
        return z
    
    def forward_decoder(self, z: th.Tensor, restore_idcs: th.Tensor) -> tuple:
        B, M, _ = z.shape #batch size, number of unmasked tokens
        ###DECODER###
        z = self.decoder_projection(z)
        #calculate position code
        s_dec = self.decoder_position_code.expand(B, -1, -1)
        #append mask token
        N = self.num_patches - M + 1 #number of masked tokens (without class token)
        mask_tokens = self.mask_token.expand(B, N, -1) #create mask tokens
        y = th.cat((z[:, 1:], mask_tokens), dim = 1) #no class token
        #restore order and add position code
        y = s_dec + y.gather(dim = 1, index = restore_idcs.expand(-1, -1, self.decoder_dim))
        #add class token
        y = th.cat((z[:, :1], y), dim = 1)
        #predict all patches
        for decoder in self.decoder_stack:
            y = decoder(y)
        #remove class token
        cls, out = y[:, :1], y[:, 1:]
        #predict mean and optionally std
        mean = self.to_mean(out)
        std = self.to_std(out) if self.to_std is not None else None
        return (mean, std), cls

    def random_masking(self, data: th.Tensor, mask_ratio: float):
        #random masking
        B = data.shape[0] #batch size
        M = int(self.num_patches * (1 - mask_ratio)) #number of patches to keep
        noise = th.rand(B, self.num_patches, 1, device = data.device)#generate random noise
        shuffle_idcs = th.argsort(noise, dim = 1) #sort idcs by noise to shuffle
        restore_idcs = th.argsort(shuffle_idcs, dim = 1) #sort by idx to revert shuffle
        keep_idcs = shuffle_idcs[:, :M] #keep first M idcs
        #create binary mask
        mask = th.ones([B, self.num_patches, self.patch_dim], device=data.device) #create mask tensor
        mask[:, :M] = 0 #mask loss for first M patches
        mask = mask.gather(dim=1, index= restore_idcs.expand(-1, -1, self.patch_dim)) #restore order
        return keep_idcs, mask, restore_idcs

    def forward(self, observation: th.Tensor, mask_ratio: float = 0.) -> tuple:
        #observation: (B, C ,T, H, W)
        #calculate random masks
        keep_idcs, mask, restore_idcs = self.random_masking(observation, mask_ratio)
        #encode
        z = self.forward_encoder(observation, keep_idcs)
        #decode
        (mean, std), cls = self.forward_decoder(z, restore_idcs)
        #mask to image
        mask = self.from_patch(mask)
        #return
        return (mean, std), mask, cls
    
    
#load config
def my_load_config(model_name, model_path = '/mnt/qb/work/goswami/jthuemmel54/enso/stforenso/bin/ckpts/'):

    config = th.load(model_path + model_name + '_config.pth')
    epoch = config['num_epochs']
    ckpt = th.load(model_path + model_name + f'_checkpoint_{epoch}')
    #Data variables
    height = config['height']
    width = config['width']
    history = config['history']
    horizon = config['horizon']

    #Model variables
    encoder_dim = config['encoder_dim']
    decoder_dim = config['decoder_dim']
    data_dim = config['data_dim']
    patch_size = config['patch_size']
    num_latents = config['num_latents']
    timesteps = history + horizon
    encoder_depth = config['encoder_depth']
    decoder_depth = config['decoder_depth']
    dropout = config['dropout']
    layer_scale = config['layer_scale']
    predict_std = config['predict_std']
    learnable_position_code = config['learnable_position_code']

    #Model
    model = ContextCast(data_dim = data_dim,
                    decoder_dim= decoder_dim,
                    encoder_dim= encoder_dim, 
                    num_latents = num_latents, 
                    patch_size = patch_size,
                    encoder_depth = encoder_depth,
                    decoder_depth = decoder_depth, 
                    dropout=dropout, 
                    layer_scale=layer_scale, 
                    predict_std=predict_std,
                    learnable_position_code=learnable_position_code,
                    grid_size=(timesteps, height, width),
                    )
    
    model.load_state_dict(ckpt['model_state_dict'])
    return model, config