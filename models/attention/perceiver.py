import torch as th
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange
from math import prod

def count_parameters(model, min_param_size = 0, verbose = False):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_params = parameter.numel()
        total_params += param_params
        if param_params > min_param_size and verbose:
            print(f"{name}: {param_params}")
    print(f"Total Parameters: {total_params}")

class TransformerBlock(nn.Module):
    '''
    Transformer block with multi-head attention and feed-forward network.
    Implements pre-normalization and layer scale.
    '''
    def __init__(self, 
                 dim: int,
                 attn_heads: int = 8, 
                 dropout: float = 0., 
                 layer_scale: float = None, 
                 query_residual: bool = True,
                 expansion_factor: int = 4,
                 activation = nn.GELU):
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
        self.attn = nn.MultiheadAttention(dim, num_heads= attn_heads, dropout = dropout, batch_first= True)
        #feed-forward network
        hidden_dim = dim * expansion_factor
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        #residual connection to query
        self.query_residual = query_residual
        #layer scale
        self.attn_layer_scale = nn.Parameter(th.ones(1, 1, dim) * layer_scale) if layer_scale is not None else 1
        self.ffn_layer_scale = nn.Parameter(th.ones(1, 1, dim) * layer_scale) if layer_scale is not None else 1

    def forward(self, x: th.Tensor, context: th.Tensor = None):
        q = self.to_q(x)
        k, v = self.to_kv(context) if context is not None else self.to_kv(x)
        out, attn_weights = self.attn(q, k, v)
        x = x + out * self.attn_layer_scale if self.query_residual else out
        x = x + self.ffn(x) * self.ffn_layer_scale
        return x
    
class MultiHeadLinear(nn.Module):
    '''Multihead linear layer'''
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 2):
        '''
        input_dim: input dimension
        output_dim: output dimension
        num_heads: number of heads
        '''
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_heads)])
    
    def forward(self, x: th.Tensor):
        '''
        x: input tensor (batch_size, *, input_dim)
        return: output tensor (batch_size, num_heads, *, output_dim)
        '''
        return th.stack([head(x) for head in self.heads], dim = 1)
    
class FieldPreprocessor(nn.Module):
    def __init__(self, dim: int, domain_size: tuple, patch_size: tuple = (1, 1, 1)):
        '''
        :param dim: embedding dimension
        :param domain_size: domain size of input variable
        :param patch_size: patch size per token
        '''
        super().__init__()
        assert len(domain_size) == len(patch_size) #domain size and patch size must have same number of dimensions
        self.patch_size, self.domain_size = th.tensor(patch_size), th.tensor(domain_size)
        self.num_tokens = int(self.domain_size.div(self.patch_size).prod())
        self.dim = dim
        self.to_tokens = Rearrange('b (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw)', 
                                    pt = patch_size[0], ph = patch_size[1], pw = patch_size[2])
        self.linear = nn.Linear(self.patch_size.prod(), dim)
        self.position_code = nn.init.normal_(nn.Parameter(th.zeros((self.num_tokens, dim)), requires_grad = True), std = 0.02)        

    def forward(self, observations: th.Tensor):
        '''
        :param observations: (batch_size, *domain_size)
        :return: Tensor (batch_size, num_tokens, dim)
        '''
        tokens = self.to_tokens(observations)
        tokens = self.linear(tokens) + self.position_code
        return tokens
    
class OneHotPreprocessor(nn.Module):
    def __init__(self, dim, num_classes: int, num_tokens: int = 1):
        '''
        :param dim: embedding dimension
        :param num_classes: number of classes
        :param num_tokens: number of tokens
        '''
        super().__init__()
        self.num_classes = num_classes
        self.num_tokens = num_tokens
        self.dim = dim
        self.embedding = nn.Embedding(num_classes, dim)
        nn.init.normal_(self.embedding.weight, std = 0.02)
        self.position_code = nn.init.normal_(nn.Parameter(th.zeros((1, num_tokens, dim)), requires_grad = True), std = 0.02)
        
    def forward(self, observations: th.Tensor):
        '''
        :param observations: (batch_size, num_tokens)
        :return: (batch_size, num_tokens, dim)
        '''
        tokens = self.embedding(observations).squeeze() + self.position_code
        return tokens
    
class FieldPostprocessor(nn.Module):
    def __init__(self, 
                 dim: int,
                 num_heads: int = 1,
                 patch_size: tuple = (1, 1, 1),
                 ):
        '''
        dim: latent dimension of tokens
        num_heads: number of prediction heads
        patch_size: patch size per token
        '''
        super().__init__()
        #model attributes
        self.patch_size = th.tensor(patch_size)
        self.patch_dim =  self.patch_size.prod()
        self.dim = dim
        #linear projection
        self.linear = MultiHeadLinear(dim, self.patch_dim, num_heads)
        
    def forward(self, tokens: th.Tensor, query_positions: tuple):
        '''
        :param tokens: (batch_size, num_tokens, latent_dim)
        :param query_positions: (*query_domain_size)
        :return: Tensor (batch_size, num_heads, *target_domain_size)
        '''
        #linear projection
        tokens = self.linear(tokens)
        #patch embedding
        observations = rearrange(tokens,
            pattern = 'b c (t h w) (pt ph pw) -> b c (t pt) (h ph) (w pw)', 
            pt = self.patch_size[0], ph = self.patch_size[1], pw = self.patch_size[2],
            t = query_positions[0], h = query_positions[1], w = query_positions[2],
            )
        return observations

class IndexPostprocessor(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1, num_classes: int = 2):
        '''
        dim: latent dimension of tokens
        num_heads: number of prediction heads
        num_classes: number of classes
        '''
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.linear = MultiHeadLinear(dim, num_classes, num_heads)

    def forward(self, tokens: th.Tensor, query_positions: tuple):
        '''
        :param tokens: (batch_size, num_tokens, latent_dim)
        :param query_positions: unused
        :return: Tensor (batch_size, num_heads, num_tokens, num_classes)
        '''
        #linear projection
        observations = self.linear(tokens)
        return observations


class StructuredEncoder(nn.Module):
    def __init__(self, modalities: dict, latent_dim: int):
        '''
        modalities: dictionary of the form {modality: {'preprocessor': preprocessor, 'mask_prob': mask_prob, ...}}
        latent_dim: dimension of common latent space
        '''
        super().__init__()
        self.mask_probs = {}
        self.encoders = nn.ModuleDict()
        self.codes = nn.ParameterDict()
        for name, modality in modalities.items():
            self.encoders[f"preprocessor_{name}" ] = modality['preprocessor']
            self.codes[f"modality_{name}"] = nn.init.normal_(nn.Parameter(th.zeros((1, 1, latent_dim - modality['preprocessor'].dim))), std = 0.02)
            self.codes[f"masking_{name}"] = nn.init.normal_(nn.Parameter(th.zeros((1, 1, latent_dim))), std = 0.02)
            self.mask_probs[name] = modality['mask_prob']

    def forward(self, structured_data: dict):
        '''
        :param structured_data: dictionary of the form {modality: data}
        :return: Tensor (batch_size, num_tokens, latent_dim), dictionary of the form {modality: mask}
        '''
        tokens, masks = {}, {}
        for modality, data in structured_data.items():
            preproc = self.encoders[f"preprocessor_{modality}"](data)
            padded = th.cat([preproc, self.codes[f"modality_{modality}"].expand(preproc.shape[0], preproc.shape[1], -1)], dim = -1)
            if self.mask_probs[modality] > 0:
                mask = th.rand(padded.shape[0], padded.shape[1], 1, device = padded.device) < self.mask_probs[modality]
                padded = th.where(mask, self.codes[f"masking_{modality}"].expand_as(padded), padded)
                masks[modality] = mask
            tokens[modality] = padded
        #create predictable order
        tokens_list = [tokens[modality] for modality in sorted(structured_data.keys())]
        return th.cat(tokens_list, dim = 1), masks
    
class StructuredDecoder(nn.Module):
    def __init__(self, modalities: dict):
        '''
        modalities: dictionary of the form {modality: {'postprocessor' : postprocessor, ...}}
        latent_dim: dimension of common latent space
        '''
        super().__init__()
        self.decoders = nn.ModuleDict({
            f"postprocess_{name}": modality['postprocessor'] 
            for name, modality in modalities.items()})

    def forward(self, tokens: th.Tensor, queries: dict):
        '''
        :param tokens: (batch_size, num_tokens, latent_dim)
        :param queries: dictionary of the form {modality: positions}
        :return: dictionary of the form {modality: data}
        '''
        structured_data = {}
        start = 0
        for modality, positions in sorted(queries.items()):
            #get modality specific tokens
            end = start + prod(positions)
            mod_tokens = tokens[:, start:end, :]
            start = end
            #get modality specific data
            mod_data = self.decoders[f"postprocess_{modality}"](mod_tokens, positions)
            #add modality specific data to dictionary
            structured_data[modality] = mod_data
        return structured_data

class ContextPerceiver(nn.Module):
    '''Context Perceiver Model'''
    def __init__(self, 
                 modalities: dict,
                 num_latents: int = 256,
                 latent_dim: int = 512,
                 latent_depth: int = 8,
                 interpolate_queries: bool = True,
                 kwargs: dict = {},
                 ):
        super().__init__()
        #model components
        self.encoder = StructuredEncoder(modalities, latent_dim)
        self.decoder = StructuredDecoder(modalities)
        # cross modality encoding
        self.encoder_cross_attn = TransformerBlock(latent_dim, **kwargs)
        #cross modality decoding
        self.decoder_cross_attn = TransformerBlock(latent_dim, **kwargs, query_residual = False)
        #Latent processing
        self.transformer = nn.ModuleList([TransformerBlock(latent_dim, **kwargs) 
                                          for _ in range(latent_depth)])
        # latent code
        self.latent_codes = nn.init.normal_(nn.Parameter(th.zeros((1, num_latents, latent_dim)), requires_grad = True), std = 0.02)
        # query codes
        self.query_codes = nn.ParameterDict({
            f"query_{name}": nn.init.normal_(nn.Parameter(th.zeros((1, modality['preprocessor'].num_tokens, latent_dim))), std = 0.02) 
            for name, modality in modalities.items()
                                         })
        #interpolate queries?
        self.interpolate = interpolate_queries
        #Initialize weights        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            th.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, structured_data: dict, structured_query: dict):
        '''
        :param structured_data: dictionary of the form {modality: data}
        :param structured_query: dictionary of the form {modality: positions}
        :return: dictionary of the form {modality: data}, dictionary of the form {modality: mask}
        '''
        ### Encoder
        tokens, masks = self.encoder(structured_data)
        ### Latent processing
        latents = self.latent_codes.expand(tokens.shape[0], -1, -1)
        latents = self.encoder_cross_attn(latents, tokens)
        for block in self.transformer:
            latents = block(latents)
        ### Decoder
        queries = []
        for modality, positions in sorted(structured_query.items()):
            qc = self.query_codes[f"query_{modality}"]
            num_queries = prod(positions) if isinstance(positions, tuple) else positions
            if qc.shape[1] != num_queries and self.interpolate:
                qc = rearrange(qc, 'b n d -> b d n')
                qc = nn.functional.interpolate(qc, size = num_queries, mode = 'linear', align_corners = False)
                qc = rearrange(qc, 'b d n -> b n d')
            queries.append(qc)
        queries = th.cat(queries, dim = 1).expand(tokens.shape[0], -1, -1)
        predictions = self.decoder_cross_attn(queries, latents)
        predictions = self.decoder(predictions, structured_query)
        return predictions, masks




if __name__ == "main":
    device = f'cuda:{th.cuda.current_device()}' if th.cuda.is_available else 'cpu'
    print(f'Using device: {device}')
    #test parameters
    batch_size = 8
    embedding_dim = 128
    latent_dim = 256
    depth = 4
    num_latents = 32
    domain_size = (32, 32, 32)
    patch_size = (4, 4, 4)
    query_size = tuple(domain_size[i] // patch_size[i] for i in range(len(domain_size)))
    num_classes = 10
    #test model
    modalities = {'field': {'preprocessor': FieldPreprocessor(embedding_dim, domain_size= domain_size, patch_size= patch_size),
                            'postprocessor': FieldPostprocessor( latent_dim, patch_size= patch_size),
                            'mask_prob': 0.5},
                    'onehot': {'preprocessor': OneHotPreprocessor( embedding_dim, num_classes= num_classes, num_tokens = domain_size[0]),
                            'postprocessor': IndexPostprocessor(latent_dim, num_classes= num_classes),
                            'mask_prob': 0.5}
                }
    model = ContextPerceiver(modalities, latent_dim = latent_dim, latent_depth = depth, num_latents = num_latents, interpolate_queries = True).to(device)
    count_parameters(model)
    #test forward pass
    structured_data = {'field': th.rand(batch_size, *domain_size, device = device),
                        'onehot': th.randint(0, num_classes, (batch_size, 1, domain_size[0]), device = device)}

    structured_query = {'field': query_size,
                            'onehot': (1, query_size[0])
                        }
    downsampling_query = {'field': (query_size[0] // 2, query_size[1] // 2, query_size[2] // 2),
                            'onehot': (query_size[0] // 2,1)}
    upsampling_query = {'field': (query_size[0] * 2, query_size[1] * 2, query_size[2] * 2),
                        'onehot': (query_size[0] * 2,1)}

    predictions, masks = model(structured_data, structured_query)
    print(predictions['field'].shape)
    print(predictions['onehot'].shape)
    print(masks['field'].shape)
    print(masks['onehot'].shape)
    #test downsampling
    predictions, masks = model(structured_data, downsampling_query)
    print(predictions['field'].shape)
    print(predictions['onehot'].shape)
    print(masks['field'].shape)
    print(masks['onehot'].shape)
    #test upsampling
    predictions, masks = model(structured_data, upsampling_query)
    print(predictions['field'].shape)
    print(predictions['onehot'].shape)
    print(masks['field'].shape)
    print(masks['onehot'].shape)