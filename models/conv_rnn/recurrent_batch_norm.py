import torch as th
from torch import nn
from torch.nn.functional import batch_norm

class LSTMCell2dBN(nn.Module):
    '''LSTMCell for 2d inputs'''
    def __init__(self,  
                 h_channels: int,
                 x_channels: int = 0,
                 kernel_size: int = 1,
                 tracked_steps: int = 1,
                 forget_gate_bias: int = 0,
                 rnn_act: callable = nn.Tanh):
        '''
        :param x_channels: Input channels
        :param h_channels: Latent state channels
        :param kernel_size: Size of the convolution kernel. Default to 1.
        :param tracked_steps: Time steps over which we track batch statistics. Default to 1.
        :param forget_gate_bias: Value to initialize the forget gate bias. Default to zero.
        :param rnn_act: (Optional) Activation function to use inside the LSTM. Default to tanh.
        '''
        super().__init__()
            
        self.phi = rnn_act()
        
        if x_channels is not None and x_channels > 0:
            self.W_xh = nn.Conv2d(
                in_channels = x_channels, 
                out_channels = 4 * h_channels,
                kernel_size = kernel_size, 
                padding = 'same', 
                bias = False)
            self.bn_x = RecurrentBatchNorm(4 * h_channels, tracked_steps)
        else:
            self.W_xh = None
            
        self.W_hh = nn.Conv2d(
            in_channels = h_channels, 
            out_channels = 4 * h_channels,
            kernel_size = kernel_size, 
            padding = 'same', 
            bias = True)
        
        self.bn_h = RecurrentBatchNorm(4 * h_channels, tracked_steps)
        self.bn_c = RecurrentBatchNorm(h_channels, tracked_steps)
        
        bias = th.zeros((4 * h_channels))
        bias[h_channels: 2 * h_channels] = forget_gate_bias #positive values here can improve long term memory
        self.W_hh.bias = nn.Parameter(bias)
        
    def forward(self, x, h, c, t = 0) -> th.Tensor:
        '''
        LSTM forward pass
        :param x: Input
        :param h: Hidden state
        :param c: Cell state
        :param t: Time step for recurrent batch normalization
        '''
        gates = self.bn_h(self.W_hh(h), t)
        
        if self.W_xh is not None and x is not None:
            gates += self.bn_x(self.W_xh(x), t)
        
        i, f, o, g = gates.chunk(chunks = 4, axis = 1)
            
        c = th.sigmoid(f) * c + th.sigmoid(i) * self.phi(g)
        h = th.sigmoid(o) * self.phi(self.bn_c(c, t))
        
        return h, c

class RecurrentBatchNorm(nn.Module):

    """
    A batch normalization module which keeps its running mean
    and variance separately per timestep.
    """

    def __init__(self, 
                 num_features: int, 
                 max_length: int, 
                 eps: float = 1e-5, 
                 momentum: float = 0.1,
                 gamma_init: float = 0.1,
                 affine: bool = True):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        
        Further modified from 
        https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/
        
        :param num_features: Number of features to normalize.
        :param max_length: Over how many timesteps we track the statistics
        :param eps: Small value for numerical stability
        :param momentum: Momentum of the running stats
        :param gamma_init: Initial value for the affine scale. Default to 0.1 as per Cooijmans et al 2017.
        :param affine: Learnable affine transformation.
        """

        super().__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.gamma_init = gamma_init
        
        if self.affine:
            self.weight = nn.Parameter(th.FloatTensor(num_features))
            self.bias = nn.Parameter(th.FloatTensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        for i in range(max_length):
            self.register_buffer(
                'running_mean_{}'.format(i), th.zeros(num_features))
            self.register_buffer(
                'running_var_{}'.format(i), th.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            if self.gamma_init:
                self.weight.data.fill_(self.gamma_init) #
            else:
                self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input_.size(1), self.num_features))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        return batch_norm(
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))