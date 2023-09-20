import torch as th
import torch.nn as nn

class distana_tk(nn.Module):
    '''
    A ConvRNN transition kernel based on the Distana architecture
    '''
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 h_channels,
                 lat_channels = None, 
                 lat_act = None, 
                 k_combined = 3, 
                 k_lat = 3):
        super().__init__()
        
        self.h_channels = h_channels
        self.x_channels = in_channels - self.h_channels
        self.lat_channels = lat_channels or self.h_channels
        
        self.phi = lat_act or th.tanh
        
        if self.lat_channels > self.h_channels:
            lat_input = self.h_channels
        else:
            lat_input = self.lat_channels
    
        self.lat_process = nn.Conv2d(lat_input, self.lat_channels, kernel_size = k_lat, padding = 'same')
        self.conv = nn.Conv2d(
            in_channels = self.x_channels + self.lat_channels, 
            out_channels = self.h_channels, 
            kernel_size = k_combined, 
            padding = 'same')
        self.lstm_fc = nn.Conv2d(self.h_channels*2, out_channels, 1)
        
    def forward(self, xh):
        #input is th.cat((x,h), dim = 1) but we need them separate
        x, h = xh.split((self.x_channels, self.h_channels), dim = 1)
        #the lateral input is a mlp of a part of h
        lateral = self.phi(self.lat_process(h[:,:self.lat_channels]))
        #lateral input and x are processed together
        x_lat = th.cat((x,lateral), dim = 1)
        #this is the step where positional information is integrated
        convolved = self.conv(x_lat)
        #now we combine the augmented x with the original h
        combined = th.cat((convolved, h), dim = 1)
        #and this layer projects to the lstm input
        out = self.lstm_fc(combined)
        return out
    
class dw_conv2d(nn.Module):
    '''
    A depth-wise separated 2d Convolution.
    '''
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 groups: int = None,
                 bias: bool = True):
        super().__init__()   
        
        self.add_module('depthwise',
                        nn.Conv2d(
                            in_channels, 
                            in_channels, 
                            kernel_size, 
                            padding = 'same', 
                            groups = groups or in_channels, 
                            bias = bias)
                           )
        self.add_module('pointwise',
                        nn.Conv2d(in_channels, out_channels, 1, bias = bias)
                           )       
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
    
class inverted_bottleneck(nn.Module):
    '''
    MobileNet style inverted bottleneck block
    '''
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size = 3, 
        groups = None, 
        scale = 1):
        super().__init__()
        
        exp_channels = int(scale * in_channels)
                
       # self.add_module('norm', nn.GroupNorm(num_groups = 1, num_channels = in_channels))
        self.add_module('expansion', 
                        nn.Conv2d(in_channels, exp_channels, 1)
                       )
        self.add_module('depthwise',
                        nn.Conv2d(
                            exp_channels, 
                            exp_channels, 
                            kernel_size, 
                            padding = 'same', 
                            groups = groups or exp_channels)
                           )
        self.add_module('pointwise',
                        nn.Conv2d(exp_channels, in_channels, 1)
                           ) 
        if in_channels != out_channels:
            self.add_module('projection', nn.Conv2d(in_channels, out_channels, 1))
        else:
            self.add_module('projection', nn.Identity())
            
    def forward(self, x):
        sc = x
       # x = self.norm(x)
        x = self.expansion(x)
        x = self.depthwise(x)
        x = self.pointwise(x) + sc
        x = self.projection(x)
        return x 
    
class dense_tk(nn.Module):
    '''
    Transition kernel that augments the rnn input with a conv layer and concatenates.
    '''
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        lat_channels = None,
        conv_function = nn.Conv2d,
        kernel_size = 3, 
        groups = 1, 
        activation = None):
        super().__init__()
        
        lat_channels = lat_channels or in_channels        
        self.act = activation or nn.Identity()
        
        self.conv = conv_function(in_channels, lat_channels, kernel_size = kernel_size,
                                  padding = 'same', groups = groups)
        self.fc = nn.Conv2d(lat_channels + in_channels, out_channels, 1)
        
    def forward(self, x):
        z = self.act(self.conv(x))
        out = self.fc(th.cat((x, z), dim = 1))
        return out
    
class residual_tk(nn.Module):
    '''
    Transition kernel that augments the rnn input with a conv layer and sums.
    '''
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size = 3, 
        conv_function = nn.Conv2d,
        groups = 1, 
        activation = None):
        super().__init__()
        
        self.act = activation or nn.Identity()
        
        self.conv = conv_function(in_channels, in_channels, kernel_size = kernel_size, 
                                  padding = 'same', groups = groups)
        self.fc = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        shortcut = x
        x = self.act(self.conv(x))
        out = self.fc(x + shortcut)
        return out
    
class basic_tk(nn.Module):
    '''
    Transition kernel that processes the rnn input and passes it on.
    '''
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        conv_channels = None, 
        conv_function = nn.Conv2d,
        kernel_size = 3, 
        groups = 1, 
        activation = None):
        super().__init__()
        
        conv_channels = conv_channels or in_channels        
        self.act = activation or nn.Identity()
        
        self.conv = conv_function(in_channels, conv_channels, kernel_size = kernel_size, padding = 'same', 
                                  groups = groups)
        self.fc = nn.Conv2d(conv_channels, out_channels, 1)
        
    def forward(self, x):
        z = self.act(self.conv(x))
        out = self.fc(z)
        return out
    
class dw_block(nn.Module):
    '''
    A pointwise -> depthwise -> pointwise convolution block.
    '''
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        scale = 1, 
        activation = None, 
        groups = None,
        padding = 'same',
        kernel_size = 3):
        super().__init__()
        exp_channels = scale * in_channels
        self.act = activation or nn.Identity()
        
        self.expand = nn.Conv2d(in_channels, exp_channels, 1)
        self.dw = nn.Conv2d(exp_channels, exp_channels, kernel_size, 
                            padding = 'same', groups = groups or exp_channels)
        self.pw = nn.Conv2d(exp_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.act(self.expand(x))
        x = self.dw(x)
        x = self.pw(x)
        return x
