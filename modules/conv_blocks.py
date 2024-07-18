import torch.nn as nn
from torch.nn.utils import weight_norm
from utils.modules import select_activation, get_padding
from torch.nn.utils.parametrizations import weight_norm
from utils.modules import TTSModule


class ConvBlock(TTSModule):
    """
    ConvBlock implements a single convolutional block with optional layer normalization, dropout, and activation.

    Args:
        module_config (dict): Dictionary containing the configuration for the convolutional block.
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels.
            - kernel_size (int): Size of the convolutional kernel.
            - dilation (int, optional): Dilation factor for the convolution. Default is 1.
            - activation (str, optional): Type of activation function to use. Default is 'leaky_relu'.
            - use_wn (bool, optional): If True, applies weight normalization. Default is False.
            - layer_norm (bool, optional): If True, applies layer normalization. Default is False.
            - dropout (float, optional): Dropout probability. Default is 0.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['input_channels', 'output_channels', 'kernel_size']
        optional_params = {'dilation': 1,
                           'activation_function': 'relu',
                           'use_wn': False,
                           'layer_norm': False,
                           'dropout': 0.0}
        self.update_keys(mandatory_keys, optional_params)
        super(ConvBlock, self).__init__(module_config, global_config)
        padding = get_padding(self.module_config['kernel_size'], 1, self.module_config['dilation'])
        self.norm_f = weight_norm if self.module_config['use_wn'] else lambda x: x
        self.conv = self.norm_f(nn.Conv1d(self.module_config['input_channels'], self.module_config['output_channels'],
                                          kernel_size=self.module_config['kernel_size'], padding=padding,
                                          dilation=self.module_config['dilation']))
        self.norm = nn.LayerNorm(self.module_config['out_channels']) if self.module_config['layer_norm'] else nn.Identity()
        self.dropout = nn.Dropout(self.module_config['dropout'])
        self.activation = select_activation(self.module_config['activation_function'])

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = x.mT
        x = self.norm(x)
        x = self.dropout(x)
        return x.mT

    @staticmethod
    def sample_config():
        return {'input_channels': 256, 'output_channels': 256, 'kernel_size': 3}


class DDSLayer(TTSModule):
    """
    DDSLayer (Dilated and Depth-Separable Convolutional Layer) implements a series of dilated and depth-separable
    convolutional layers.

    Args:
        module_config (dict): Dictionary containing the configuration for the DDSLayer.
            - channels (int): Number of input and output channels.
            - kernel_size (int): Size of the convolutional kernel.
            - dilation (int): Dilation factor for the convolution.
            - dropout (float, optional): Dropout probability. Default is 0.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['channels', 'kernel_size', 'dilation']
        optional_params = {'dropout': 0.0}
        self.update_keys(mandatory_keys, optional_params)
        super().__init__(module_config, global_config)
        conv_config = {'input_channels': self.module_config['channels'],
                       'output_channels': self.module_config['channels'],
                       'kernel_size': self.module_config['kernel_size'],
                       'dilation': self.module_config['dilation'],
                       'dropout': self.module_config['dropout']}
        self.dss_layers = nn.ModuleList()
        self.dss_layers.append(ConvBlock(conv_config, global_config))
        conv_config['dilation'] = 1
        self.dss_layers.append(ConvBlock(conv_config, global_config))

    def forward(self, x, x_mask):
        y = x * x_mask
        for layer in self.dss_layers:
            y = layer(y)
        x = x + y
        return x * x_mask

    @staticmethod
    def sample_config():
        return {'channels': 256, 'kernel_size': 3, 'dilation': 1}


class DDSConv(TTSModule):
    """
    DDSConv implements a stack of DDSLayers with increasing dilation factors.

    Args:
        module_config (dict): Dictionary containing the configuration for the DDSConv stack.
            - channels (int): Number of input and output channels.
            - kernel_size (int): Size of the convolutional kernel.
            - n_layers (int): Number of DDSLayers in the stack.
            - dropout (float, optional): Dropout probability. Default is 0.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['channels', 'kernel_size', 'n_layers']
        optional_params = {'dropout': 0.0}
        self.update_keys(mandatory_keys, optional_params)
        super().__init__(module_config, global_config)
        self.convs = nn.ModuleList()
        conv_config = {'channels': self.module_config['channels'],
                       'kernel_size': self.module_config['kernel_size'],
                       'dropout': self.module_config['dropout']}
        for i in range(self.module_config['n_layers']):
            dilation = self.module_config['kernel_size'] ** i
            conv_config['dilation'] = dilation
            self.convs.append(DDSLayer(conv_config, global_config))

    def forward(self, x, x_mask, g_cond=None):
        if g_cond is not None:
            x = x + g_cond
        for i in range(self.n_layers):
            x = self.convs[i](x, x_mask)
        return x * x_mask

    @staticmethod
    def sample_config():
        return {'channels': 256, 'kernel_size': 3, 'n_layers': 5}
