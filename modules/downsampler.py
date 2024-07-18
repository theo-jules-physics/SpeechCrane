import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm
from utils.modules import select_activation, TTSModule


class DownsamplingLayer(TTSModule):
    """
    DownsamplingLayer applies downsampling using either 1D or 2D convolution, with optional weight normalization or
    spectral normalization and a specified activation function.

    Args:
        module_config (dict): Configuration dictionary for the module.
            - input_channels (int): Number of input channels.
            - output_channels (int): Number of output channels.
            - kernel_size (int): Size of the convolutional kernel.
            - stride (int): Stride for the convolution.
            - padding (int): Padding for the convolution.
            - groups (int): Number of groups for the convolution.
            - layer_type (str): Type of the layer, either 'period' or 'scale'.
            - activation_function (str, optional): Type of activation function to use. Default is 'leaky_relu'.
            - use_spec_norm (bool, optional): If True, applies spectral normalization. Default is False.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['input_channels', 'output_channels', 'kernel_size', 'stride', 'padding', 'groups',
                          'layer_type']
        optional_params = {'activation_function': 'leaky_relu',
                           'use_spec_norm': False}
        self.update_keys(mandatory_keys, optional_params)
        super(DownsamplingLayer, self).__init__(module_config, global_config)
        self.norm_f = weight_norm if not self.module_config['use_spec_norm'] else spectral_norm
        if self.module_config['layer_type'] == 'period':
            self.conv_layer = nn.Conv2d(self.module_config['input_channels'], self.module_config['output_channels'],
                                        (self.module_config['kernel_size'], 1), (self.module_config['stride'], 1),
                                        (self.module_config['padding'], 0))
        elif self.module_config['layer_type'] == 'scale':
            self.conv_layer = nn.Conv1d(self.module_config['input_channels'], self.module_config['output_channels'],
                                        self.module_config['kernel_size'], self.module_config['stride'],
                                        padding=self.module_config['padding'], groups=self.module_config['groups'])
        else:
            layer_type = self.module_config['layer_type']
            raise ValueError(f'Unknown layer type: {layer_type}')
        self.conv_layer = self.norm_f(self.conv_layer)
        self.leaky_relu = select_activation(self.module_config['activation_function'])

    def forward(self, x):
        return self.leaky_relu(self.conv_layer(x))

    @staticmethod
    def sample_config():
        return {'input_channels': 256, 'output_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1,
                'groups': 1, 'layer_type': 'scale'}
