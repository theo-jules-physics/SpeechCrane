import torch.nn as nn
from torch.nn.utils import weight_norm
from utils.modules import select_activation, TTSModule, get_padding


class UpsampleBlock(TTSModule):
    """
    UpsampleBlock performs upsampling using a transposed convolutional layer, with optional weight normalization and activation.

    Args:
        module_config (dict): Dictionary containing the configuration for the upsampling block.
            - input_channels (int): Number of input channels.
            - output_channels (int): Number of output channels.
            - kernel_size (int): Size of the convolutional kernel.
            - stride (int): Stride size for the convolution.
            - dim (int, optional): Dimension of the convolution. Default is 1.
            - activation_function (str, optional): Type of activation function to use. Default is 'leaky_relu'.
            - use_wn (bool, optional): If True, applies weight normalization. Default is False.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['input_channels', 'output_channels', 'kernel_size', 'stride']
        optional_params = {'dim': 1, 'activation_function': 'leaky_relu', 'use_wn': False}
        self.update_keys(mandatory_keys, optional_params)
        super(UpsampleBlock, self).__init__(module_config, global_config)
        padding = get_padding(self.module_config['kernel_size'], self.module_config['stride'])
        self.norm_f = weight_norm if self.module_config['use_wn'] else lambda x: x
        if self.module_config['dim'] == 1:
            padding = get_padding(self.module_config['kernel_size'], self.module_config['stride'])
            self.up_conv_layer = nn.ConvTranspose1d(self.module_config['input_channels'],
                                                    self.module_config['output_channels'],
                                                    self.module_config['kernel_size'],
                                                    self.module_config['stride'], padding)
        elif self.module_config['dim'] == 2:
            self.up_conv_layer = nn.ConvTranspose2d(self.module_config['input_channels'],
                                                    self.module_config['output_channels'],
                                                    self.module_config['kernel_size'],
                                                    self.module_config['stride'], padding)
        self.up_conv_layer = self.norm_f(self.up_conv_layer)
        self.activation = select_activation(self.module_config['activation_function'])

    def forward(self, x):
        return self.up_conv_layer(self.activation(x))

    @staticmethod
    def sample_config():
        return {'input_channels': 256, 'output_channels': 256, 'kernel_size': 3, 'stride': 2, 'dim': 1, 'use_wn': False}


class MelUpsampler(TTSModule):
    """
    MelUpsampler stacks multiple UpsampleBlock layers to perform sequential upsampling on mel spectrograms.

    Args:
        upsampler_config (dict): Dictionary containing the configuration for the upsampler.
            - kernel_list (list): List of kernel sizes for each upsampling layer.
            - stride_list (list): List of stride sizes for each upsampling layer.
            - use_wn (bool): If True, applies weight normalization.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['kernel_list', 'stride_list', 'upsampler_block']
        optional_params = {'flow': False}
        self.update_keys(mandatory_keys, optional_params)
        super(MelUpsampler, self).__init__(module_config, global_config)
        zip_loop = zip(self.module_config['kernel_list'], self.module_config['stride_list'])
        self.blocks = nn.ModuleList()
        for kernel_size, stride in zip_loop:
            blocks_config = self.module_config['upsampler_block']
            blocks_config['kernel_size'] = kernel_size
            blocks_config['stride'] = stride
            self.blocks.append(UpsampleBlock(blocks_config, global_config))

    def forward(self, x):
        x = x.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        x = x.squeeze(1)
        return x

    @staticmethod
    def sample_config():
        return {'kernel_list': [3, 3, 3, 3],
                'stride_list': [2, 2, 2, 2],
                'upsampler_block': {'input_channels': 256, 'output_channels': 256},
                }
