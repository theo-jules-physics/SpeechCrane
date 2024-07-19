import torch.nn as nn
import torch
from torch.nn.utils import weight_norm, spectral_norm
from utils.modules import select_activation, TTSModule


class DownsamplingLayer(TTSModule):
    """
    DownsamplingLayer applies downsampling using either 1D or 2D convolution.

    This layer is designed to reduce the spatial or temporal dimensions of the input,
    with options for different types of normalization and activation functions. It's
    particularly useful in discriminator networks or for reducing the resolution of
    feature maps in generator networks.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the DownsamplingLayer.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['input_channels', 'output_channels', 'kernel_size', 'stride', 'padding', 'groups',
                          'layer_type']
        optional_params = {'activation_function': 'leaky_relu',
                           'use_spec_norm': False}
        self.update_keys(mandatory_keys, optional_params)
        super(DownsamplingLayer, self).__init__(module_config, global_config)

        # Choose between weight normalization and spectral normalization
        self.norm_f = weight_norm if not self.module_config['use_spec_norm'] else spectral_norm

        # Create the convolutional layer based on the layer type
        if self.module_config['layer_type'] == 'period':
            self.conv_layer = nn.Conv2d(self.module_config['input_channels'], self.module_config['output_channels'],
                                        (self.module_config['kernel_size'], 1), (self.module_config['stride'], 1),
                                        (self.module_config['padding'], 0))
        elif self.module_config['layer_type'] == 'scale':
            self.conv_layer = nn.Conv1d(self.module_config['input_channels'], self.module_config['output_channels'],
                                        self.module_config['kernel_size'], self.module_config['stride'],
                                        padding=self.module_config['padding'], groups=self.module_config['groups'])
        else:
            raise ValueError(f'Unknown layer type: {self.module_config["layer_type"]}')

        # Apply normalization to the convolutional layer
        self.conv_layer = self.norm_f(self.conv_layer)
        # Select the activation function
        self.leaky_relu = select_activation(self.module_config['activation_function'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DownsamplingLayer.

        Args:
            x: Input tensor.

        Returns:
            Downsampled tensor after convolution and activation.
        """
        return self.leaky_relu(self.conv_layer(x))

    @staticmethod
    def sample_config() -> dict:
        """
        Provides a sample configuration for the DownsamplingLayer.
        """
        return {'input_channels': 256, 'output_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1,
                'groups': 1, 'layer_type': 'scale'}
