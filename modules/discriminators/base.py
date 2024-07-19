import torch.nn as nn
import torch
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from utils.modules import remove_weight_norm_recursively, select_activation, TTSModule


class BaseDiscriminator(TTSModule):
    """
    Base class for building discriminators with customizable convolutional layers and normalization.

    This class provides a foundation for creating various discriminator architectures
    by allowing flexible configuration of convolutional layers and normalization techniques.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the BaseDiscriminator.

        Args:
            module_config: Configuration dictionary for the discriminator module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['channels', 'kernel_sizes', 'strides']
        optional_params = {'dilations': 1,
                           'paddings': None,
                           'groups': None,
                           'use_spec_norm': False,
                           'activation_function': 'leaky_relu'}
        self.update_keys(mandatory_keys, optional_params)
        super(BaseDiscriminator, self).__init__(module_config, global_config)

        self.norm_f = weight_norm if not self.module_config['use_spec_norm'] else spectral_norm
        self.activation = select_activation(self.module_config['activation_function'])
        self._set_layers(global_config)

    def _set_layers(self, global_config: dict):
        """
        Sets the layers for the discriminator.
        Initializes the pre-processing network, convolutional layers, and post-processing network.

        Args:
            global_config: Global configuration dictionary.
        """
        self.pre_net = nn.Identity()
        self.layers = nn.ModuleList()
        self.post_net = nn.Identity()

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare the input tensor by passing it through the pre-processing network.

        Args:
            x: Input tensor.

        Returns:
            Processed input tensor.
        """
        return self.pre_net(x)

    def _process_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the output tensor through the post-processing network.

        Args:
            x: Output tensor from the last convolutional layer.

        Returns:
            Processed tensor with flattened last two dimensions.
        """
        x = self.post_net(x)
        return x.flatten(-2, -1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the discriminator.

        Args:
            x: Input tensor.

        Returns:
            Tuple containing:
            - Output tensor after passing through all layers.
            - List of feature maps from intermediate layers.
        """
        fmap = []
        x = self._prepare_input(x)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            fmap.append(x)
        x = self._process_output(x)
        return x, fmap

    @staticmethod
    def sample_config() -> dict:
        """
        Provide a sample configuration for the BaseDiscriminator.
        """
        return {'channels': [64, 128, 256, 512],
                'kernel_sizes': [3, 3, 3, 3],
                'strides': [2, 2, 2, 2]}
