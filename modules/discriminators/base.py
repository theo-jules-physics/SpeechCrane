import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from utils.modules import remove_weight_norm_recursively, select_activation, TTSModule


class BaseDiscriminator(TTSModule):
    """
    BaseDiscriminator provides a base class for building discriminators with customizable convolutional layers and normalization.

    Args:
        module_config (dict): Configuration dictionary for the discriminator module.
            - channels (list): List of integers representing the number of channels in each convolutional layer.
            - kernel_sizes (list): List of integers representing the kernel sizes in each convolutional layer.
            - strides (list): List of integers representing the strides in each convolutional layer.
            - paddings (list): List of integers representing the paddings in each convolutional layer.
            - dilations (list): List of integers representing the dilations in each convolutional layer.
            - groups (list): List of integers representing the groups in each convolutional layer.
            - use_spec_norm (bool): Boolean value indicating whether to use spectral normalization.
    """
    def __init__(self, module_config, global_config):
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

    def _set_layers(self, global_config):
        """
        Sets the layers for the discriminator. Initializes the pre-processing network, convolutional layers, and post-processing network.
        """
        self.pre_net = nn.Identity()
        self.layers = nn.ModuleList()
        self.post_net = nn.Identity()

    def _prepare_input(self, x):
        """
        Prepares the input tensor by passing it through the pre-processing network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Processed input tensor.
        """
        return self.pre_net(x)

    def _process_output(self, x):
        """
        Processes the output tensor by passing it through the post-processing network and flattening the last two dimensions.

        Args:
            x (Tensor): Output tensor from the last convolutional layer.

        Returns:
            Tensor: Processed output tensor.
        """
        x = self.post_net(x)
        return x.flatten(-2, -1)

    def forward(self, x):
        fmap = []
        x = self._prepare_input(x)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            fmap.append(x)
        x = self._process_output(x)
        return x, fmap

    def remove_weight_norm(self):
        remove_weight_norm_recursively(self)

    @staticmethod
    def sample_config():
        return {'channels': [64, 128, 256, 512],
                'kernel_sizes': [3, 3, 3, 3],
                'strides': [2, 2, 2, 2]}
