import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
from utils.modules import select_activation, get_padding
from torch.nn.utils.parametrizations import weight_norm
from utils.modules import TTSModule


class ConvBlock(TTSModule):
    """
    Implements a single convolutional block with optional layer normalization, dropout, and activation.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the ConvBlock.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvBlock.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the convolutional block.
        """
        x = self.conv(x)
        x = self.activation(x)
        x = x.mT
        x = self.norm(x)
        x = self.dropout(x)
        return x.mT

    @staticmethod
    def sample_config() -> dict:
        """
        Provides a sample configuration for the ConvBlock.
        """
        return {'input_channels': 256, 'output_channels': 256, 'kernel_size': 3}


class DDSLayer(TTSModule):
    """
    Implements a Dilated and Depth-Separable (DDS) Convolutional Layer.

    This layer combines dilated convolutions with depth-wise separable convolutions,
    which can increase the receptive field while maintaining computational efficiency.
    It consists of two ConvBlocks: the first with specified dilation, and the second
    with dilation of 1, effectively creating a depth-wise separable convolution.
    """

    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the DDSLayer.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['channels', 'kernel_size', 'dilation']
        optional_params = {'dropout': 0.0}
        self.update_keys(mandatory_keys, optional_params)
        super().__init__(module_config, global_config)

        # Create two ConvBlocks: one with specified dilation, another with dilation=1
        conv_config = {'input_channels': self.module_config['channels'],
                       'output_channels': self.module_config['channels'],
                       'kernel_size': self.module_config['kernel_size'],
                       'dilation': self.module_config['dilation'],
                       'dropout': self.module_config['dropout']}
        self.dss_layers = nn.ModuleList()
        self.dss_layers.append(ConvBlock(conv_config, global_config))
        conv_config['dilation'] = 1
        self.dss_layers.append(ConvBlock(conv_config, global_config))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DDSLayer.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.

        Returns:
            Output tensor after passing through the DDSLayer.
        """
        y = x * x_mask
        for layer in self.dss_layers:
            y = layer(y)
        x = x + y
        return x * x_mask

    @staticmethod
    def sample_config() -> dict:
        """
        Provides a sample configuration for the DDSLayer.
        """
        return {'channels': 256, 'kernel_size': 3, 'dilation': 1}


class DDSBlock(TTSModule):
    """
    Implements a stack of DDSLayers with increasing dilation factors.

    This block creates a deep network of Dilated and Depth-Separable Convolutions,
    where each layer has an increasing dilation factor. This architecture allows
    the network to capture long-range dependencies in the input sequence while
    maintaining a relatively small number of parameters.
    """

    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the DDSBlock.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['channels', 'kernel_size', 'n_layers']
        optional_params = {'dropout': 0.0}
        self.update_keys(mandatory_keys, optional_params)
        super().__init__(module_config, global_config)

        # Create a stack of DDSLayers with increasing dilation
        self.convs = nn.ModuleList()
        conv_config = {'channels': self.module_config['channels'],
                       'kernel_size': self.module_config['kernel_size'],
                       'dropout': self.module_config['dropout']}
        for i in range(self.module_config['n_layers']):
            dilation = self.module_config['kernel_size'] ** i
            conv_config['dilation'] = dilation
            self.convs.append(DDSLayer(conv_config, global_config))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g_cond: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the DDSConv.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.
            g_cond: Global conditioning tensor (optional).

        Returns:
            Output tensor after passing through all DDSLayers.
        """
        if g_cond is not None:
            x = x + g_cond
        for i in range(self.n_layers):
            x = self.convs[i](x, x_mask)
        return x * x_mask

    @staticmethod
    def sample_config() -> dict:
        """
        Provides a sample configuration for the DDSConv.
        """
        return {'channels': 256, 'kernel_size': 3, 'n_layers': 5}
