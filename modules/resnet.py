from modules.conv_blocks import ConvBlock
from torch.nn.utils import weight_norm
from torch import nn
import torch
import math
from utils.general import fused_gate_op
from utils.modules import select_activation, get_padding, TTSModule
from torch.utils.checkpoint import checkpoint


class ResidualLayer(TTSModule):
    """
    Implements a single residual layer using one or two convolutional blocks with activation functions.

    This layer applies one or two convolutional blocks to the input, with residual connections.
    It's designed to allow the network to learn residual functions with reference to the layer inputs,
    which helps in training deeper networks.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the ResidualLayer.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['conv_block']
        optional_params = {'activation_function': 'leaky_relu',
                           'dual': True}
        self.update_keys(mandatory_keys, optional_params)
        super(ResidualLayer, self).__init__(module_config, global_config)

        # Create the first convolutional block and its activation
        self.conv_layers = nn.ModuleList([ConvBlock(self.module_config['conv_block'], global_config)])
        self.activations = nn.ModuleList([select_activation(self.module_config['activation_function'])])

        # If dual is True, create a second convolutional block with dilation=1
        if self.module_config['dual']:
            conv_block_config = self.module_config['conv_block'].copy()
            conv_block_config['dilation'] = 1
            self.conv_layers.append(ConvBlock(conv_block_config, global_config))
            self.activations.append(select_activation(self.module_config['activation_function']))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualLayer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the residual layer.
        """
        for conv_layer, activation in zip(self.conv_layers, self.activations):
            x = activation(x)
            x = x + conv_layer(x)
        return x

    @staticmethod
    def sample_config() -> dict:
        """
        Provides a sample configuration for the ResidualLayer.
        """
        return {'conv_block': ConvBlock.sample_config()}


class ResidualBlock(TTSModule):
    """
    Stacks multiple ResidualLayer instances with varying dilation factors.

    This block creates a series of ResidualLayers, each with a different dilation factor.
    The varying dilation factors allow the network to capture dependencies at different scales,
    which is particularly useful for sequential data like audio or text.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the ResidualBlock.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['res_layer', 'dilation_list']
        self.update_keys(mandatory_keys)
        super(ResidualBlock, self).__init__(module_config, global_config)

        # Create a list of ResidualLayers with different dilation factors
        self.layers = nn.ModuleList()
        for dilation in self.module_config['dilation_list']:
            res_layer_config = self.module_config['res_layer'].copy()
            res_layer_config['dilation'] = dilation
            self.layers.append(ResidualLayer(res_layer_config, global_config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through all residual layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x / len(self.layers)

    @staticmethod
    def sample_config() -> dict:
        """
        Provides a sample configuration for the ResidualBlock.
        """
        return {'res_layer': ResidualLayer.sample_config(),
                'dilation_list': [1, 2, 4, 8, 16]}


class WaveNetLayer(TTSModule):
    """
    Implements a single layer of the WaveNet architecture with optional time and global conditioning.

    This layer is a key component of the WaveNet model, featuring dilated causal convolutions,
    gated activation units, and skip connections. It also supports various forms of conditioning,
    making it highly flexible for different audio generation tasks.

    Wavenet original paper (Google DeepMind, 2016) : https://arxiv.org/abs/1609.03499
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the WaveNetLayer.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['hidden_channels', 'output_channels', 'kernel_size', 'dilation']
        optional_params = {'dropout': 0.3,
                           'use_wn': False,
                           'global_cond_dim': 0,
                           'local_cond_dim': 0,
                           'time_emb_dim': 0}
        self.update_keys(mandatory_keys, optional_params)
        super(WaveNetLayer, self).__init__(module_config, global_config)

        # Initialize main convolutional layer
        padding = get_padding(self.module_config['kernel_size'], 1, self.module_config['dilation'])
        self.norm_f = weight_norm if self.module_config['use_wn'] else lambda x: x
        self.conv = self.norm_f(nn.Conv1d(self.module_config['hidden_channels'],
                                          2 * self.module_config['hidden_channels'],
                                          self.module_config['kernel_size'], padding=padding,
                                          dilation=self.module_config['dilation']))

        # Initialize skip and residual connection layer
        self.skip_res_layers = self.norm_f(nn.Conv1d(self.module_config['hidden_channels'],
                                                     self.module_config['output_channels'], kernel_size=1))
        self.dropout = nn.Dropout(self.module_config['dropout'])

        # Initialize optional conditioning layers
        self.time_emb_layer = None
        self.g_cond_layer = None
        self.loc_cond_layer = None

        if self.module_config['time_emb_dim'] > 0:
            self.time_emb_layer = self.norm_f(nn.Linear(self.module_config['time_emb_dim'],
                                                        self.module_config['hidden_channels']))
        if self.module_config['global_cond_dim'] > 0:
            self.g_cond_layer = self.norm_f(nn.Conv1d(self.module_config['global_cond_dim'],
                                                      2 * self.module_config['hidden_channels'],
                                                      kernel_size=1))
        if self.module_config['local_cond_dim'] > 0:
            self.loc_cond_layer = self.norm_f(nn.Conv1d(self.module_config['local_cond_dim'],
                                                        2 * self.module_config['hidden_channels'],
                                                        kernel_size=1))

    def checkpoint_fn(self, x: torch.Tensor, g_cond: torch.Tensor | None = None,
                      loc_cond: torch.Tensor | None = None, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        Checkpoint function for the WaveNetLayer.

        Args:
            x: Input tensor.
            g_cond: Global conditioning tensor.
            loc_cond: Local conditioning tensor.
            time_emb: Time embedding tensor.

        Returns:
            Processed tensor.
        """
        if self.time_emb_layer is not None and time_emb is not None:
            x = x + self.time_emb_layer(time_emb).unsqueeze(-1)
        x = self.conv(x)
        x = self.dropout(x)
        if self.g_cond_layer is not None and g_cond is not None:
            x = x + self.g_cond_layer(g_cond)
        if self.loc_cond_layer is not None and loc_cond is not None:
            x = x + self.loc_cond_layer(loc_cond)
        x = fused_gate_op(x)
        x = self.skip_res_layers(x)
        return x

    def forward(self, x: torch.Tensor, g_cond: torch.Tensor | None = None,
                loc_cond: torch.Tensor | None = None, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the WaveNetLayer. Uses checkpointing for memory efficiency.

        Args:
            x: Input tensor.
            g_cond: Global conditioning tensor.
            loc_cond: Local conditioning tensor.
            time_emb: Time embedding tensor.

        Returns:
            Output tensor after passing through the WaveNetLayer.
        """
        def forward_fn(x):
            return self.checkpoint_fn(x, g_cond, loc_cond, time_emb)
        x = checkpoint(forward_fn, x, use_reentrant=False)
        return x

    @staticmethod
    def sample_config() -> dict:
        """
        Provides a sample configuration for the WaveNetLayer.
        """
        return {'hidden_channels': 256, 'output_channels': 256, 'kernel_size': 3, 'dilation': 1}


class GenericResBlock(TTSModule):
    """
    Stacks multiple layers of a specified type, typically used for WaveNet-like architectures.

    This block is a generalization of the WaveNet architecture, allowing for the stacking of
    any specified layer type (default is WaveNetLayer) with customizable dilation patterns.
    It's designed to be highly flexible and can be adapted for various sequential modeling tasks.
    """
    def __init__(self, module_config: dict, global_config: dict, layer_module: nn.Module = WaveNetLayer):
        """
        Initialize the GenericResBlock.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
            layer_module: Type of layer to stack.
        """
        mandatory_keys = ['n_layers', 'dilation_rate', 'dilation_cycle', 'output_channels', 'layer_config']
        self.update_keys(mandatory_keys)
        super(GenericResBlock, self).__init__(module_config, global_config)
        self.layers = nn.ModuleList()
        layer_config = self.module_config['layer_config']
        layer_config['output_channels'] = 2 * self.module_config['output_channels']
        for i in range(self.module_config['n_layers']):
            dilation = self.module_config['dilation_rate']**(i % self.module_config['dilation_cycle'])
            layer_config['dilation'] = dilation
            if i < self.module_config['n_layers'] - 1:
                self.layers.append(layer_module(layer_config, global_config))
            else:
                layer_config['output_channels'] = self.module_config['output_channels']
                self.layers.append(layer_module(layer_config, global_config))

    def input_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Preprocessed input tensor.
        """
        return x

    def output_postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Postprocess the output tensor.

        Args:
            x: Output tensor.

        Returns:
            Postprocessed output tensor.
        """
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor, g_cond: torch.Tensor | None = None,
                loc_cond: torch.Tensor | None = None, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the GenericResBlock.

        Args:
            x: Input tensor.
            mask: Mask tensor.
            g_cond: Global conditioning tensor.
            loc_cond: Local conditioning tensor.
            time_emb: Time embedding tensor.

        Returns:
            Output tensor after passing through all layers.
        """
        x = self.input_preprocess(x)
        skip_out = None
        for i, res_layer in enumerate(self.layers):
            out_res = res_layer(x, g_cond=g_cond, loc_cond=loc_cond, time_emb=time_emb)
            if i < self.module_config['n_layers'] - 1:
                skip, res = out_res.split(out_res.size(1)//2, dim=1)
                x = x + res / math.sqrt(2.0) * mask
                if skip_out is None:
                    skip_out = skip
                else:
                    skip_out = skip_out + skip
            else:
                skip_out = skip_out + x
        x = skip_out / math.sqrt(len(self.layers)) * mask
        x = self.output_postprocess(x)
        return x * mask

    @staticmethod
    def sample_config() -> dict:
        """
        Provides a sample configuration for the GenericResBlock.
        """
        return {'n_layers': 10, 'dilation_rate': 2, 'dilation_cycle': 3, 'output_channels': 256,
                'layer_config': WaveNetLayer.sample_config()}
