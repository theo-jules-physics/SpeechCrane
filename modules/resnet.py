from modules.conv_blocks import ConvBlock
from torch.nn.utils import weight_norm
from torch import nn
import math
from utils.general import fused_gate_op
from utils.modules import select_activation, get_padding, TTSModule
from torch.utils.checkpoint import checkpoint


class ResidualLayer(TTSModule):
    """
    ResidualLayer implements a single residual layer using one or two convolutional blocks with activation functions.

    Args:
        module_config (dict): Configuration dictionary for the module.
            - conv_block (dict): Configuration dictionary for the convolutional block.
            Necessitate 'input_channels' and 'output_channels'.
            - activation_function (str, optional): Type of activation function to use. Default is 'leaky_relu'.
            - dual (bool, optional): If True, uses dual convolutional blocks. Default is True.
    """

    def __init__(self, module_config, global_config):
        mandatory_keys = ['conv_block']
        optional_params = {'activation_function': 'leaky_relu',
                           'dual': True}
        self.update_keys(mandatory_keys, optional_params)
        super(ResidualLayer, self).__init__(module_config, global_config)

        self.conv_layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        self.conv_layers.append(ConvBlock(self.module_config['conv_block'], global_config))
        self.activations.append(select_activation(self.module_config['activation_function']))
        if self.module_config['dual']:
            conv_block_config = self.module_config['conv_block'].copy()
            conv_block_config['dilation'] = 1
            self.conv_layers.append(ConvBlock(conv_block_config, global_config))
            self.activations.append(select_activation(self.module_config['activation_function']))

    def forward(self, x):
        for conv_layer, activation in zip(self.conv_layers, self.activations):
            x = activation(x)
            x = x + conv_layer(x)
        return x

    @staticmethod
    def sample_config():
        return {'conv_block': ConvBlock.sample_config()}


class ResidualBlock(TTSModule):
    """
    ResidualBlock stacks multiple ResidualLayer instances with varying dilation factors.

    Args:
        module_config (dict): Configuration dictionary for the module.
            - res_layer (dict): Configuration dictionary for the ResidualLayer.
            - dilation_list (list of int): List of dilation factors for each ResidualLayer.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['res_layer', 'dilation_list']
        self.update_keys(mandatory_keys)
        super(ResidualBlock, self).__init__(module_config, global_config)
        self.layers = nn.ModuleList()
        for dilation in self.module_config['dilation_list']:
            res_layer_config = self.module_config['res_layer'].copy()
            res_layer_config['dilation'] = dilation
            self.layers.append(ResidualLayer(res_layer_config, global_config))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x / len(self.layers)

    @staticmethod
    def sample_config():
        return {'res_layer': ResidualLayer.sample_config(),
                'dilation_list': [1, 2, 4, 8, 16]}


class WaveNetLayer(TTSModule):
    """
    WaveNetLayer implements a single layer of the WaveNet architecture with optional time and global conditioning.

    Args:
        module_config (dict): Configuration dictionary for the module.
            - res_channels (int): Number of residual channels.
            - output_channels (int): Number of output channels.
            - kernel_size (int): Size of the convolutional kernel.
            - dilation (int): Dilation factor for the convolution.
            - dropout (float, optional): Dropout probability. Default is 0.
            - use_wn (bool, optional): If True, applies weight normalization. Default is False.
        g_cond_dim (int, optional): Dimension of the global conditioning. Default is 0.
        loc_cond_dim (int, optional): Dimension of the local conditioning. Default is 0.
        time_emb_dim (int, optional): Dimension of the time embedding. Default is 0.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['hidden_channels', 'output_channels', 'kernel_size', 'dilation']
        optional_params = {'dropout': 0.3,
                           'use_wn': False,
                           'global_cond_dim': 0,
                           'local_cond_dim': 0,
                           'time_emb_dim': 0}
        self.update_keys(mandatory_keys, optional_params)
        super(WaveNetLayer, self).__init__(module_config, global_config)
        padding = get_padding(self.module_config['kernel_size'], 1, self.module_config['dilation'])
        self.norm_f = weight_norm if self.module_config['use_wn'] else lambda x: x
        self.conv = self.norm_f(nn.Conv1d(self.module_config['hidden_channels'],
                                          2 * self.module_config['hidden_channels'],
                                          self.module_config['kernel_size'], padding=padding,
                                          dilation=self.module_config['dilation']))
        self.skip_res_layers = self.norm_f(nn.Conv1d(self.module_config['hidden_channels'],
                                                     self.module_config['output_channels'], kernel_size=1))
        self.dropout = nn.Dropout(self.module_config['dropout'])
        if self.module_config['time_emb_dim'] > 0:
            self.time_emb_layer = self.norm_f(nn.Linear(self.module_config['time_emb_dim'], self.module_config['hidden_channels']))
        else:
            self.time_emb_layer = None
        if self.module_config['global_cond_dim'] > 0:
            self.g_cond_layer = self.norm_f(nn.Conv1d(self.module_config['global_cond_dim'],
                                                      2 * self.module_config['hidden_channels'],
                                                      kernel_size=1))
        else:
            self.g_cond_layer = None
        if self.module_config['local_cond_dim'] > 0:
            self.loc_cond_layer = self.norm_f(nn.Conv1d(self.module_config['local_cond_dim'],
                                                        2 * self.module_config['hidden_channels'],
                                                        kernel_size=1))
        else:
            self.loc_cond_layer = None

    def checkpoint_fn(self, x, g_cond=None, loc_cond=None, time_emb=None):
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

    def forward(self, x, g_cond=None, loc_cond=None, time_emb=None):
        def forward_fn(x):
            return self.checkpoint_fn(x, g_cond, loc_cond, time_emb)
        x = checkpoint(forward_fn, x, use_reentrant=False)
        return x

    @staticmethod
    def sample_config():
        return {'hidden_channels': 256, 'output_channels': 256, 'kernel_size': 3, 'dilation': 1}


class GenericResBlock(TTSModule):
    """
    GenericResBlock stacks multiple layers of a specified type, typically used for WaveNet-like architectures.

    Args:
        layer_module (nn.Module): Type of layer to stack.
        module_config (dict): Configuration dictionary for the module.
            - n_layers (int): Number of layers to stack.
            - dilation_rate (int): Base dilation rate for the layers.
            - dilation_cycle (int): Number of layers before increasing the dilation rate.
            - output_channels (int): Number of output channels for the final layer.
            - layer_config (dict): Configuration dictionary for the layer.
        g_cond_dim (int, optional): Dimension of the global conditioning. Default is 0.
        loc_cond_dim (int, optional): Dimension of the local conditioning. Default is 0.
        time_emb_dim (int, optional): Dimension of the time embedding. Default is 0.
    """
    def __init__(self, module_config, global_config, layer_module=WaveNetLayer):
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

    def input_preprocess(self, x):
        return x

    def output_postprocess(self, x):
        return x

    def forward(self, x, mask, g_cond=None, loc_cond=None, time_emb=None):
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
    def sample_config():
        return {'n_layers': 10, 'dilation_rate': 2, 'dilation_cycle': 3, 'output_channels': 256,
                'layer_config': WaveNetLayer.sample_config()}
