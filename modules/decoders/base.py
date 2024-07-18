import torch.nn as nn
import torch
from utils.modules import remove_weight_norm_recursively, select_activation, get_padding, TTSModule
from torch.nn.utils import weight_norm
from modules.transformers import FFTBlock


class BaseDecoder(TTSModule):
    """
    BaseDecoder provides a base class for building decoders with customizable pre-net and decoding blocks.

    Args:
        module_config (dict): Configuration dictionary for the decoder module.
            - preprocess_net (dict): Configuration dictionary for the pre-net.
                - in_channels (int): Number of input channels.
                - out_channels (int): Number of output channels.
                - kernel_size (int): Kernel size for the convolutional layer.
                - stride (int): Stride for the convolutional layer.
            - use_wn (bool): Boolean value indicating whether to use weight normalization.
            - tanh_out (bool): Boolean value indicating whether to apply tanh activation to the output.
            - activation_function (str, optional): Type of activation function to use. Default is None.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['preprocess_net']
        optional_params = {'use_wn': False,
                           'tanh_out': True,
                           'activation_function': None,
                           'global_cond_dim': 0}
        self.update_keys(mandatory_keys, optional_params)
        super(BaseDecoder, self).__init__(module_config, global_config)

        self.tanh_out = self.module_config['tanh_out']
        self.norm_f = weight_norm if self.module_config['use_wn'] else lambda x: x
        self.pre_net = self.norm_f(nn.Conv1d(self.module_config['preprocess_net']['in_channels'],
                                             self.module_config['preprocess_net']['out_channels'],
                                             self.module_config['preprocess_net']['kernel_size'],
                                             self.module_config['preprocess_net']['stride'],
                                             get_padding(self.module_config['preprocess_net']['kernel_size'])))
        if self.module_config['global_cond_dim'] > 0:
            self.g_cond_layer = self.norm_f(nn.Conv1d(self.module_config['global_cond_dim'],
                                                      self.module_config['preprocess_net']['out_channels'],
                                                      1))

        self.dec_blocks = nn.ModuleList()
        self.post_net = nn.Identity()
        self.activation = select_activation(self.module_config['activation_function'])

    def forward(self, x, g_cond=None):
        x = self.pre_net(x)
        if g_cond is not None:
            x = x + self.g_cond_layer(g_cond)

        for block in self.dec_blocks:
            x = block(x)
        x = self.activation(x)
        x = self.post_net(x)
        if self.tanh_out:
            return torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm_recursively(self)

    @staticmethod
    def sample_config():
        return {'preprocess_net': {'in_channels': 192,
                                   'out_channels': 256,
                                   'kernel_size': 7,
                                   'stride': 1}}


class MelDecoder(TTSModule):
    """
    MelDecoder implements a decoder that uses FFT blocks and a post-net to generate Mel spectrograms.

    Args:
        module_config (dict): Configuration dictionary for the decoder module.
            - num_layers (int): Number of FFT blocks.
            - fft_block (dict): Configuration dictionary for the FFT block.
            - embed_dim (int): Dimension of the input embeddings.
            - n_mels (int): Number of Mel spectrogram channels.
            - dropout (float, optional): Dropout rate. Default is 0.1.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['num_layers', 'fft_block', 'embed_dim', 'n_mels']
        optional_params = {'dropout': 0.1}
        self.update_keys(mandatory_keys, optional_params)
        super(MelDecoder, self).__init__(module_config, global_config)
        self.fft_blocks = nn.ModuleList([FFTBlock(self.module_config['fft_block'], global_config)
                                         for _ in range(self.module_config['num_layers'])])
        self.post_net = nn.Sequential(
            nn.Linear(self.module_config['embed_dim'], self.module_config['embed_dim']),
            nn.ReLU(),
            nn.Dropout(self.module_config['dropout']),
            nn.Linear(self.module_config['embed_dim'], self.module_config['n_mels'])
        )

    def forward(self, src, src_mask):
        x = src
        for fft_block in self.fft_blocks:
            x = fft_block(x, ~src_mask)
        x = self.post_net(x)
        return x

    def inference(self, src, src_mask):
        return self.forward(src, src_mask)

    @staticmethod
    def sample_config():
        return {'num_layers': 3,
                'fft_block': FFTBlock.sample_config(),
                'embed_dim': 256,
                'n_mels': 80}