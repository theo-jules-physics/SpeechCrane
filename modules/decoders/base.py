import torch.nn as nn
import torch
from utils.modules import remove_weight_norm_recursively, select_activation, get_padding, TTSModule
from torch.nn.utils import weight_norm
from modules.transformers import FFTBlock


class BaseDecoder(TTSModule):
    """
    BaseDecoder provides a foundation for building decoders with customizable pre-net and decoding blocks.

    This class serves as a flexible base for various decoder architectures in text-to-speech models.
    It includes a pre-processing network, optional global conditioning, and a series of decoding blocks.
    """

    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the BaseDecoder.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['preprocess_net']
        optional_params = {'use_wn': False,
                           'tanh_out': True,
                           'activation_function': None,
                           'global_cond_dim': 0}
        self.update_keys(mandatory_keys, optional_params)
        super(BaseDecoder, self).__init__(module_config, global_config)

        self.tanh_out = self.module_config['tanh_out']
        self.norm_f = weight_norm if self.module_config['use_wn'] else lambda x: x

        # Pre-processing network
        self.pre_net = self.norm_f(nn.Conv1d(
            self.module_config['preprocess_net']['in_channels'],
            self.module_config['preprocess_net']['out_channels'],
            self.module_config['preprocess_net']['kernel_size'],
            self.module_config['preprocess_net']['stride'],
            get_padding(self.module_config['preprocess_net']['kernel_size'])
        ))

        # Global conditioning layer
        if self.module_config['global_cond_dim'] > 0:
            self.g_cond_layer = self.norm_f(nn.Conv1d(
                self.module_config['global_cond_dim'],
                self.module_config['preprocess_net']['out_channels'],
                1
            ))

        # Placeholder for decoder blocks (to be defined in subclasses)
        self.dec_blocks = nn.ModuleList()
        self.post_net = nn.Identity()
        self.activation = select_activation(self.module_config['activation_function'])

    def forward(self, x: torch.Tensor, g_cond: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the BaseDecoder.

        Args:
            x: Input tensor.
            g_cond: Global conditioning tensor (optional).

        Returns:
            Decoded output tensor.
        """
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

    @staticmethod
    def sample_config() -> dict:
        """Provides a sample configuration for the BaseDecoder."""
        return {'preprocess_net': {'in_channels': 192,
                                   'out_channels': 256,
                                   'kernel_size': 7,
                                   'stride': 1}}


class MelDecoder(TTSModule):
    """
    MelDecoder implements a decoder that uses FFT blocks and a post-net to generate Mel spectrograms.

    This decoder is specifically designed for generating mel spectrograms in text-to-speech models.
    It uses a series of FFT blocks followed by a post-processing network.
    """

    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the MelDecoder.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['num_layers', 'fft_block', 'embed_dim', 'n_mels']
        optional_params = {'dropout': 0.1}
        self.update_keys(mandatory_keys, optional_params)
        super(MelDecoder, self).__init__(module_config, global_config)

        # Stack of FFT blocks
        self.fft_blocks = nn.ModuleList([
            FFTBlock(self.module_config['fft_block'], global_config)
            for _ in range(self.module_config['num_layers'])
        ])

        # Post-processing network
        self.post_net = nn.Sequential(
            nn.Linear(self.module_config['embed_dim'], self.module_config['embed_dim']),
            nn.ReLU(),
            nn.Dropout(self.module_config['dropout']),
            nn.Linear(self.module_config['embed_dim'], self.module_config['n_mels'])
        )

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MelDecoder.

        Args:
            src: Source tensor.
            src_mask: Source mask tensor.

        Returns:
            Decoded mel spectrogram tensor.
        """
        x = src
        for fft_block in self.fft_blocks:
            x = fft_block(x, ~src_mask)
        x = self.post_net(x)
        return x

    def inference(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Inference method for the MelDecoder.

        This method is identical to the forward pass for this decoder.

        Args:
            src: Source tensor.
            src_mask: Source mask tensor.

        Returns:
            Decoded mel spectrogram tensor.
        """
        return self.forward(src, src_mask)

    @staticmethod
    def sample_config() -> dict:
        """Provides a sample configuration for the MelDecoder."""
        return {
            'num_layers': 3,
            'fft_block': FFTBlock.sample_config(),
            'embed_dim': 256,
            'n_mels': 80
        }