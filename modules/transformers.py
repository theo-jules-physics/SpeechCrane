import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.modules import convert_pad_shape, modulate, TTSModule


class FFN(TTSModule):
    """
    Feed-Forward Network (FFN) with two convolutional layers, optional causal padding, activation, and dropout.

    This module is typically used in transformer architectures as part of the position-wise feed-forward networks.
    It applies two 1D convolutions with an activation function in between, and optional dropout.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the FFN.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['input_channels', 'output_channels', 'hidden_channels', 'kernel_size']
        optional_params = {'dropout': 0.,
                           'activation_function': None,
                           'causal': False}
        self.update_keys(mandatory_keys, optional_params)
        super().__init__(module_config, global_config)

        # Choose padding type based on causal flag
        self.padding = self._causal_padding if self.module_config['causal'] else self._same_padding

        # Initialize convolutional layers
        self.conv_1 = nn.Conv1d(self.module_config['input_channels'], self.module_config['hidden_channels'],
                                kernel_size=self.module_config['kernel_size'])
        self.conv_2 = nn.Conv1d(self.module_config['hidden_channels'], self.module_config['output_channels'],
                                kernel_size=self.module_config['kernel_size'])
        self.drop = nn.Dropout(self.module_config['dropout'])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FFN.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.

        Returns:
            Output tensor after passing through the FFN.
        """
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies causal padding to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Padded input tensor.
        """
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x

    def _same_padding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies same padding to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Padded input tensor.
        """
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, convert_pad_shape(padding))
        return x

    @staticmethod
    def sample_config() -> dict:
        """
        Provides a sample configuration for the FFN.
        """
        return {'input_channels': 256, 'output_channels': 256, 'hidden_channels': 512, 'kernel_size': 3}


class FFTBlock(TTSModule):
    """
    FFTBlock implements a transformer block with multihead attention, feed-forward network, and optional global conditioning.

    This block is a key component in transformer-based architectures, containing self-attention and a position-wise
    feed-forward network, with additional support for global conditioning.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the FFTBlock.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['embed_dim', 'num_heads', 'kernel_size', 'hidden_channels']
        optional_params = {'dropout': 0.,
                           'global_cond_dim': 0}
        self.update_keys(mandatory_keys, optional_params)
        super(FFTBlock, self).__init__(module_config, global_config)
        embed_dim = self.module_config['embed_dim']
        self.attention = nn.MultiheadAttention(embed_dim, self.module_config['num_heads'],
                                               dropout=self.module_config['dropout'], batch_first=True)
        ffn_config = {'input_channels': embed_dim,
                      'output_channels': embed_dim,
                      'hidden_channels': self.module_config['hidden_channels'],
                      'kernel_size': self.module_config['kernel_size'],
                      'dropout': self.module_config['dropout']}
        self.ffn = FFN(ffn_config, global_config)
        self.dropout = nn.Dropout(self.module_config['dropout'])
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if self.module_config['global_cond_dim'] > 0:
            self.g_cond_proj_attn = nn.Conv1d(self.module_config['global_cond_dim'], embed_dim, kernel_size=1)
            self.g_cond_proj_ffn = nn.Conv1d(self.module_config['global_cond_dim'], embed_dim, kernel_size=1)
        else:
            self.g_cond_proj_attn = None
            self.g_cond_proj_ffn = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor, g_cond: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the FFTBlock.

        Args:
            x: Input tensor.
            mask: Mask tensor.
            g_cond: Global conditioning tensor (optional).

        Returns:
            Output tensor after passing through the FFTBlock.
        """
        # Apply global conditioning if provided
        g_cond_attn = self.g_cond_proj_attn(g_cond).mT if g_cond is not None else 0
        g_cond_ffn = self.g_cond_proj_ffn(g_cond).mT if g_cond is not None else 0

        # Self-attention layer
        x = x.mT
        residual = x
        x = x + g_cond_attn
        x, _ = self.attention(x, x, x, key_padding_mask=~mask.squeeze(1))
        x = residual + self.dropout(x)
        x = self.norm1(x)

        # Feed-forward network
        residual = x
        x = x + g_cond_ffn
        x = self.ffn(x.mT, mask).mT
        x = residual + self.dropout(x)
        x = self.norm2(x)

        return (x.mT) * mask

    @staticmethod
    def sample_config() -> dict:
        """
        Provides a sample configuration for the FFTBlock.
        """
        return {'embed_dim': 256, 'num_heads': 4, 'kernel_size': 3, 'hidden_channels': 512}


class RelativePositionTransformer(TTSModule):
    """
    RelativePositionTransformer implements a transformer with relative position encoding and optional global conditioning.

    This transformer architecture uses multiple FFTBlocks and incorporates relative positional encoding,
    making it suitable for tasks where relative positions are important, such as in speech synthesis.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the RelativePositionTransformer.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['num_blocks', 'hidden_channels', 'fft_block_config']
        optional_params = {'dropout': 0.1,
                           'global_cond_dim': 0}
        self.update_keys(mandatory_keys, optional_params)
        super(RelativePositionTransformer, self).__init__(module_config, global_config)
        self.dropout = nn.Dropout(self.module_config['dropout'])

        # Global conditioning layer
        if self.module_config['global_cond_dim'] > 0:
            self.g_cond_layer = nn.Conv1d(self.module_config['global_cond_dim'],
                                          self.module_config['hidden_channels'], 1)

        # Stack of FFTBlocks
        self.fft_blocks = nn.ModuleList([FFTBlock(self.module_config['fft_block_config'], global_config)
                                         for _ in range(self.module_config['num_blocks'])])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g_cond: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the RelativePositionTransformer.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.
            g_cond: Global conditioning tensor (optional).

        Returns:
            Output tensor after passing through the transformer.
        """
        for fft_block in self.fft_blocks:
            x = fft_block(x, x_mask)
        if g_cond is not None:
            g_cond = self.g_cond_layer(g_cond)
            x = x + g_cond
        return x

    @staticmethod
    def sample_config() -> dict:
        """
        Provides a sample configuration for the RelativePositionTransformer.
        """
        return {'num_blocks': 3, 'hidden_channels': 256, 'fft_block_config': FFTBlock.sample_config()}
