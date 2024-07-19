import torch.nn as nn
import torch
from modules.resnet import ResidualBlock
from modules.decoders.base import BaseDecoder
from modules.upsampler import UpsampleBlock
from utils.modules import get_padding, TTSModule


class MRFBlock(TTSModule):
    """
    Multi-Receptive Field (MRF) block for capturing features at various scales.

    This block enhances the model's capability to process input at different receptive fields
    by utilizing multiple parallel paths with varying kernel sizes and dilation rates. It's
    particularly useful in audio generation tasks where capturing both local and global
    context is crucial for high-quality output.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the MRF block.

        Args:
            module_config: Configuration dictionary for the MRF block.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['res_block', 'kernel_sizes', 'dilation_rates']
        self.update_keys(mandatory_keys)
        super(MRFBlock, self).__init__(module_config, global_config)

        self.nbr_blocks = len(self.module_config['kernel_sizes'])
        self.blocks = nn.ModuleList()
        res_block_config = self.module_config['res_block']
        zip_loop = zip(self.module_config['kernel_sizes'], self.module_config['dilation_rates'])
        # Create multiple residual blocks with different kernel sizes and dilation rates
        for kernel_size, dilation_rate in zip_loop:
            res_block_config['res_layer']['conv_block']['kernel_size'] = kernel_size
            res_block_config['dilation_list'] = dilation_rate
            self.blocks.append(ResidualBlock(res_block_config, global_config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MRF block.

        Args:
            x: Input tensor.

        Returns:
            Processed tensor after passing through all parallel paths and averaging.
        """
        return sum(block(x) for block in self.blocks) / self.nbr_blocks

    @staticmethod
    def sample_config() -> dict:
        """
        Provide a sample configuration for the MRF block.
        """
        return {'res_block': ResidualBlock.sample_config(),
                'kernel_sizes': [3, 5, 7],
                'dilation_rates': [1, 3, 5]}


class HiFiGANDecoder(BaseDecoder):
    """
    HiFiGAN Decoder for high-fidelity waveform generation in text-to-speech systems.

    This decoder implements a hierarchical structure inspired by the HiFiGAN architecture,
    combining upsampling blocks with Multi-Receptive Field (MRF) blocks. It progressively
    increases temporal resolution while refining audio features, allowing for the generation
    of high-quality audio waveforms from mel-spectrogram inputs.

    Original paper for HiFiGAN (Kong et al. 2020): https://arxiv.org/abs/2010.05646.
    Inspired by the official implementation: https://github.com/jik876/hifi-gan.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the HiFiGAN decoder.

        Args:
            module_config: Configuration dictionary for the HiFiGAN decoder.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['upsample_kernels', 'upsample_strides', 'postprocess_net', 'mrf_block']
        self.update_keys(mandatory_keys)
        super(HiFiGANDecoder, self).__init__(module_config, global_config)

        base_upsample_config = {}
        base_mrf_block_config = self.module_config['mrf_block']
        zip_loop = zip(self.module_config['upsample_kernels'], self.module_config['upsample_strides'])
        init_out = self.module_config['preprocess_net']['out_channels']

        # Create a series of upsampling and MRF blocks
        for i, (up_k_size, up_rate) in enumerate(zip(self.module_config['upsample_kernels'], self.module_config['upsample_strides'])):
            channels = init_out // 2**i
            base_upsample_config['input_channels'] = channels
            base_upsample_config['output_channels'] = channels // 2
            base_upsample_config['kernel_size'] = up_k_size
            base_upsample_config['stride'] = up_rate
            self.dec_blocks.append(UpsampleBlock(base_upsample_config, global_config))
            conv_block_config = {'input_channels': channels // 2,
                                 'output_channels': channels // 2}
            base_mrf_block_config['res_block'] = {'res_layer': {'conv_block': conv_block_config}}
            self.dec_blocks.append(MRFBlock(base_mrf_block_config, global_config))

        # Final convolutional layer to produce the output waveform
        self.post_net = self.norm_f(nn.Conv1d(channels // 2, 1,
                                              kernel_size=self.module_config['postprocess_net']['kernel_size'],
                                              stride=self.module_config['postprocess_net']['stride'],
                                              padding=get_padding(self.module_config['postprocess_net']['kernel_size']),
                                              bias=False))

    @staticmethod
    def sample_config() -> dict:
        """
        Provide a sample configuration for the HiFiGAN decoder.
        """
        return {'upsample_kernels': [10, 8, 8, 4],
                'upsample_strides': [5, 4, 4, 2],
                'preprocess_net': {'in_channels': 192, 'out_channels': 256, 'kernel_size': 7, 'stride': 1},
                'postprocess_net': {'kernel_size': 7, 'stride': 1},
                'mrf_block': MRFBlock.sample_config()}
