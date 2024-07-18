import torch.nn as nn
from modules.resnet import ResidualBlock
from modules.decoders.base import BaseDecoder
from modules.upsampler import UpsampleBlock
from utils.modules import get_padding, TTSModule


class MRFBlock(TTSModule):
    """
    MRFBlock implements a Multi-Receptive Field block consisting of multiple residual blocks with different kernel sizes and dilation rates.

    Args:
        module_config (dict): Configuration dictionary for the MRF block.
            - res_block (dict): Configuration dictionary for the residual block.
            - kernel_sizes (list): List of kernel sizes for the residual blocks.
            - dilation_rates (list): List of dilation rates for the residual blocks.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['res_block', 'kernel_sizes', 'dilation_rates']
        self.update_keys(mandatory_keys)
        super(MRFBlock, self).__init__(module_config, global_config)
        self.nbr_blocks = len(self.module_config['kernel_sizes'])
        self.blocks = nn.ModuleList()
        res_block_config = self.module_config['res_block']
        zip_loop = zip(self.module_config['kernel_sizes'], self.module_config['dilation_rates'])
        for kernel_size, dilation_rate in zip_loop:
            res_block_config['res_layer']['conv_block']['kernel_size'] = kernel_size
            res_block_config['dilation_list'] = dilation_rate
            self.blocks.append(ResidualBlock(res_block_config, global_config))

    def forward(self, x):
        return sum(block(x) for block in self.blocks) / self.nbr_blocks

    @staticmethod
    def sample_config():
        return {'res_block': ResidualBlock.sample_config(),
                'kernel_sizes': [3, 5, 7],
                'dilation_rates': [1, 3, 5]}


class HiFiGANDecoder(BaseDecoder):
    """
    HiFiGANDecoder implements a hierarchical decoder with upsampling blocks and Multi-Receptive Field (MRF) blocks, inspired by the HiFiGAN architecture.

    Args:
        pre_net_arch (dict): Architecture configuration for the pre-net, containing 'in_channels', 'out_channels', 'kernel_size', and 'stride'.
        upsample_kernels (list of int): List of kernel sizes for the upsampling blocks.
        upsample_strides (list of int): List of strides for the upsampling blocks.
        block_k_sizes (list of int): List of kernel sizes for the MRF blocks.
        block_rates (list of list of int): List of lists containing dilation rates for each MRF block.
        post_net_arch (dict): Architecture configuration for the post-net, containing 'kernel_size' and 'stride'.
        g_cond_dim (int, optional): Dimension of the global conditioning. Default is 0.
        use_wn (bool, optional): If True, applies weight normalization. Default is False.
        activation (str, optional): Type of activation function to use. Default is 'leaky_relu'.
        dual (bool, optional): If True, uses dual layers. Default is True.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['upsample_kernels', 'upsample_strides', 'postprocess_net', 'mrf_block']
        self.update_keys(mandatory_keys)
        super(HiFiGANDecoder, self).__init__(module_config, global_config)
        base_upsample_config = {}
        base_mrf_block_config = self.module_config['mrf_block']
        zip_loop = zip(self.module_config['upsample_kernels'], self.module_config['upsample_strides'])
        init_out = self.module_config['preprocess_net']['out_channels']

        for i, (up_k_size, up_rate) in enumerate(zip_loop):
            channels = init_out//2**i
            base_upsample_config['input_channels'] = channels
            base_upsample_config['output_channels'] = channels//2
            base_upsample_config['kernel_size'] = up_k_size
            base_upsample_config['stride'] = up_rate
            self.dec_blocks.append(UpsampleBlock(base_upsample_config, global_config))
            conv_block_config = {'input_channels': channels//2,
                                'output_channels': channels//2}
            base_mrf_block_config['res_block'] = {'res_layer': {'conv_block': conv_block_config}}
            self.dec_blocks.append(MRFBlock(base_mrf_block_config, global_config))

        self.post_net = self.norm_f(nn.Conv1d(channels//2, 1,
                                              kernel_size=self.module_config['postprocess_net']['kernel_size'],
                                              stride=self.module_config['postprocess_net']['stride'],
                                              padding=get_padding(self.module_config['postprocess_net']['kernel_size']),
                                              bias=False))

    @staticmethod
    def sample_config():
        return {'upsample_kernels': [10, 8, 8, 4],
                'upsample_strides': [5, 4, 4, 2],
                'preprocess_net': {'in_channels': 192, 'out_channels': 256, 'kernel_size': 7, 'stride': 1},
                'postprocess_net': {'kernel_size': 7, 'stride': 1},
                'mrf_block': MRFBlock.sample_config()}
