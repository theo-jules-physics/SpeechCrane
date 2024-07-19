import torch.nn as nn
import torch
from modules.downsampler import DownsamplingLayer
from modules.discriminators.base import BaseDiscriminator
from utils.modules import TTSModule


class DiscriminatorSBlock(BaseDiscriminator):
    """
    Scale-based discriminator block using downsampling layers as described in the MelGAN paper.

    This discriminator processes the input at different scales, allowing it to capture
    features at various resolutions.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the DiscriminatorSBlock.

        Args:
            module_config: Configuration dictionary for the discriminator module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['paddings', 'groups']
        self.update_keys(mandatory_keys)
        super(DiscriminatorSBlock, self).__init__(module_config, global_config)

    def _set_layers(self, global_config: dict):
        """
        Set up the layers of the discriminator.

        Args:
            global_config: Global configuration dictionary.
        """
        self.pre_net = nn.Identity()
        self.layers = nn.ModuleList()
        in_channels = self.module_config['channels'][:-1]
        out_channels = self.module_config['channels'][1:]
        for i in range(len(in_channels)):
            downsampler_config = {'input_channels': in_channels[i],
                                  'output_channels': out_channels[i],
                                  'kernel_size': self.module_config['kernel_sizes'][i],
                                  'stride': self.module_config['strides'][i],
                                  'padding': self.module_config['paddings'][i],
                                  'groups': self.module_config['groups'][i],
                                  'layer_type': 'scale',
                                  'use_spec_norm': self.module_config['use_spec_norm']}
            self.layers.append(DownsamplingLayer(downsampler_config, global_config))
        self.post_net = self.norm_f(nn.Conv1d(out_channels[-1], 1, 3, 1, 1))

    @staticmethod
    def sample_config() -> dict:
        """
        Provide a sample configuration for the DiscriminatorSBlock.
        """
        return {'channels': [1, 16, 64, 256, 1024],
                'kernel_sizes': [15, 41, 41, 41],
                'strides': [1, 4, 4, 4],
                'paddings': [7, 20, 20, 20],
                'groups': [1, 4, 16, 64]}


class MultiScaleDiscriminator(TTSModule):
    """
    Multi-Scale Discriminator as described in the HiFiGAN paper.

    This discriminator uses multiple scale-based discriminators to process the input
    at different resolutions, allowing for a more comprehensive analysis of the audio.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the MultiScaleDiscriminator.

        Args:
            module_config: Configuration dictionary for the discriminator module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['scales', 'discriminator_scale_block']
        self.update_keys(mandatory_keys)
        super(MultiScaleDiscriminator, self).__init__(module_config, global_config)
        self.scale_discriminators = nn.ModuleList([DiscriminatorSBlock(self.module_config['discriminator_scale_block'],
                                                                       global_config)
                                                   for _ in range(self.module_config['scales'])])
        # Use average pooling to downsample the input
        self.meanpool = nn.AvgPool1d(4, 2, 1)

    def __iter__(self):
        """
        Return an iterator for the scale discriminators.

        Returns:
            Iterator over the list of scale discriminators.
        """
        return iter(self.scale_discriminators)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Forward pass of the Multi-Scale Discriminator.

        Args:
            x: Input tensor.

        Returns:
            Tuple containing:
            - List of output tensors from each scale discriminator.
            - List of feature maps from each scale discriminator.
        """
        outs, fmaps = [], []
        for disc in self.scale_discriminators:
            out, fmap = disc(x)
            x = self.meanpool(x)
            outs.append(out)
            fmaps.append(fmap)
        return outs, fmaps

    @staticmethod
    def sample_config() -> dict:
        """
        Provide a sample configuration for the MultiScaleDiscriminator.
        """
        return {'scales': 3, 'discriminator_s_block': DiscriminatorSBlock.sample_config()}

