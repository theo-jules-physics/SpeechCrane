import torch.nn as nn
import torch
from modules.downsampler import DownsamplingLayer
from modules.discriminators.base import BaseDiscriminator
from utils.modules import TTSModule


class DiscriminatorPBlock(BaseDiscriminator):
    """
    Period-based discriminator block as described in the HiFiGAN paper.

    This discriminator uses downsampling layers to process input audio at different periods,
    allowing it to capture features at various time scales.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the DiscriminatorPBlock.

        Args:
            module_config: Configuration dictionary for the discriminator module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['period']
        self.update_keys(mandatory_keys)
        super(DiscriminatorPBlock, self).__init__(module_config, global_config)
        self.period = self.module_config['period']

    def _set_layers(self, global_config: dict):
        """
        Set up the layers of the discriminator.

        Args:
            global_config: Global configuration dictionary.
        """
        self.pre_net = nn.Identity()
        self.layers = nn.ModuleList()
        nbr_layers = len(self.module_config['channels'])
        padding = (self.module_config['kernel_sizes'] - 1) // 2
        downsampler_config = {'kernel_size': self.module_config['kernel_sizes'],
                              'stride': self.module_config['strides'],
                              'padding': padding,
                              'groups': 1,
                              'layer_type': 'period'}

        # Create downsampling layers
        for i in range(nbr_layers - 1):
            downsampler_config['input_channels'] = self.module_config['channels'][i]
            downsampler_config['output_channels'] = self.module_config['channels'][i + 1]
            self.layers.append(DownsamplingLayer(downsampler_config, global_config))

        # Add final layer with different stride and padding
        downsampler_config['input_channels'] = downsampler_config['output_channels']
        downsampler_config['stride'] = 1
        downsampler_config['padding'] = 2
        self.layers.append(DownsamplingLayer(downsampler_config, global_config))

        # Post-processing network
        self.post_net = self.norm_f(nn.Conv2d(downsampler_config['output_channels'], 1, 3, 1, 1))

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare the input tensor by reshaping it according to the period.

        Args:
            x: Input tensor of shape (batch_size, channels, time).

        Returns:
            Reshaped tensor of shape (batch_size, channels, time // period, period).
        """
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = nn.functional.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t//self.period, self.period)
        return x

    def _process_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the output tensor through the post-processing network.

        Args:
            x: Output tensor from the last convolutional layer.

        Returns:
            Processed tensor with flattened last two dimensions.
        """
        x = self.post_net(x)
        return x.flatten(-2, -1).squeeze(1)

    @staticmethod
    def sample_config() -> dict:
        """
        Provide a sample configuration for the DiscriminatorPBlock.
        """
        return {'period': 2,
                'channels':  [1, 32, 128, 256, 512],
                'kernel_sizes': 5,
                'strides': 3}


class MultiPeriodDiscriminator(TTSModule):
    """
    Multi-Period Discriminator as described in the HiFiGAN paper.

    This discriminator uses multiple period-based discriminators to process the input
    at different time scales, allowing for a more comprehensive analysis of the audio.
    """

    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the MultiPeriodDiscriminator.

        Args:
            module_config: Configuration dictionary for the discriminator module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['periods', 'discriminator_period_block']
        self.update_keys(mandatory_keys)
        super(MultiPeriodDiscriminator, self).__init__(module_config, global_config)
        self.period_discriminators = nn.ModuleList()
        disc_config = self.module_config['discriminator_period_block']
        for period in self.module_config['periods']:
            disc_config['period'] = period
            self.period_discriminators.append(DiscriminatorPBlock(disc_config, global_config))

    def __iter__(self):
        """
        Return an iterator for the period discriminators.

        Returns:
            Iterator over the list of period discriminators.
        """
        return iter(self.period_discriminators)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """
        Forward pass of the Multi-Period Discriminator.

        Args:
            x: Input tensor.

        Returns:
            Tuple containing:
            - List of output tensors from each period discriminator.
            - List of feature maps from each period discriminator.
        """
        outs, fmaps = [], []
        for disc in self.period_discriminators:
            out, fmap = disc(x)
            outs.append(out)
            fmaps.append(fmap)
        return outs, fmaps

    @staticmethod
    def sample_config() -> dict:
        """
        Provide a sample configuration for the MultiPeriodDiscriminator.
        """
        return {'period_list': [2, 3, 5, 7],
                'discriminator_period_block': DiscriminatorPBlock.sample_config()}