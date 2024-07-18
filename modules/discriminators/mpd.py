import torch.nn as nn
from modules.downsampler import DownsamplingLayer
from modules.discriminators.base import BaseDiscriminator
from utils.modules import TTSModule


class DiscriminatorPBlock(BaseDiscriminator):
    """
    DiscriminatorPBlock implements a period-based discriminator using downsampling layers
    as described by the HiFiGAN paper.

    Args:
        module_config (dict): Configuration dictionary for the discriminator module.
        For more details, refer to the BaseDiscriminator class.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['period']
        self.update_keys(mandatory_keys)
        super(DiscriminatorPBlock, self).__init__(module_config, global_config)
        self.period = self.module_config['period']

    def _set_layers(self, global_config):
        self.pre_net = nn.Identity()
        self.layers = nn.ModuleList()
        nbr_layers = len(self.module_config['channels'])
        padding = (self.module_config['kernel_sizes'] - 1) // 2
        downsampler_config = {'kernel_size': self.module_config['kernel_sizes'],
                              'stride': self.module_config['strides'],
                              'padding': padding,
                              'groups': 1,
                              'layer_type': 'period'}
        for i in range(nbr_layers-1):
            downsampler_config['input_channels'] = self.module_config['channels'][i]
            downsampler_config['output_channels'] = self.module_config['channels'][i+1]
            self.layers.append(DownsamplingLayer(downsampler_config, global_config))
        downsampler_config['input_channels'] = downsampler_config['output_channels']
        downsampler_config['stride'] = 1
        downsampler_config['padding'] = 2
        self.layers.append(DownsamplingLayer(downsampler_config, global_config))
        self.post_net = self.norm_f(nn.Conv2d(downsampler_config['output_channels'], 1, 3, 1, 1))

    def _prepare_input(self, x):
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = nn.functional.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t//self.period, self.period)
        return x

    def _process_output(self, x):
        x = self.post_net(x)
        return x.flatten(-2, -1).squeeze(1)

    @staticmethod
    def sample_config():
        return {'period': 2,
                'channels':  [1, 32, 128, 256, 512],
                'kernel_sizes': 5,
                'strides': 3}


class MultiPeriodDiscriminator(TTSModule):
    """
    MultiPeriodDiscriminator implements a discriminator that uses multiple period-based discriminators, as described in the HiFiGAN paper.

    Args:
        module_config (dict): Configuration dictionary for the discriminator module.
            - period_list (list): List of integers representing the periods for each discriminator.
            - discriminator_period_block (dict): Configuration dictionary for the period-based discriminator block.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['periods', 'discriminator_period_block']
        self.update_keys(mandatory_keys)
        super(MultiPeriodDiscriminator, self).__init__(module_config, global_config)
        self.period_discriminators = nn.ModuleList()
        disc_config = self.module_config['discriminator_period_block']
        for period in self.module_config['periods']:
            disc_config['period'] = period
            self.period_discriminators.append(DiscriminatorPBlock(disc_config, global_config))

    def __iter__(self):
        # Return an iterator for the list_discriminators
        return iter(self.period_discriminators)

    def forward(self, x):
        outs, fmaps = [], []
        for disc in self.period_discriminators:
            out, fmap = disc(x)
            outs.append(out)
            fmaps.append(fmap)
        return outs, fmaps

    @staticmethod
    def sample_config():
        return {'period_list': [2, 3, 5, 7],
                'discriminator_period_block': DiscriminatorPBlock.sample_config()}