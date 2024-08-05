import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio.functional as audioF
from modules.upsampler import MelUpsampler
from modules.diffusion import DiffusionEmbedding
from modules.resnet import WaveNetLayer, GenericResBlock
from models.base_diffusion import BaseDiff
from utils.general import get_mask
from utils.modules import remove_weight_norm_recursively


class ResDiffWave(GenericResBlock):
    def __init__(self, module_config, global_config):
        super(ResDiffWave, self).__init__(module_config, global_config, WaveNetLayer)
        hidden_channels = self.module_config['layer_config']['hidden_channels']
        self.skip_projection = nn.Conv1d(hidden_channels, hidden_channels, 1)

    def output_postprocess(self, x):
        x = self.skip_projection(x)
        x = F.relu(x)
        return x


class DiffWave(BaseDiff):
    def __init__(self, model_config, data_config, device):
        super(DiffWave, self).__init__(model_config, data_config, device)

    def _init_model(self):
        global_config = self.arch_config['global']
        global_config['time_emb_dim'] = self.arch_config['time_embedding']['output_dimension']
        self.upsampler = MelUpsampler(self.arch_config['mel_upsampler'], global_config)
        self.diff_emb = DiffusionEmbedding(self.arch_config['time_embedding'], global_config)
        self.res_model = ResDiffWave(self.arch_config['res_model'], global_config)
        self.input_projection = nn.Conv1d(1, self.arch_config['res_model']['layer_config']['hidden_channels'], 1)
        self.output_projection = nn.Conv1d(self.arch_config['res_model']['output_channels'], 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x, diffusion_step, x_mask, conditions):
        if x_mask is None:
            x_mask = torch.ones_like(x).bool()
        x = self.input_projection(x)
        loc_cond = conditions['loc_cond']
        g_cond = conditions['g_cond']
        time_emb = self.diff_emb(diffusion_step)
        x = self.res_model(x, x_mask, g_cond=g_cond, loc_cond=loc_cond, time_emb=time_emb)
        x = self.output_projection(x) * x_mask
        return x

    def _preprocess_batch(self, batch):
        pre_emph_wave = audioF.preemphasis(batch['waveform'])
        preprocessed_dict = {'input': pre_emph_wave.to(self.device)}
        if self.training_config['masking']:
            preprocessed_dict['mask'] = get_mask(batch['waveform_length']).to(self.device).unsqueeze(1)
        else:
            preprocessed_dict['mask'] = torch.ones_like(preprocessed_dict['input']).bool()
        if 'mel' in batch:
            loc_cond = self.upsampler(batch['mel'])
        else:
            loc_cond = None
        preprocessed_dict['conditions'] = {'loc_cond': loc_cond, 'g_cond': None}
        return preprocessed_dict

    def remove_weight_norm(self):
        remove_weight_norm_recursively(self)


if __name__ == '__main__':
    None