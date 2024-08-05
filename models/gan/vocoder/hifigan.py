from modules.discriminators.msd import MultiScaleDiscriminator
from modules.discriminators.mpd import MultiPeriodDiscriminator
from modules.decoders.hifigan import HiFiGANDecoder
from models.base_gan import BaseGAN
from utils.preprocessing import DBMelSpectrogram
from utils.general import get_mask
from torch import nn
import torch
import torchaudio.functional as audioF


class HiFiGAN(BaseGAN):
    def __init__(self, model_config, data_config, device):
        super(HiFiGAN, self).__init__(model_config, data_config, device)

    def _init_models(self):
        dec = HiFiGANDecoder(self.arch_config['decoder'], self.arch_config['global'])
        self.list_gen = nn.ModuleDict({'decoder': dec})

        msd = MultiScaleDiscriminator(self.arch_config['msd'], self.arch_config['global'])
        mpd = MultiPeriodDiscriminator(self.arch_config['mpd'], self.arch_config['global'])
        self.list_disc = nn.ModuleDict({'msd': msd, 'mpd': mpd})

        mel_config = self.data_config['mel']
        mel_characs = self.data_config['features_characs']['mel']
        self.mel_processor = DBMelSpectrogram(mel_config['n_fft'], mel_config['win_length'], mel_config['hop_length'],
                                              mel_config['n_mels'], self.data_config['sample_rate'], self.device,
                                              mel_characs['mean'], mel_characs['std'])

        self.mel_loss_function = nn.L1Loss()

    def implement_characs_dataset(self, feature_characs):
        self.mel_processor.mean_mel = feature_characs['mel']['mean']
        self.mel_processor.std_mel = feature_characs['mel']['std']

    def _main_loss(self, gen_output, true_output):
        db_mel_raw, _ = self.mel_processor(true_output)
        db_mel_gen, _ = self.mel_processor(gen_output)
        main_loss_dict = {'mel_loss': self.mel_loss_function(db_mel_raw, db_mel_gen)}
        return main_loss_dict

    def _val_loss(self, gen_output, true_output):
        return self._main_loss(gen_output, true_output)['mel_loss']

    def _preprocess_batch(self, batch):
        preprocessed_dict = {}
        for key, value in batch.items():
            if key == 'waveform':
                preprocessed_dict[key] = audioF.preemphasis(value)
            if key == 'text':
                preprocessed_dict[key] = value
            else:
                preprocessed_dict[key] = value.to(torch.float32)
        preprocessed_dict['true_ouput'] = preprocessed_dict['waveform']
        return preprocessed_dict

    def forward(self, preprocessed_batch):
        output_dict = {}
        output_mask = get_mask(preprocessed_batch['waveform_length']).unsqueeze(1)
        output_dict['gen_out'] = self.list_gen['decoder'](preprocessed_batch['mel'])
        return output_dict

    def inference(self, mel, **kwargs):
        return self.list_gen['decoder'](mel)
