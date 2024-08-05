import json
import importlib
from utils.general import import_config
from pathlib import Path
import os
import torch
import torchaudio
import numpy as np
from argparse import ArgumentParser


class VocoderInference:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.config = import_config(self.model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()

    def load_model(self, ckpt_name='last'):
        ckpt_file = self.model_path / 'files' / f'{ckpt_name}.ckpt'
        name_net = self.config['model_config']['name']
        print(f'Loading {name_net} from {ckpt_file}')
        module_path, features = json.load(open(f'configs/model_list.json', 'r'))[name_net]
        model_module = importlib.import_module(module_path)
        model_class = getattr(model_module, name_net)
        model = model_class.load_from_checkpoint(ckpt_file).to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def inference(self, mel):
        preemph_wave = self.model.inference(mel)[0].cpu()
        final_wave = torchaudio.functional.deemphasis(preemph_wave)
        return final_wave

    def generate_audio(self, mel_path):
        print(f'Generating audio from mel spectrogram: {mel_path}')
        mel = torch.tensor(np.load(str(mel_path))).unsqueeze(0).to(torch.float32).to(self.device)
        output = self.inference(mel)
        gen_folder = 'gen_audio'
        os.makedirs(gen_folder, exist_ok=True)
        output_path = os.path.join(gen_folder, 'output.wav')
        torchaudio.save(output_path, output, self.config['data_config']['sample_rate'])
        print(f'Audio saved to {output_path}')


def run():
    vocoder = VocoderInference(args.model_path)
    vocoder.generate_audio(args.mel_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mel_path', type=str, required=True)
    args = parser.parse_args()
    run()
