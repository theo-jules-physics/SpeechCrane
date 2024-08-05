import os
import pytorch_lightning as pl
import torch
import torchaudio
import numpy as np
import torchaudio.functional as audioF
import logging
import json


class AudioLoggingCallback(pl.Callback):
    """
    A PyTorch Lightning callback for logging audio samples during model training.

    This callback selects random samples from the validation dataset at the start of training,
    saves their raw audio, and then periodically generates and saves audio from the model
    during training. It also saves metadata about the training process.

    Attributes:
        save_epoch_freq (int): Frequency of epochs at which to save generated audio.
        log_folder (str): Path to the folder where audio files and metadata will be saved.
        save_waveform (bool): Whether to save waveform data as .npy files.
        num_samples (int): Number of samples to track and generate audio for.
        sample_indices (List[int]): Indices of the selected samples in the dataset.
        log_samples (List[Dict]): The actual sample data for the selected indices.
        logger (logging.Logger): Logger for recording events and errors.
    """

    def __init__(self, run_dir: str, config: dict):
        """
        Initialize the AudioLoggingCallback.

        Args:
            run_folder (str): Base folder for this training run.
            save_epoch_freq (int): Save generated audio every n epochs.
            save_waveform (bool): If True, save waveform data as .npy files.
            num_samples (int): Number of samples to track and generate.
            log_folder_name (str): Name of the subfolder for audio logs.
        """
        super().__init__()
        self.log_audio_freq = config['log_audio_freq']
        self.log_folder = os.path.join(run_dir, 'logs', config['log_folder_name'])
        os.makedirs(self.log_folder, exist_ok=True)

        self.save_waveform = config['save_waveform']
        self.num_samples = config['num_samples']
        self.sample_indices: list[int] = []
        self.log_samples: list[dict] = []

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when the train begins.

        Selects random samples from the validation set, saves their raw audio,
        and initializes logging for these samples.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The module being trained.
        """
        self.sample_indices = np.random.choice(trainer.datamodule.val_indices, size=self.num_samples, replace=False)
        for idx in self.sample_indices:
            try:
                sample = trainer.datamodule.dataset[idx]
                self.log_samples.append(sample)
                waveform = sample['waveform'].unsqueeze(0)
                self.save_audio(waveform, f'raw_audio_{idx}', pl_module.data_config['sample_rate'])

                name = 'test_processing'
                gen_waveform = self.gen_audio(pl_module, sample)
                self.save_audio(gen_waveform, name, pl_module.data_config['sample_rate'])

            except Exception as e:
                self.logger.error(f"Error processing sample {idx}: {str(e)}")

        self.save_metadata(trainer, pl_module)

    @staticmethod
    def gen_audio(pl_module: pl.LightningModule, sample: dict) -> torch.Tensor:
        """
        Generate audio from a sample using the current state of the model.

        Args:
            pl_module (pl.LightningModule): The module being trained.
            sample (Dict): A dictionary containing the sample data.

        Returns:
            torch.Tensor: The generated audio waveform.
        """
        with torch.no_grad():
            sample_mel = sample['mel'].mT.unsqueeze(0).to(pl_module.device)
            print(sample_mel.shape)
            gen_waveform = pl_module.inference(sample_mel)[0]
        return audioF.deemphasis(gen_waveform)

    def save_audio(self, waveform: torch.Tensor, file_name: str, sample_rate: int) -> None:
        """
        Save an audio waveform to disk.

        Args:
            waveform (torch.Tensor): The audio waveform to save.
            file_name (str): Name of the file to save (without extension).
            sample_rate (int): Sample rate of the audio.
        """
        try:
            torchaudio.save(os.path.join(self.log_folder, f'{file_name}.wav'), waveform.cpu(), sample_rate=sample_rate)
            if self.save_waveform:
                np.save(os.path.join(self.log_folder, f'{file_name}.npy'), waveform.cpu().numpy())
        except Exception as e:
            self.logger.error(f"Error saving audio {file_name}: {str(e)}")

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when the validation ends.

        Generates and saves audio for the tracked samples at specified epoch intervals.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The module being trained.
        """
        current_epoch = trainer.current_epoch
        if trainer.sanity_checking:
            return
        if current_epoch % self.log_audio_freq == 0:
            for idx, sample in zip(self.sample_indices, self.log_samples):
                try:
                    file_name = f'gen_audio_{idx}_epoch_{current_epoch}'
                    gen_waveform = self.gen_audio(pl_module, sample)
                    self.save_audio(gen_waveform, file_name, pl_module.data_config['sample_rate'])
                except Exception as e:
                    self.logger.error(f"Error generating audio for sample {idx} at epoch {current_epoch}: {str(e)}")

    def save_metadata(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Save metadata about the training process.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pl.LightningModule): The module being trained.
        """
        metadata = {
            "model_name": pl_module.__class__.__name__,
            "dataset_name": trainer.datamodule.__class__.__name__,
            "sample_rate": pl_module.data_config['sample_rate'],
            "sample_indices": self.sample_indices.tolist()
        }
        try:
            with open(os.path.join(self.log_folder, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")