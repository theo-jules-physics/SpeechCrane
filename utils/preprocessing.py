import numpy as np
import h5py
import os
import torch
import csv
import json
from tqdm import tqdm
import pandas as pd
import torchaudio.transforms as audioT
import torchaudio.functional as audioF
import torchaudio
from abc import ABC, abstractmethod
from faster_whisper.feature_extractor import FeatureExtractor
import multiprocessing
from functools import partial


class CharacTracker:
    """
    Tracker for calculating and storing feature characteristics.
    """

    def __init__(self, feature_list: list[str]):
        """
        Initialize the CharacTracker.

        Args:
            feature_list: List of features to track.
        """
        self.feature_list = feature_list
        self.charac_dict = {feature: {'min': np.inf, 'max': -np.inf, 'mean': 0.0, 'count': 0, 'm2': 0.0}
                            for feature in feature_list}

    def update_charac(self, data_array: np.ndarray, feature: str) -> None:
        """
        Update characteristics for a given feature based on new data.

        Args:
            data_array: New data array for the feature.
            feature: Name of the feature being updated.
        """
        data_array = np.array(data_array).astype(np.float64)
        self.charac_dict[feature]['min'] = min(self.charac_dict[feature]['min'], data_array.min())
        self.charac_dict[feature]['max'] = max(self.charac_dict[feature]['max'], data_array.max())

        n = len(data_array.flatten())
        new_mean = data_array.mean()
        new_m2 = np.sum((data_array - new_mean) ** 2)

        total_count = self.charac_dict[feature]['count'] + n
        delta = new_mean - self.charac_dict[feature]['mean']
        new_mean = (self.charac_dict[feature]['mean'] * self.charac_dict[feature]['count'] + new_mean * n) / total_count
        self.charac_dict[feature]['m2'] += new_m2 + delta ** 2 * self.charac_dict[feature]['count'] * n / total_count
        self.charac_dict[feature]['mean'] = new_mean
        self.charac_dict[feature]['count'] = total_count

    def finalize_characs(self) -> dict:
        """
        Finalize the characteristics calculations.

        Returns:
            Dictionary of finalized characteristics for each feature.
        """
        for feature in self.charac_dict.keys():
            self.charac_dict[feature]['std'] = np.sqrt(
                self.charac_dict[feature]['m2'] / self.charac_dict[feature]['count'])
            del self.charac_dict[feature]['m2']
            del self.charac_dict[feature]['count']
        return self.charac_dict


class MainIndexManager:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        preprocessing_folder = os.path.join(dataset_folder, 'preprocessing')
        os.makedirs(preprocessing_folder, exist_ok=True)
        self.index_file = os.path.join(preprocessing_folder, 'main_index.h5')
        self.metadata = None
        self.feature_lengths = {}
        self.feature_characs = {}
        self.create_or_load_index(os.path.join(dataset_folder, 'metadata.csv'))

    def create_or_load_index(self, metadata_path):
        if not os.path.exists(self.index_file):
            self._create_index(metadata_path)
        else:
            self._load_index()

    def _create_index(self, metadata_path):
        metadata = pd.read_csv(metadata_path, sep='|', header=0, quotechar='\\', quoting=csv.QUOTE_NONE, engine='python')
        with h5py.File(self.index_file, 'w') as hf:
            hf.create_dataset('file_names', data=np.array(metadata['file_name'], dtype=h5py.string_dtype()))
            hf.create_dataset('transcriptions', data=np.array(metadata['text'], dtype=h5py.string_dtype()))
            hf.create_dataset('speaker_ids', data=np.array(metadata['speaker_id'], dtype=h5py.string_dtype()))

            # Create a dataset to track available preprocessed data
            hf.create_group('feature_lengths')
            hf.create_group('feature_characs')

        self.metadata = metadata

    def _load_index(self):
        with h5py.File(self.index_file, 'r') as hf:
            self.metadata = {
                'name': [name.decode('utf-8') for name in hf['file_names'][:]],
                'text': [text.decode('utf-8') for text in hf['transcriptions'][:]],
                'speaker_id': [speaker_id.decode('utf-8') for speaker_id in hf['speaker_ids'][:]]
            }
            if 'feature_lengths' in hf:
                for feature in hf['feature_lengths']:
                    self.feature_lengths[feature] = hf[f'feature_lengths/{feature}'][:]
            if 'feature_characs' in hf:
                for feature in hf['feature_characs']:
                    self.feature_characs[feature] = {
                        k: v[()] for k, v in hf[f'feature_characs/{feature}'].items()
                    }

    def update_feature_length(self, index, feature, length):
        with h5py.File(self.index_file, 'r+') as hf:
            if feature not in hf['feature_lengths']:
                hf['feature_lengths'].create_dataset(feature, (len(self),), dtype=np.int32)
            hf[f'feature_lengths/{feature}'][index] = length

        if feature not in self.feature_lengths:
            self.feature_lengths[feature] = np.zeros(len(self), dtype=np.int32)
        self.feature_lengths[feature][index] = length

    def update_feature_characs(self, feature, characs):
        with h5py.File(self.index_file, 'r+') as hf:
            if feature not in hf['feature_characs']:
                hf['feature_characs'].create_group(feature)
            for k, v in characs.items():
                if k in hf[f'feature_characs/{feature}']:
                    del hf[f'feature_characs/{feature}/{k}']
                hf[f'feature_characs/{feature}'].create_dataset(k, data=v)
        self.feature_characs[feature] = characs

    def get_data_info(self, index):
        return {
            'file_name': self.metadata['name'][index],
            'transcription': self.metadata['text'][index],
            'speaker_id': self.metadata['speaker_id'][index],
            'feature_lengths': {
                feature: length[index] for feature, length in self.feature_lengths.items() if length[index] != -1
            }
        }

    def is_feature_available(self, index, feature):
        return feature in self.feature_lengths and self.feature_lengths[feature][index] != -1

    def get_available_features(self):
        return list(self.feature_lengths.keys())

    def get_feature_characs(self, feature):
        return self.feature_characs.get(feature, None)

    def __len__(self):
        return len(self.metadata['name'])


class FeatureProcessor(ABC):
    """
    Abstract base class for feature processors.
    """

    @abstractmethod
    def process(self, audio_file: str, text: str | None = None) -> np.ndarray:
        """
        Process an audio file to extract features.

        Args:
            audio_file: Path to the audio file.
            text: Associated text (optional).

        Returns:
            Feature array.
        """
        pass


class FeatureProcessorFactory:
    """
    Factory class for creating feature processors.
    """
    @staticmethod
    def get_processor(processor_name: str) -> FeatureProcessor:
        """
        Get a feature processor instance based on the processor name.

        Args:
            processor_name: Name of the processor to create.

        Returns:
            An instance of the specified feature processor.

        Raises:
            ValueError: If an unknown processor name is provided.
        """
        if processor_name == 'mel':
            return WhisperFeaturePreprocessor()
        elif processor_name == 'waveform':
            return WaveformPreprocessor()
        else:
            raise ValueError(f"Unknown feature processor: {processor_name}")


class WhisperFeaturePreprocessor(FeatureExtractor):
    """
    Feature preprocessor for Whisper features.
    """
    def process(self, audio_file: str, text: str | None = None) -> tuple[np.ndarray, int]:
        """
        Process an audio file to extract features, by default the Mel spectrogram. cf feature_extractor.py
        Args:
            audio_file: Path to the audio file.
            text: Associated text (optional).

        Returns:
            Tuple containing the extracted feature array and its shape.
        """
        audio, sr = torchaudio.load(audio_file)
        self.sampling_rate = sr
        features = self(audio[0], padding=False)
        return features


class WaveformPreprocessor:
    """
    Preprocessor for raw audio waveform data.
    """
    @staticmethod
    def process(audio_file: str, text: str | None = None) -> tuple[np.ndarray, int]:
        """
        Process an audio file to extract the raw waveform data.
        Notably, normalizes the waveform by its absolute maximum value.
        Args:
            audio_file: Path to the audio file.
            text: Associated text (optional).

        Returns:
            Tuple containing the extracted waveform array and its shape.
        """
        audio, sr = torchaudio.load(audio_file)
        waveform = audio[0]
        waveform = waveform.unsqueeze(0)
        waveform = waveform / waveform.abs().max()
        return waveform


class DatasetPreprocessor:
    """
    Preprocessor for audio datasets, handling feature extraction and storage.
    """
    def __init__(self, dataset_folder: str, feature_processors: dict, audio_format: str):
        """
        Initialize the DatasetPreprocessor.

        Args:
            dataset_folder: Path to the dataset folder.
            feature_processors: Dictionary of feature processors.
            format: Audio file format.
        """
        self.dataset_folder = dataset_folder
        self.index_manager = MainIndexManager(dataset_folder)
        self.feature_processors = feature_processors
        self.audio_format = audio_format
        self.charac_tracker = CharacTracker(feature_processors.keys())

    @staticmethod
    def _process_sample(args):
        idx, audio_path, processor = args
        try:
            feature_data = processor.process(audio_path)
            return idx, feature_data
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return idx, None

    def process_sample(self, feature_name, idx):
        if not self.index_manager.is_feature_available(idx, feature_name):
            data_info = self.index_manager.get_data_info(idx)
            audio_path = os.path.join(self.dataset_folder, 'audio', f"{data_info['file_name']}.{self.audio_format}")
            try:
                feature_data = self.feature_processors[feature_name].process(audio_path)
                return idx, feature_data
            except Exception as e:
                print(f"Error processing {feature_name} for {data_info['file_name']}: {str(e)}")
                return idx, None
        return idx, None

    def process_feature(self, feature_name, parallel=False, num_workers=None):

        num_samples = len(self.index_manager)
        feature_file = os.path.join(self.dataset_folder, 'preprocessing', f'{feature_name}.h5')
        processor = self.feature_processors[feature_name]

        if parallel:
            print(f'Processing {feature_name} in parallel...')
            if num_workers is None:
                num_workers = multiprocessing.cpu_count()

            with multiprocessing.Pool(num_workers) as pool:
                args_list = []
                for idx in range(num_samples):
                    if not self.index_manager.is_feature_available(idx, feature_name):
                        data_info = self.index_manager.get_data_info(idx)
                        audio_path = os.path.join(self.dataset_folder, 'audio',
                                                  f"{data_info['file_name']}.{self.audio_format}")
                        args_list.append((idx, audio_path, processor))

                results = list(tqdm(pool.imap(self._process_sample, args_list),
                                    total=len(args_list), desc=f"Processing {feature_name}"))
            print('Successfully processed in parallel')

        else:
            print(f'Processing {feature_name} sequentially...')
            results = []
            for idx in tqdm(range(num_samples), desc=f"Processing {feature_name}"):
                if not self.index_manager.is_feature_available(idx, feature_name):
                    data_info = self.index_manager.get_data_info(idx)
                    audio_path = os.path.join(self.dataset_folder, 'audio',
                                              f"{data_info['file_name']}.{self.audio_format}")
                    results.append(self._process_sample((idx, audio_path, processor)))
            print('Successfully processed sequentially')

        with h5py.File(feature_file, 'w') as hf:
            feature_group = hf.create_group(feature_name)
            dt = h5py.special_dtype(vlen=np.dtype('float32'))
            channels = next(result[1].shape[0] for result in results if result[1] is not None)
            dataset = feature_group.create_dataset('data', (num_samples, channels), dtype=dt)
            for idx, feature_data in results:
                if feature_data is not None:
                    dataset[idx] = feature_data
                    self.index_manager.update_feature_length(idx, feature_name, feature_data.shape[1])
                    self.charac_tracker.update_charac(feature_data, feature_name)

    def preprocess_data(self, parallel_features=None, num_workers=None) -> None:
        if parallel_features is None:
            parallel_features = []

        for feature_name in self.feature_processors.keys():
            parallel = feature_name in parallel_features
            self.process_feature(feature_name, parallel, num_workers)

        for feature, characs in self.charac_tracker.finalize_characs().items():
            self.index_manager.update_feature_characs(feature, characs)

        print("Preprocessing completed.")


def preprocess_data(data_config: dict) -> dict:

    dataset_folder = data_config['path']
    format = data_config['format']
    force_preprocessing = data_config.get('force_preprocessing', False)
    index_manager = MainIndexManager(dataset_folder)

    feature_processors = dict()
    for feature in data_config['features']:
        if feature not in index_manager.get_available_features() or force_preprocessing:
            feature_processors[feature] = FeatureProcessorFactory.get_processor(feature)

    if len(feature_processors) == 0:
        print("All features are already preprocessed.")
    else:
        print('Features to preprocess:', list(feature_processors.keys()))
        print('Preprocessing data...')
        preprocessor = DatasetPreprocessor(dataset_folder, feature_processors, format)
        preprocessor.preprocess_data(parallel_features=data_config.get('parallel_features', None),
                                     num_workers=data_config.get('num_workers', None))

    data_config['features_characs'] = {}
    for feature in data_config['features']:
        characs = index_manager.get_feature_characs(feature)
        if characs:
            data_config['features_characs'][feature] = characs
    return data_config


class DBSpectrogram:
    """
    Computes and normalizes the spectrogram in decibel scale.
    """

    def __init__(self, n_fft: int, win_length: int, hop_length: int, device: torch.device,
                 mean_spec: float = 0, std_spec: float = 1):
        """
        Initialize the DBSpectrogram.

        Args:
            n_fft: Number of FFT bins.
            win_length: Window length for STFT.
            hop_length: Hop length for STFT.
            device: Torch device to use.
            mean_spec: Mean for spectrogram normalization.
            std_spec: Standard deviation for spectrogram normalization.
        """
        self.mean_spec = mean_spec
        self.std_spec = std_spec
        self.spectro_t = audioT.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length).to(device)
        self.device = device

    @staticmethod
    def amplitude_to_db(spec: torch.Tensor) -> torch.Tensor:
        """
        Convert amplitude spectrogram to decibel scale.

        Args:
            spec: Input amplitude spectrogram.

        Returns:
            Spectrogram in decibel scale.
        """
        return audioF.amplitude_to_DB(spec, multiplier=10.0, amin=1e-6, db_multiplier=0.)

    def normalize_mel(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Normalize the spectrogram.

        Args:
            spec: Input spectrogram.

        Returns:
            Normalized spectrogram.
        """
        return (spec - self.mean_spec) / self.std_spec

    def __call__(self, batch_signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the spectrogram for a batch of signals.

        Args:
            batch_signals: Batch of input signals.

        Returns:
            Tuple of (normalized DB spectrogram, log energy).
        """
        batch_spec = self.spectro_t(batch_signals)
        db_spec = self.amplitude_to_db(batch_spec)
        db_spec = self.normalize_mel(db_spec).squeeze(1)
        energy = torch.norm(batch_spec, dim=1)
        log_energy = torch.log(energy + 1e-5)
        return db_spec.to(self.device), log_energy.to(self.device)


class DBMelSpectrogram:
    """
    Computes and normalizes the Mel spectrogram in decibel scale.
    """
    def __init__(self, n_fft: int, win_length: int, hop_length: int, n_mels: int,
                 sample_rate: int, device: torch.device, mean_mel: torch.Tensor | None = None,
                 std_mel: torch.Tensor | None = None):
        """
        Initialize the DBMelSpectrogram.

        Args:
            n_fft: Number of FFT bins.
            win_length: Window length for STFT.
            hop_length: Hop length for STFT.
            n_mels: Number of Mel filter banks.
            sample_rate: Audio sample rate.
            device: Torch device to use.
            mean_mel: Mean for Mel spectrogram normalization.
            std_mel: Standard deviation for Mel spectrogram normalization.
        """
        self.mean_mel = mean_mel
        self.std_mel = std_mel
        self.mel_spectro = audioT.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
                                                 hop_length=hop_length, n_mels=n_mels).to(device)
        self.device = device

    @staticmethod
    def amplitude_to_db(mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Convert amplitude Mel spectrogram to decibel scale.

        Args:
            mel_spec: Input amplitude Mel spectrogram.

        Returns:
            Mel spectrogram in decibel scale.
        """
        return audioF.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-6, db_multiplier=0.)

    def normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Normalize the Mel spectrogram.

        Args:
            mel: Input Mel spectrogram.

        Returns:
            Normalized Mel spectrogram.
        """
        if self.mean_mel is None or self.std_mel is None:
            return mel
        else:
            return (mel - self.mean_mel) / self.std_mel

    def __call__(self, batch_signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Mel spectrogram for a batch of signals.

        Args:
            batch_signals: Batch of input signals.

        Returns:
            Tuple of (normalized DB Mel spectrogram, log energy).
        """
        batch_mels = self.mel_spectro(batch_signals)
        db_mels = self.amplitude_to_db(batch_mels)
        db_mels = self.normalize_mel(db_mels).squeeze(1)
        energy = torch.norm(batch_mels, dim=1)
        log_energy = torch.log(energy + 1e-5)
        return db_mels.to(self.device), log_energy.to(self.device)


if __name__ == '__main__':
    None
