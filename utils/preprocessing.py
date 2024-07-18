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


def check_preprocessing(features: list[str], data_config: dict) -> dict:
    """
    Check and perform preprocessing for specified features.

    Args:
        features: List of features to preprocess.
        data_config: Configuration dictionary for the dataset.

    Returns:
        Updated data configuration with preprocessing information.

    Example:
        >>> features = ['mel', 'waveform']
        >>> data_config = {'path': '/data', 'format': '.wav'}
        >>> updated_config = check_preprocessing(features, data_config)
    """
    dataset_folder = data_config['path']
    format = data_config['format']
    force_preprocessing = data_config.get('force_preprocessing', False)
    preprocessing_folder = os.path.join(dataset_folder, 'preprocessing')
    preprocess_status_file = os.path.join(preprocessing_folder, 'feature_characs.json')
    if os.path.exists(preprocess_status_file):
        feature_preprocessed = json.load(open(preprocess_status_file, 'r')).keys()

    feature_processors = dict()
    for feature in features:
        if feature not in feature_preprocessed or force_preprocessing:
            feature_processors[feature] = FeatureProcessorFactory.get_processor(feature)

    if len(feature_processors) < 0:
        print("All features are already preprocessed.")
    else:
        print('Preprocessing data...')
        preprocessor = DatasetPreprocessor(dataset_folder, feature_processors, format)
        preprocessor.preprocess_data()

    data_config['features_characs'] = {}
    features_characs = json.load(open(os.path.join(preprocessing_folder, 'feature_characs.json'), 'r'))
    for feature in features:
        if feature in features_characs.keys():
            data_config['features_characs'][feature] = features_characs[feature]
    return data_config


class FeatureProcessor(ABC):
    """
    Abstract base class for feature processors.
    """

    @abstractmethod
    def process(self, audio_file: str, text: str | None = None) -> tuple[np.ndarray, int]:
        """
        Process an audio file to extract features.

        Args:
            audio_file: Path to the audio file.
            text: Associated text (optional).

        Returns:
            Tuple containing the extracted feature array and its shape.
        """
        pass


class DatasetPreprocessor:
    """
    Preprocessor for audio datasets, handling feature extraction and storage.
    """
    def __init__(self, dataset_folder: str, feature_processors: dict, format: str):
        """
        Initialize the DatasetPreprocessor.

        Args:
            dataset_folder: Path to the dataset folder.
            feature_processors: Dictionary of feature processors.
            format: Audio file format.
        """
        self.dataset_folder = dataset_folder
        self.feature_processors = feature_processors
        self.preprocessing_folder = os.path.join(dataset_folder, 'preprocessing')
        self.metadata = self._load_metadata(format)

    def _load_metadata(self, format: str) -> pd.DataFrame:
        """
        Load metadata from the csv file in the dataset folder.

        Args:
            format: Audio file format.

        Returns:
            DataFrame containing metadata.

        """
        metadata_file = os.path.join(self.dataset_folder, 'metadata.csv')
        metadata = pd.read_csv(metadata_file, sep='|', header=0, quotechar='\\', quoting=csv.QUOTE_NONE,
                               engine='python')
        metadata['name'] = metadata['file']
        metadata['audio_path'] = metadata.apply(
            lambda row: os.path.join(self.dataset_folder, 'audio', row['speaker_id'], row['name'] + format),
            axis=1
        )
        return metadata

    def process_sample(self, sample: dict) -> dict:
        """
        Process a single sample, extracting specified features.

        Args:
            sample: Dictionary containing sample information.

        Returns:
            Dictionary of processed features for the sample.
        """
        results = {}
        for feature_name, processor in self.feature_processors.items():
            try:
                feature_data, feature_shape = processor.process(sample['audio_path'])
                results[feature_name] = {
                    'data': feature_data,
                    'shape': feature_shape
                }
            except Exception as e:
                print(f"Error processing {feature_name} for {sample['name']}: {str(e)}")
        return results

    def preprocess_data(self) -> None:
        """
        Preprocess the entire dataset, extracting and saving features for all samples.
        """
        feature_shapes = {feature: {} for feature in self.feature_processors.keys()}
        charac_tracker = CharacTracker(self.feature_processors.keys())

        for _, sample in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            features = self.process_sample(sample)

            for feature_name, feature_data in features.items():
                feature_dir = os.path.join(self.preprocessing_folder, feature_name)
                os.makedirs(feature_dir, exist_ok=True)

                file_path = os.path.join(feature_dir, f"{sample['name']}_{feature_name}.npy")
                np.save(file_path, feature_data['data'])
                charac_tracker.update_charac(feature_data['data'], feature_name)
                feature_shapes[feature_name][sample['name']] = feature_data['shape']

        charac_tracker.finalize_characs()
        charac_tracker.save_features_characs(self.preprocessing_folder)
        save_features(self.preprocessing_folder, feature_shapes)
        print("Preprocessing completed.")


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
        return features, features.shape[1]


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
        return waveform, waveform.shape[1]


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
            self.charac_dict[feature]['std'] = np.sqrt(self.charac_dict[feature]['m2'] / self.charac_dict[feature]['count'])
            del self.charac_dict[feature]['m2']
            del self.charac_dict[feature]['count']
        return self.charac_dict

    def save_features_characs(self, preprocessing_folder: str) -> None:
        """
        Save the feature characteristics to a JSON file.

        Args:
            preprocessing_folder: Folder to save the characteristics file.
        """
        feature_characs_file = os.path.join(preprocessing_folder, 'feature_characs.json')
        if os.path.exists(feature_characs_file):
            feature_characs = json.load(open(os.path.join(preprocessing_folder, 'feature_characs.json'), 'r'))
            feature_characs.update(self.charac_dict)
        else:
            feature_characs = self.charac_dict

        for feature, stats in feature_characs.items():
            for key, value in stats.items():
                if isinstance(value, np.generic):
                    feature_characs[feature][key] = value.item()
                if isinstance(value, torch.Tensor):
                    feature_characs[feature][key] = value.item()
        json.dump(feature_characs, open(os.path.join(preprocessing_folder, 'feature_characs.json'), 'w'))


def save_features(preprocessing_folder: str, data_dict: dict) -> None:
    """
    Save the preprocessed features to HDF5 files.

    Args:
        preprocessing_folder: Folder containing the preprocessed features.
        data_dict: Dictionary of preprocessed feature data.
    """
    for feature in data_dict.keys():
        feature_file = os.path.join(preprocessing_folder, feature + '.h5')
        with h5py.File(feature_file, 'w') as f:
            for sample_name, array in data_dict[feature].items():
                f.create_dataset(sample_name, data=array)


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
