import os
import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import json
import csv
import pandas as pd
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Iterator


def correct_waveform(batch_waveform: torch.Tensor, max_len_waveform: int | None = None) -> torch.Tensor:
    """
    Corrects the waveform to ensure uniform length by either truncating or padding.

    Args:
        batch_waveform: Batch of waveforms of shape (batch_size, time).
        max_len_waveform: Maximum length for the waveforms. If None,
                          no length correction is applied.

    Returns:
        Corrected waveform with an added channel dimension,
        shape (batch_size, 1, time).

    Example:
        >>> waveform = torch.randn(32, 16000)  # 32 samples of 1-second audio at 16kHz
        >>> corrected = correct_waveform(waveform, 20000)
        >>> print(corrected.shape)
        torch.Size([32, 1, 20000])
    """
    if max_len_waveform is None:
        return batch_waveform.unsqueeze(1)

    len_wave = batch_waveform.size(1)
    if len_wave > max_len_waveform:
        batch_waveform = batch_waveform[:, :max_len_waveform]
    else:
        batch_waveform = F.pad(batch_waveform, (0, max_len_waveform - len_wave))
    return batch_waveform.unsqueeze(1)


class BucketSampler(torch.utils.data.Sampler):
    """
    Organizes dataset samples into buckets based on their lengths,
    creating batches with similar-length samples for efficient training.

    Supports shuffling and ensures that all batches are filled to
    the specified batch size, improving training efficiency by reducing
    padding and optimizing memory usage.
    """

    def __init__(self, dataset: Dataset, batch_size: int, divide_number: int = 5,
                 subsample_idx: list[int] | None = None, shuffle: bool = True):
        """
        Initialize the BucketSampler.

        Args:
            dataset: Dataset to sample from.
            batch_size: Size of each batch.
            divide_number: Number of divisions for creating bucket boundaries.
            subsample_idx: Indices to subsample from the dataset.
            shuffle: Whether to shuffle the data each epoch.
        """
        super().__init__(dataset)
        self.shuffle = shuffle
        self.epoch = 0
        self.length_val = np.asarray(dataset.bucket_length)
        max_val, min_val = np.max(self.length_val), np.min(self.length_val)
        self.boundaries = np.asarray([min_val + (max_val - min_val) * i / divide_number for i in range(1, divide_number)])
        self.subsample_idx = subsample_idx
        self.batch_size = batch_size
        self.buckets = self._create_buckets()
        self.num_samples_per_bucket = self._get_num_samples_buckets()
        self.num_samples = sum(self.num_samples_per_bucket)

    def _create_buckets(self) -> list[list[int]]:
        """
        Create buckets of indices based on the lengths of the samples.

        Returns:
            List of buckets, where each bucket is a list of sample indices.
        """
        buckets = [[] for _ in range(len(self.boundaries)+1)]
        for i, length in enumerate(self.length_val):
            if self.subsample_idx is not None and i not in self.subsample_idx:
                continue
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
        return buckets

    def _get_num_samples_buckets(self) -> list[int]:
        """
        Calculate the number of samples in each bucket, ensuring all buckets are full.

        Returns:
            List of the number of samples per bucket.
        """
        num_samples_per_bucket = []
        for i in range(len(self.buckets)):
            len_bucket = len(self.buckets[i])
            total_batch_size = self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return num_samples_per_bucket

    def _bisect(self, length: int, lo: int = 0, hi: int | None = None) -> int:
        """
        Find the appropriate bucket index for a given sample length using binary search.

        Args:
            length: Length of the sample.
            lo: Lower bound for the search.
            hi: Upper bound for the search.

        Returns:
            Index of the bucket.
        """
        if hi is None:
            hi = len(self.boundaries)-1
        if length < self.boundaries[0]:
            return 0

        if length >= self.boundaries[-1]:
            return len(self.boundaries)

        while hi > lo:
            mid = (lo + hi) // 2
            if self.boundaries[mid] <= length < self.boundaries[mid + 1]:
                return mid + 1
            elif self.boundaries[mid] > length:
                hi = mid
            else:
                lo = mid + 1

        return -1

    def __iter__(self) -> Iterator[list[int]]:
        """
        Generate batches of indices for each epoch.

        Returns:
            Iterator over batches of sample indices.
        """
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i, bucket in enumerate(self.buckets):
            len_bucket = len(bucket)
            idx_bucket = indices[i]
            num_samples = self.num_samples_per_bucket[i]
            rem = num_samples - len_bucket
            idx_bucket = idx_bucket + idx_bucket * (rem // len_bucket) + idx_bucket[:(rem % len_bucket)]

            for j in range(len(idx_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in idx_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches
        self.epoch += 1

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def __len__(self) -> int:
        """
        Get the total number of batches in the sampler.

        Returns:
            Number of batches.
        """
        return self.num_samples // self.batch_size


class RawDataset(Dataset):
    """
    Handles the loading and preprocessing of dataset samples, including features
    stored in H5 and NPY formats. Supports preloading data into memory, normalization,
    and bucketing samples based on feature lengths.
    """
    def __init__(self, data_dir: str, h5_features: list[str], npy_features: list[str],
                 preloading: bool = False, preprocessed_text: bool = True,
                 bucket_feature: str | None = None):
        """
        Initialize the RawDataset.

        Args:
            data_dir: Directory containing the data.
            h5_features: List of features stored in H5 format.
            npy_features: List of features stored in NPY format.
            preloading: Whether to preload data into memory.
            preprocessed_text: Whether the text data is preprocessed.
            bucket_feature: Feature used for bucketing samples by length.
        """
        self.h5_features = h5_features
        self.npy_features = npy_features
        self.preprocessing_folder = os.path.join(data_dir, 'preprocessing')
        self.preloading = preloading
        self.preprocessed_text = preprocessed_text
        self._load_df_dataset(data_dir)
        self.bucket_length = self._load_bucking_length(bucket_feature)
        self._get_characs()
        self._load_data_into_memory()

    def _load_df_dataset(self, data_dir: str) -> None:
        """
        Load the dataset metadata from a CSV file.

        Args:
            data_dir: Directory containing the metadata file.
        """
        self.df_dataset = pd.read_csv(os.path.join(data_dir, 'metadata.csv'), sep='|', header=0,
                                      quotechar='\\', quoting=csv.QUOTE_NONE)

    def _load_bucking_length(self, bucket_feature: str | None) -> list[int] | None:
        """
        Load the lengths of features for bucketing from an H5 file.

        Args:
            bucket_feature: Feature used for bucketing.

        Returns:
            List of lengths of the samples, or None if bucket_feature is None.
        """
        if bucket_feature is None:
            return None
        feature_length = f'{bucket_feature}_length'
        data_file = os.path.join(self.preprocessing_folder, feature_length + '.h5')
        length_dict = {}
        with h5py.File(data_file, 'r') as f:
            for sample_name in f.keys():
                length_dict[sample_name] = f[sample_name][()]
        length_val = list(length_dict.values())
        return length_val

    def _normalize(self, data: torch.Tensor, feature_name: str) -> torch.Tensor:
        """
        Normalize the data based on precomputed mean and standard deviation.

        Args:
            data: Data to be normalized.
            feature_name: Name of the feature.

        Returns:
            Normalized data.
        """
        mean, std = self.feature_characs[feature_name]['mean'], self.feature_characs[feature_name]['std']
        return (data - mean) / std

    def _get_characs(self) -> None:
        """
        Load and normalize feature characteristics (mean, std, min, max) from a JSON file.
        """
        self.feature_characs = json.load(open(os.path.join(self.preprocessing_folder, 'feature_characs.json'), 'r'))
        for feature, stats in self.feature_characs.items():
            mean, std = stats['mean'], stats['std']
            stats['min'] = (stats['min'] - mean) / std
            stats['max'] = (stats['max'] - mean) / std

    def _load_npy_feature(self, idx: int, feature_name: str) -> tuple[torch.Tensor, int]:
        """
        Load a specific feature from NPY files.

        Args:
            idx: Index of the sample.
            feature_name: Name of the feature.

        Returns:
            Loaded feature tensor and its length.
        """
        sample_name = self.df_dataset.iloc[idx]['file']
        speaker_id = self.df_dataset.iloc[idx]['speaker_id']
        data_file = os.path.join(self.preprocessing_folder, feature_name, speaker_id,
                                 sample_name + f'_{feature_name}.npy')
        tensor_data = torch.tensor(np.load(data_file))
        if feature_name == 'waveform':
            tensor_data = tensor_data[0]
            return tensor_data, tensor_data.shape[0]
        elif feature_name == 'distilbert':
            tensor_data = tensor_data.mT[0]
            return tensor_data, tensor_data.shape[0]
        elif feature_name == 'weo':
            tensor_data = tensor_data.squeeze(0)
            return tensor_data, tensor_data.shape[0]
        elif feature_name == 'spk_emb':
            tensor_data = self._normalize(tensor_data, feature_name)
            tensor_data = tensor_data.unsqueeze(0)
            return tensor_data, 1
        else:
            tensor_data = tensor_data.mT
            tensor_data = self._normalize(tensor_data, feature_name)
            return tensor_data, tensor_data.shape[0]

    def preload_h5_feature(self, feature: str) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        """
        Preload H5 feature data into memory.

        Args:
            feature: Feature to preload.

        Returns:
            Tuple of dictionaries containing preloaded feature data and their lengths.
        """
        print('Loading {}...'.format(feature))
        data_file = os.path.join(self.preprocessing_folder, feature + '.h5')
        data_feature, length_feature = {}, {}
        with h5py.File(data_file, 'r') as f:
            for sample_name in tqdm(f.keys()):
                tensor_data = torch.tensor(f[sample_name][()])
                if feature in self.feature_characs.keys():
                    tensor_data = self._normalize(tensor_data, feature)
                data_feature[sample_name] = tensor_data
                length_feature[sample_name] = len(tensor_data)
        print('{} loaded.'.format(feature))
        return data_feature, length_feature

    def preload_npy_feature(self, feature: str) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        """
        Preload NPY feature data into memory.

        Args:
            feature: Feature to preload.

        Returns:
            Tuple of dictionaries containing preloaded feature data and their lengths.
        """
        print('Loading {}...'.format(feature))
        data_feature, length_feature = {}, {}
        for idx in tqdm(range(len(self.df_dataset))):
            sample_name = self.df_dataset.iloc[idx]['file']
            tensor_data, data_length = self._load_npy_feature(idx, feature)
            data_feature[sample_name] = tensor_data
            length_feature[sample_name] = data_length
        print('{} loaded.'.format(feature))
        return data_feature, length_feature

    def _load_data_into_memory(self) -> None:
        """
        Load all specified features into memory.
        """
        print('Loading data into memory...')
        self.data = {}
        for feature in self.h5_features:
            self.data[feature], self.data[f'{feature}_length'] = self.preload_h5_feature(feature)
        if self.preloading:
            for feature in self.npy_features:
                self.data[feature], self.data[f'{feature}_length'] = self.preload_npy_feature(feature)
        print('Data loaded.')

    def _get_feature(self, idx: int, feature_name: str) -> tuple[torch.Tensor, int]:
        """
        Get the feature data for a specific sample.

        Args:
            idx: Index of the sample.
            feature_name: Name of the feature.

        Returns:
            Feature data tensor and its length.
        """
        if feature_name in self.data.keys():
            sample_name = self.df_dataset.iloc[idx]['file']
            sample_data = self.data[feature_name][sample_name]
            sample_data_length = self.data[f'{feature_name}_length'][sample_name]
            return sample_data, sample_data_length
        else:
            return self._load_npy_feature(idx, feature_name)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """
        Get a sample and its features.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary of features and their lengths.
        """
        dict_out = {}
        for feature in self.h5_features + self.npy_features:
            dict_out[feature], dict_out[f'{feature}_length'] = self._get_feature(idx, feature)
        if not self.preprocessed_text:
            dict_out['text'] = self.df_dataset.iloc[idx]['text']
        return dict_out

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return len(self.df_dataset)


class TTSDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule that manages the setup of training and validation
    DataLoaders for a text-to-speech (TTS) dataset. Handles data partitioning,
    feature collation, and provides interfaces to the training and validation data.
    """
    def __init__(self, data_config: dict, h5_features: list[str], npy_features: list[str],
                 training_path: str | None = None):
        """
        Initialize the TTSDataModule.

        Args:
            data_config: Configuration for the data.
            h5_features: List of H5 features.
            npy_features: List of NPY features.
            training_path: Path for training data partition.
        """
        super().__init__()
        self.data_config = data_config
        self.mel_config = data_config['mel']
        self.spec_config = data_config['spec']
        self.h5_features = h5_features
        self.npy_features = npy_features
        self.dataset = RawDataset(self.data_config['path'], h5_features, npy_features, data_config['preload_data'],
                                  data_config['preprocessed_text'], self.data_config['bucket_feature'])
        self.feature_characs = self.dataset.feature_characs
        self._partition_dataset(training_path)

    def _collate_batch(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate a batch of samples into padded tensors.

        Args:
            batch: List of samples.

        Returns:
            Dictionary of padded feature tensors and lengths.
        """
        batch_flist = list(batch[0].keys())
        batch_dict = {feature: [] for feature in batch_flist}
        if 'text' in batch_dict:
            batch_dict['text'] = [sample['text'] for sample in batch]
        feature_list = self.h5_features + self.npy_features
        max_len_feat = None

        for sample in batch:
            for feature in feature_list:
                batch_dict[feature].append(sample[feature])
                batch_dict[f'{feature}_length'].append(sample[f'{feature}_length'])

        for feature in feature_list:
            batch_dict[feature] = pad_sequence(batch_dict[feature], batch_first=True)
            batch_dict[f'{feature}_length'] = torch.tensor(batch_dict[f'{feature}_length'], dtype=torch.long)

        for feat in ['spec', 'mel']:
            if feat in batch_dict:
                batch_dict[feat] = batch_dict[feat].mT
                max_len_feat = batch_dict[feat].size(-1)
                max_len_wave = max_len_feat * self.data_config[feat]['hop_length']

        if 'weo' in batch_dict:
            batch_dict['weo'] = batch_dict['weo'].mT

        if 'waveform' in batch_dict:
            batch_dict['waveform'] = correct_waveform(batch_dict['waveform'], max_len_wave)

        return batch_dict

    def _partition_dataset(self, training_path: str | None) -> None:
        """
        Partition the dataset into training and validation sets.

        If a partition file exists, it is loaded. Otherwise, a new partition
        is created and saved. The indices are stored in the class attributes.

        Args:
            training_path: Path for saving/loading the partition.
        """
        if training_path is None:
            dataset_size = len(self.dataset)
            train_size = int(self.data_config['train_val_split'] * dataset_size)
            perm_idx = torch.randperm(dataset_size).tolist()
            self.train_indices = perm_idx[:train_size]
            self.val_indices = perm_idx[train_size:]
        else:
            partition_file = os.path.join(training_path, 'partition.json')
            if os.path.exists(partition_file):
                with open(partition_file, 'r') as file:
                    partition = json.load(file)
                self.train_indices = partition['training']
                self.val_indices = partition['validation']
                print('Partition loaded.')

            else:
                dataset_size = len(self.dataset)
                train_size = int(self.data_config['train_val_split'] * dataset_size)
                perm_idx = torch.randperm(dataset_size).tolist()
                self.train_indices = perm_idx[:train_size]
                self.val_indices = perm_idx[train_size:]
                with open(partition_file, 'w') as file:
                    json.dump({'training': self.train_indices, 'validation': self.val_indices}, file)
                print('Partition saved.')

    def train_dataloader(self) -> DataLoader:
        """
        Get the DataLoader for the training dataset.

        Returns:
            DataLoader for training.
        """
        train_sampler = BucketSampler(self.dataset, self.data_config['batch_size'], subsample_idx=self.train_indices,
                                      shuffle=True)
        num_workers = self.data_config['num_workers']
        return DataLoader(self.dataset, batch_sampler=train_sampler, num_workers=num_workers,
                          collate_fn=self._collate_batch)

    def val_dataloader(self) -> DataLoader:
        """
        Get the DataLoader for the validation dataset.

        Returns:
            DataLoader for validation.
        """
        val_sampler = BucketSampler(self.dataset, self.data_config['batch_size'], subsample_idx=self.val_indices,
                                    shuffle=False)
        num_workers = self.data_config['num_workers']
        return DataLoader(self.dataset, batch_sampler=val_sampler, num_workers=num_workers,
                          collate_fn=self._collate_batch)
