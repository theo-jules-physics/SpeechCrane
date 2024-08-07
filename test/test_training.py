from torch.utils.data import Dataset, DataLoader
from utils.dataset import collate_batch
import pytorch_lightning as pl
import torch
import json
import importlib


def mask_data(data: torch.Tensor, data_length: int) -> torch.Tensor:
    """
    Apply a mask to the data based on the specified length.

    Args:
        data (torch.Tensor): The input data to be masked.
        data_length (int): The length up to which the data should be kept.

    Returns:
        torch.Tensor: The masked data.
    """
    max_len = data.shape[-1]
    mask = torch.zeros(max_len)
    mask[:data_length] = 1
    return data * mask


class DummyDataset(Dataset):
    """
    A dataset that generates dummy data for TTS model testing.

    Args:
        num_sample (int): The number of samples in the dataset.
        feature_list (list): List of feature names to generate.
        feature_config (dict): Configuration for each feature.
    """
    def __init__(self, num_sample: int, feature_list: list[str], feature_config: dict):
        self.num_sample = num_sample
        self.feature_list = feature_list
        self.feature_config = feature_config

    def __len__(self) -> int:
        return self.num_sample

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Generate a dummy item with all required features.

        Args:
            idx (int): Index of the item to generate.

        Returns:
            dict: A dictionary containing the generated features.
        """
        dummy_item = {}
        for feature in self.feature_list:
            gen_data = self._generate_dummy_feature(feature)
            dummy_item.update(gen_data)
        return dummy_item

    def _generate_dummy_feature(self, feature_name: str) -> dict[str, torch.Tensor]:
        """
        Generate dummy data for a specific feature.

        Args:
            feature_name (str): Name of the feature to generate.

        Returns:
            dict: A dictionary containing the generated feature and its length.

        Raises:
            ValueError: If an unsupported dtype is specified in the feature config.
        """
        feature_characs = self.feature_config[feature_name]
        shape = feature_characs['shape']
        dtype = feature_characs['dtype']
        gen_data = {}
        if dtype == 'float32':
            dummy_data = 2 * torch.rand(shape) - 1
            gen_data[f'{feature_name}_length'] = torch.randint(shape[-1] // 2, shape[-1], (1,))
            gen_data[feature_name] = mask_data(dummy_data, gen_data[f'{feature_name}_length'])

        elif dtype == 'int64':
            dummy_data = torch.randint(0, 10, shape)
            gen_data[f'{feature_name}_length'] = torch.randint(shape[-1] // 2, shape[-1], (1,))
            gen_data[feature_name] = mask_data(dummy_data, gen_data[f'{feature_name}_length'])

        elif dtype == 'str':
            dummy_data = ' '.join([chr(65 + i) for i in range(shape[-1] // 2)])
            gen_data[feature_name] = dummy_data

        else:
            raise ValueError(f"Unimplemented dtype: {dtype}")

        return gen_data


class DummyDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule that uses the DummyDataset for training and validation.

    Args:
        test_config (dict): Configuration for the test run.
        feature_config (dict): Configuration for each feature.
        feature_list (list): List of feature names to generate.
    """
    def __init__(self, test_config: dict, feature_config: dict, feature_list: list[str]):
        super().__init__()
        self.test_config = test_config
        self.feature_config = feature_config
        self.feature_list = feature_list

    def _collate_batch(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate a batch of samples into padded tensors.

        Args:
            batch (list[dict]): List of samples.

        Returns:
            dict[str, torch.Tensor]: Dictionary of padded feature tensors and lengths.
        """
        return collate_batch(batch, self.feature_list, self.feature_config)

    def train_dataloader(self) -> DataLoader:
        """
        Create and return a DataLoader for the training data.

        Returns:
            DataLoader: The training data loader.
        """
        dataset = DummyDataset(self.test_config['data_size'],
                               self.feature_list,
                               self.test_config['feature_config'])
        return DataLoader(dataset, batch_size=self.test_config['batch_size'], collate_fn=self._collate_batch)

    def val_dataloader(self) -> DataLoader:
        """
        Create and return a DataLoader for the validation data.

        Returns:
            DataLoader: The validation data loader.
        """
        dataset = DummyDataset(self.test_config['data_size'] // 10,
                               self.feature_list,
                               self.test_config['feature_config'])
        return DataLoader(dataset, batch_size=self.test_config['batch_size'], collate_fn=self._collate_batch)


def run() -> None:
    """
    Main function to run the training process with dummy data.

    This function performs the following steps:
    1. Load configurations from JSON files
    2. Initialize the model
    3. Set up the DummyDataModule
    4. Configure and start the PyTorch Lightning Trainer
    """
    test_config = json.load(open('test/test_config.json', 'r'))
    features_config = json.load(open('configs/features.json', 'r'))
    trainer_config = test_config['trainer_config']
    name_net = test_config['model_name']
    model_config = json.load(open(f'configs/{name_net}.json', 'r'))
    model_config['training'].update(test_config['generic'])
    module_path, feature_list = json.load(open(f'configs/model_list.json', 'r'))[name_net]
    model_module = importlib.import_module(module_path)

    model_class = getattr(model_module, name_net)
    device = model_config['training']['device']
    if device == 'cuda':
        torch.set_float32_matmul_precision('medium')
    model = model_class(model_config, test_config, device)

    data_module = DummyDataModule(test_config, features_config, feature_list)

    trainer = pl.Trainer(**trainer_config)
    trainer.fit(model, data_module)
