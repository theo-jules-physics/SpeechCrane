from torch.utils.data import Dataset, DataLoader
from utils.dataset import collate_batch
import pytorch_lightning as pl
import torch
import json
import importlib


def mask_data(data, data_length):
    max_len = data.shape[-1]
    mask = torch.zeros(max_len)
    mask[:data_length] = 1
    return data * mask


class DummyDataset(Dataset):
    def __init__(self, num_sample, feature_list, feature_config):
        self.num_sample = num_sample
        self.feature_list = feature_list
        self.feature_config = feature_config

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        dummy_item = {}
        for feature in self.feature_list:
            gen_data = self._generate_dummy_feature(feature)
            dummy_item.update(gen_data)
        return dummy_item

    def _generate_dummy_feature(self, feature_name):
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
    def __init__(self, test_config, feature_config, feature_list):
        super().__init__()
        self.test_config = test_config
        self.feature_config = feature_config
        self.feature_list = feature_list

    def _collate_batch(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate a batch of samples into padded tensors.

        Args:
            batch: List of samples.

        Returns:
            Dictionary of padded feature tensors and lengths.
        """
        return collate_batch(batch, self.feature_list, self.feature_config)

    def train_dataloader(self) -> DataLoader:
        dataset = DummyDataset(self.test_config['data_size'],
                               self.feature_list,
                               self.test_config['feature_config'])
        return DataLoader(dataset, batch_size=self.test_config['batch_size'], collate_fn=self._collate_batch)

    def val_dataloader(self):
        dataset = DummyDataset(self.test_config['data_size'] // 10,
                               self.feature_list,
                               self.test_config['feature_config'])
        return DataLoader(dataset, batch_size=self.test_config['batch_size'], collate_fn=self._collate_batch)


def run() -> None:
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
