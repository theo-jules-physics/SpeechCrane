from dummy_data import create_dummy_dataset
from utils.preprocessing import preprocess_data
import os


def run():
    base_path = "E:\\ML\\Datasets"
    dummy_dataset_path = create_dummy_dataset(base_path, num_samples=20000, parallel=True)
    # dummy_dataset_path = os.path.join(base_path, 'dummy_dataset')
    data_config = {'path': dummy_dataset_path,
                   'features': ['waveform'],
                   'format': 'wav',
                   'parallel_features': ['waveform']}

    preprocess_data(data_config)


if __name__ == '__main__':
    run()

