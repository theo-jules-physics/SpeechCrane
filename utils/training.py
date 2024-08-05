import json
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.callbacks import AudioLoggingCallback
import os


def get_config() -> tuple[dict, dict, dict, str, dict]:
    """
    Load and merge configuration files.

    Returns:
        Tuple containing training config, trainer config, data config, model name, and model config.
    """
    training_config = json.load(open('configs/training.json', 'r'))
    trainer_config = training_config['trainer_config']
    data_config = json.load(open('configs/data.json', 'r'))
    name_net = training_config['model_name']
    model_config = json.load(open(f'configs/{name_net}.json', 'r'))
    model_config['training'].update(training_config['generic'])
    trainer_config.update(model_config.get('trainer', {}))
    return training_config, trainer_config, data_config, name_net, model_config


def set_callbacks(run_dir, callback_config):
    """
    Set up and configure callbacks for the training process.

    This function creates a list of callbacks based on the provided configuration.
    It always includes a ModelCheckpoint callback and optionally adds an
    AudioLoggingCallback if enabled in the configuration.

    Args:
        run_dir (str): The directory path where the run data will be saved.
        callback_config (dict): A dictionary containing configuration for the callbacks.
            Expected to have 'model_checkpoint' and 'audio_logging' keys.

    Returns:
        list: A list of configured callback objects.

    The function does the following:
    1. Creates a ModelCheckpoint callback with the configuration provided in
       callback_config['model_checkpoint'], saving checkpoints in a 'checkpoints'
       subdirectory of run_dir.
    2. If audio logging is enabled (callback_config['audio_logging']['enable'] is True),
       it creates an AudioLoggingCallback with the provided configuration.

    Example callback_config structure:
    {
        'model_checkpoint': {
            'save_top_k': 3,
            'monitor': 'val_loss',
            ...
        },
        'audio_logging': {
            'enable': True,
            'log_every_n_steps': 1000,
            ...
        }
    }
    """
    callback_list = []

    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(**callback_config['model_checkpoint'], dirpath=checkpoint_dir)
    callback_list.append(checkpoint_callback)

    if callback_config['audio_logging']['enable']:
        audio_log_callback = AudioLoggingCallback(run_dir, callback_config['audio_logging'])
        callback_list.append(audio_log_callback)

    return callback_list
