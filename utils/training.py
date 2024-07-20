import json
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


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


def create_callbacks(file_dir, monitor='val_loss'):
    """
    Create a ModelCheckpoint callback for saving model checkpoints.

    Args:
        file_dir: Directory to save checkpoints.
        monitor: Quantity to monitor. Defaults to 'val_loss'.

    Returns:
        Checkpoint callback object configured to save the best models.

    Example:
        >>> callback = create_callbacks('/path/to/checkpoints', monitor='accuracy')
        >>> print(callback.dirpath)
        '/path/to/checkpoints'
    """
    checkpoint_callback = ModelCheckpoint(
        dirpath=file_dir,  # path to save checkpoints
        verbose=True,
        monitor=monitor,
        mode="min",
        save_top_k=3,  # save the top 1 models
        save_last=True
    )
    return checkpoint_callback
