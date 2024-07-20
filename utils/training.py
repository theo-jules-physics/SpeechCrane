import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


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
