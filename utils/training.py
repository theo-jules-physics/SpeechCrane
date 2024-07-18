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


def wandb_init(data_dir: str, project_name: str, run_name: str | None = None,
               run_id: str | None = None) -> WandbLogger:
    """
    Initialize a Weights and Biases (wandb) run and create a WandbLogger.

    This function sets up wandb logging, either starting a new run or resuming an existing one.
    It also modifies the run name by appending the run ID and finalizes the wandb run.

    Args:
        data_dir: Directory to save wandb data.
        project_name: Name of the wandb project.
        run_name: Name of the wandb run. If None, wandb will auto-generate a name.
        run_id: ID of an existing wandb run to resume. If None, a new run is started.

    Returns:
        Initialized WandbLogger for use with PyTorch Lightning.

    Note:
        This function modifies the run name by appending the run ID and finalizes the wandb run.
        The WandbLogger is then reinitialized to continue logging.

    Example:
        >>> logger = wandb_init('/path/to/data', 'my_project', 'experiment_1')
        >>> print(logger.name)
        'experiment_1_<generated_id>'
    """
    if run_id:
        wandb.init(project=project_name, dir=data_dir, name=run_name, id=run_id, resume=True)

    else:
        wandb.init(project=project_name, dir=data_dir, name=run_name)
    wandb.run.name += '_' + wandb.run.id
    wandb.finish()
    wandb_logger = WandbLogger()
    return wandb_logger
