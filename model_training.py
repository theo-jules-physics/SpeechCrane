import os
import wandb
import torch
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import importlib
from utils.dataset import TTSDataModule
from utils.training import get_config, set_callbacks
from utils.preprocessing import check_preprocessing
from datetime import datetime


def run() -> None:
    """
    Main function to run the training process.
    """
    # Load configuration files and corresponding model
    training_config, trainer_config, data_config, name_net, model_config = get_config()
    module_path, features = json.load(open(f'configs/model_list.json', 'r'))[name_net]
    data_config = check_preprocessing(features, data_config)
    model_module = importlib.import_module(module_path)
    model_class = getattr(model_module, name_net)
    model_dir = os.path.join(training_config['base_path'], name_net, 'models')
    os.makedirs(model_dir, exist_ok=True)
    device = model_config['training']['device']
    if device == 'cuda':
        torch.set_float32_matmul_precision('medium')
    loggers, callbacks = [], []

    # Initialize model, if continuing from a previous run, load the last checkpoint
    if training_config['run_id'] is None:
        model = model_class(model_config, data_config, device)

    else:
        ckpt_name = 'last.ckpt'
        ckpt_file = os.path.join(model_dir, 'wandb', training_config['run_id'], 'files', ckpt_name)
        model = model_class.load_from_checkpoint(ckpt_file)

    model.to('cuda')
    # Initialize logging and callbacks. If using Weights & Biases, log to the run directory, otherwise create a new one.
    if training_config['use_wandb']:
        wandb_logger = WandbLogger(project=name_net, save_dir=model_dir, name=training_config['run_name'])
        wandb_logger.watch(model, log='all')
        loggers.append(wandb_logger)
        file_dir = wandb.run.dir
        run_dir = os.path.dirname(file_dir)
    else:
        random_id = wandb.util.generate_id()
        formated_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(model_dir, 'run_' + formated_time + '_' + random_id)
        file_dir = os.path.join(run_dir, 'files')
        model.save_config(file_dir)

    callbacks_list = set_callbacks(run_dir, training_config['callbacks'])

    data_module = TTSDataModule(model.data_config, [], features, training_path=file_dir)
    trainer_config.update({'logger': loggers,
                           'callbacks': callbacks_list,
                           'default_root_dir': file_dir,
                           'max_time': training_config['max_time']})

    # Initialize the trainer and start the training process
    trainer = pl.Trainer(**trainer_config)
    trainer.fit(model, data_module)

    # Clean up the GPU memory
    torch.cuda.empty_cache()


if __name__ == '__main__':
    run()
