from utils.loss_functions import gloss, dloss, fmap_loss
from utils.modules import remove_weight_norm_recursively
import torch
from torch import nn
import pytorch_lightning as pl
import itertools
from torch.cuda.amp import GradScaler, autocast


class BaseGAN(pl.LightningModule):
    """
    BaseGAN provides a base class for Generative Adversarial Networks (GANs) using PyTorch Lightning. It handles
    common operations and configurations required for training GANs, including initialization, training and validation
    steps, optimizer configuration, and gradient accumulation.

    Attributes:
        data_config (dict): Configuration dictionary for the data.
        arch_config (dict): Configuration dictionary for the model architecture.
        optimizers_config (dict): Configuration dictionary for the optimizers.
        training_config (dict): Configuration dictionary for the training process.
        n_speakers (int): Number of speakers in the dataset.
        use_amp (bool): Whether to use automatic mixed precision training.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        use_sdp (bool): Whether to use stochastic differential processes.
        accu_steps (int): Number of steps for gradient accumulation.
        automatic_optimization (bool): Whether to use automatic optimization.
        accu_counter (int): Counter for gradient accumulation.
        main_loss_flag (bool): Flag to indicate whether to compute main loss.
        disc_loss_flag (bool): Flag to indicate whether to compute discriminator loss.
        gen_loss_flag (bool): Flag to indicate whether to compute generator loss.
        gen_training (bool): Flag to indicate whether the generator is in training mode.
        list_gen (list): List to store generator models.
        list_disc (list): List to store discriminator models.
    """
    def __init__(self, model_config: dict, data_config: dict, device: torch.device):
        """
        Initializes the BaseGAN.

        Args:
            model_config: Configuration dictionary for the model.
            data_config: Configuration dictionary for the data.
            device: Device to run the model on.
        """
        super(BaseGAN, self).__init__()
        self.to(device)
        self.save_hyperparameters()
        self.data_config = data_config
        self.arch_config = model_config['architecture']
        self.optimizers_config = model_config['optimizers']
        self.training_config = model_config['training']

        self.n_speakers = self.data_config.get('n_speakers', 1)
        self.use_amp = self.training_config.get('use_amp', True)
        self.scaler = GradScaler(init_scale=10_000, enabled=self.use_amp)
        self.use_sdp = model_config.get('use_sdp', False)
        self.accu_steps = self.training_config.get('grad_accumulate_steps', 1)
        self.automatic_optimization = False
        self.accu_counter = 0
        self.main_loss_flag = True
        self.disc_loss_flag = True
        self.gen_loss_flag = True
        self.gen_training = True
        self._init_models()

    def _init_models(self) -> None:
        """
        Initializes the generator and discriminator models. This method should be overridden by subclasses.
        """
        self.list_gen = nn.ModuleList()
        self.list_disc = nn.ModuleList()

    def get_disc_outs(self, gen_output: torch.Tensor, true_output: torch.Tensor) -> dict:
        """
        Get discriminator outputs for generated and true data.

        Args:
            gen_output: Generated output.
            true_output: True output.

        Returns:
            Dictionary containing discriminator outputs and feature maps.
        """
        disc_out_dict = {'gen_outs': [], 'true_outs': [], 'gen_fmaps': [], 'true_fmaps': []}

        for disc in self.list_disc.values():
            disc_gen_outs, disc_gen_fmaps = disc(gen_output)
            disc_true_outs, disc_true_fmaps = disc(true_output)
            disc_out_dict['gen_outs'].append(disc_gen_outs)
            disc_out_dict['true_outs'].append(disc_true_outs)
            disc_out_dict['gen_fmaps'].append(disc_gen_fmaps)
            disc_out_dict['true_fmaps'].append(disc_true_fmaps)

        return disc_out_dict

    def _preprocess_batch(self, batch: dict) -> dict:
        """
        Preprocesses a batch of data. This method should be overridden by subclasses.

        Args:
            batch: Batch of data.

        Returns:
            Preprocessed batch of data.
        """
        return batch

    def _main_loss(self, gen_output: torch.Tensor, true_output: torch.Tensor) -> dict:
        """
        Calculates the main loss. This method should be overridden by subclasses.

        Args:
            gen_output: Generated output.
            true_output: True output.

        Returns:
            Dictionary of main loss values.
        """
        raise NotImplementedError

    def _val_loss(self, gen_output: torch.Tensor, true_output: torch.Tensor) -> dict:
        """
        Calculates the validation loss. This method should be overridden by subclasses.

        Args:
            gen_output: Generated output.
            true_output: True output.

        Returns:
            Dictionary of validation loss values.
        """
        raise NotImplementedError

    def forward(self, preprocessed_batch: dict) -> dict:
        """
        Forward pass for the BaseGAN. This method should be overridden by subclasses.

        Args:
            preprocessed_batch: Preprocessed batch of data.

        Returns:
            Output from the model.
        """
        raise NotImplementedError

    def training_step(self, batch: dict, batch_idx: int) -> None:
        """
        Performs a single training step for the GAN, including preprocessing the batch,
        computing losses, and updating the model parameters.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.
        """
        preprocessed_batch = self._preprocess_batch(batch)

        with autocast(enabled=self.use_amp):
            output_dict = self(preprocessed_batch)
            gen_output = output_dict['gen_out']
            true_output = preprocessed_batch['true_ouput']

            # Compute main loss if flag is set
            if self.main_loss_flag:
                main_loss = self._main_loss(gen_output, true_output)
                self.log_dict(main_loss)

            # Compute discriminator loss if flag is set
            if self.disc_loss_flag:
                disc_out_dict = self.get_disc_outs(gen_output, true_output)
                d_loss, _, _ = dloss(disc_out_dict, self.training_config['gan_loss'])
                self.log('disc_loss', d_loss)

            # Compute generator loss if flag is set
            if self.gen_loss_flag:
                gen_loss, _ = gloss(disc_out_dict, self.training_config['gan_loss'])
                fm_loss = fmap_loss(disc_out_dict) * self.training_config['lambda_fmap']
                self.log_dict({'gen_loss': gen_loss, 'fmap_loss': fm_loss})

        # Backpropagate discriminator loss if flag is set
        if self.disc_loss_flag:
            self.manual_backward(self.scaler.scale(d_loss), retain_graph=self.gen_loss_flag)

        # Backpropagate main loss if flag is set
        if self.main_loss_flag:
            loss = sum(main_loss.values())
            if self.gen_training:
                loss = loss + gen_loss + fm_loss
            self.manual_backward(self.scaler.scale(loss))

        # Gradient accumulation and optimization step
        self.accu_counter += 1
        if self.accu_counter == self.accu_steps:
            opt_g, opt_d = self.optimizers()
            if self.disc_loss_flag:
                self.scaler.step(opt_d)
                self.optimizer_zero_grad(batch_idx, 0, opt_d)
            if self.main_loss_flag:
                self.scaler.step(opt_g)
                self.optimizer_zero_grad(batch_idx, 1, opt_g)
            self.accu_counter = 0
            self.scaler.update()
            if self.gen_loss_flag:
                self.gen_training = not self.gen_training

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Performs a single validation step.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            Validation loss.
        """
        preprocessed_batch = self._preprocess_batch(batch)
        output_dict = self(preprocessed_batch)
        gen_output = output_dict['gen_out']
        true_output = preprocessed_batch['true_ouput']
        val_loss = self._val_loss(gen_output, true_output)
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self) -> tuple[list, list]:
        """
        Configures optimizers and learning rate schedulers.

        Returns:
            Tuple of lists containing optimizers and schedulers.
        """
        mod = __import__('torch.optim', fromlist=[self.optimizers_config['optimizer']])
        opti = getattr(mod, self.optimizers_config['optimizer'])
        gen_params = itertools.chain(*[model.parameters() for model in self.list_gen.values()])
        g_optim = opti(gen_params,
                       lr=self.optimizers_config['g_lr'],
                       weight_decay=self.optimizers_config['g_weight_decay'],
                       betas=(self.optimizers_config['g_beta1'], self.optimizers_config['g_beta2']))
        disc_params = itertools.chain(*[model.parameters() for model in self.list_disc.values()])
        d_optim = opti(disc_params,
                       lr=self.optimizers_config['d_lr'],
                       weight_decay=self.optimizers_config['d_weight_decay'],
                       betas=(self.optimizers_config['d_beta1'], self.optimizers_config['d_beta2']))
        g_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_optim, gamma=self.optimizers_config['g_lr_decay'])
        d_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optim, gamma=self.optimizers_config['d_lr_decay'])
        return [g_optim, d_optim], [g_lr_scheduler, d_lr_scheduler]

    def inference(self, src: torch.Tensor, src_length: torch.Tensor, sid: torch.Tensor | None = None,
                  noise_scale: float = 1) -> torch.Tensor:
        """
        Performs inference. This method should be overridden by subclasses.

        Args:
            src: Source tensor.
            src_length: Length of the source tensor.
            sid: Speaker ID. Default is None.
            noise_scale: Scale of the noise. Default is 1.

        Returns:
            Inference result.
        """
        raise NotImplementedError

    def remove_weight_norm(self) -> None:
        """
        Removes weight normalization from all modules in the model.
        """
        remove_weight_norm_recursively(self)