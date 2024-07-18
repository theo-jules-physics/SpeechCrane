import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from utils.modules import remove_weight_norm_recursively


class BaseDiff(LightningModule):
    """
    BaseDiff is a PyTorch Lightning module implementing a denoising diffusion probabilistic model for waveform
    generation.

    Attributes:
        data_config: Configuration dictionary for the data.
        arch_config: Configuration dictionary for the model architecture.
        optimizers_config: Configuration dictionary for the optimizers.
        training_config: Configuration dictionary for the training process.
        sampling_config: Configuration dictionary for the sampling process.
    """
    def __init__(self, model_config: dict, data_config: dict, device: torch.device):
        """
        Initializes the BaseDiff model with the provided configuration and device.

        Args:
            model_config: Configuration dictionary for the model.
            data_config: Configuration dictionary for the data.
            device: Device to run the model on.
        """
        super(BaseDiff, self).__init__()
        self.to(device)
        self.save_hyperparameters()
        self.data_config = data_config
        self.arch_config = model_config['architecture']
        self.optimizers_config = model_config['optimizers']
        self.training_config = model_config['training']
        self.sampling_config = model_config['sampling']

        self._init_model()
        self._init_noise_schedule()
        self._init_loss()
        self.to(device)

    def _init_model(self) -> None:
        """
        Initializes the model architecture. Should be implemented by subclasses.
        """
        raise NotImplementedError

    def _init_noise_schedule(self) -> None:
        """
        Initializes the noise schedule for both training and inference.
        """
        self.beta_schedule = {'training': torch.linspace(self.sampling_config['beta_min'],
                                                         self.sampling_config['beta_max'],
                                                         self.sampling_config['timesteps_diffusion'],
                                                         device=self.device),
                              'inference': torch.tensor(self.sampling_config['fast_sampling_schedule'],
                                                        device=self.device)}
        self.alpha_schedule = {'training': 1 - self.beta_schedule['training'], 'inference': 1 - self.beta_schedule['inference']}

        self.alpha_bar_schedule = {'training': torch.cumprod(self.alpha_schedule['training'], 0).to(self.device),
                                   'inference': torch.cumprod(self.alpha_schedule['inference'], 0).to(self.device)}

        self.fast_inference_timesteps = self._compute_time_step_fast_inference()

    def _init_loss(self) -> None:
        """
        Initializes the loss function based on the configuration.
        """
        if self.training_config['loss_function'] == 'l1':
            self.loss_function = F.l1_loss
        elif self.training_config['loss_function'] == 'mse':
            self.loss_function = F.mse_loss
        else:
            raise ValueError('Invalid loss function')

    def _compute_time_step_fast_inference(self) -> list[float]:
        """
        Computes the time steps for fast inference by matching training and inference noise schedules.

        Returns:
            Time steps for fast inference.
        """
        times_steps = []
        for s in range(len(self.beta_schedule['inference'])):
            for t in range(len(self.beta_schedule['training'])):
                if (self.alpha_bar_schedule['training'][t+1] <= self.alpha_bar_schedule['inference'][s]
                        <= self.alpha_bar_schedule['training'][t]):
                    nom = self.alpha_bar_schedule['training'][t]**0.5 - self.alpha_bar_schedule['inference'][s]**0.5
                    denom = self.alpha_bar_schedule['training'][t]**0.5 - self.alpha_bar_schedule['training'][t+1]**0.5
                    t_correction = (nom/denom).cpu().item()
                    times_steps.append(t + t_correction)
                    break
        return times_steps

    def _generate_random_noisy_input(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates random noisy input for a given input tensor based on the noise schedule.

        Args:
            input: Input tensor.

        Returns:
            Tuple of alpha bar and random time tensors.
        """
        batch_size = input.size(0)
        rand_time = torch.randint(0, self.sampling_config['timesteps_diffusion'], (batch_size,), device=self.device)
        alpha_bar = self.alpha_bar_schedule['training'][rand_time.unsqueeze(-1)].unsqueeze(-1)
        return alpha_bar, rand_time

    @staticmethod
    def _check_for_nan(inp: torch.Tensor) -> None:
        """
        Checks for NaN values in the input tensor and raises an error if found.

        Args:
            inp: Input tensor.

        Raises:
            ValueError: If NaN values are found in the tensor.
        """
        if torch.isnan(inp).any():
            raise ValueError('NaN values in tensor')

    def _preprocess_batch(self, batch: dict) -> dict:
        """
        Preprocesses a batch of data. This method should be overridden by subclasses.

        Args:
            batch: Batch of data.

        Returns:
            Preprocessed batch of data.
        """
        return batch

    def forward(self, x: torch.Tensor, diffusion_step: torch.Tensor, x_mask: torch.Tensor, condition: dict) -> torch.Tensor:
        raise NotImplementedError

    def _compute_loss(self, preprocessed_batch: dict) -> torch.Tensor:
        """
        Computes the loss for a preprocessed batch of data.

        Args:
            preprocessed_batch: Preprocessed batch of data.

        Returns:
            Computed loss.
        """
        input = preprocessed_batch['input']
        mask = preprocessed_batch['mask']
        conditions = preprocessed_batch['conditions']
        noise = torch.randn_like(input)
        alpha_bar, rand_time = self._generate_random_noisy_input(input)
        noisy_input = (alpha_bar**0.5 * input + (1 - alpha_bar)**0.5 * noise) * mask
        noise_prediction = self(noisy_input, rand_time, mask, conditions)
        clipped_noise = noise * mask
        return self.loss_function(clipped_noise, noise_prediction)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step, including preprocessing the batch, computing the loss, and logging.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            Computed training loss.
        """
        preprocessed_batch = self._preprocess_batch(batch)
        train_loss = self._compute_loss(preprocessed_batch)
        self._check_for_nan(train_loss)
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Performs a single validation step, including preprocessing the batch, computing the loss, and logging.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            Computed validation loss.
        """
        preprocessed_batch = self._preprocess_batch(batch)
        val_loss = self._compute_loss(preprocessed_batch)
        self._check_for_nan(val_loss)
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizers for the model based on the training configuration.

        Returns:
            Dictionary containing the optimizer.
        """
        mod = __import__('torch.optim', fromlist=[self.optimizers_config['optimizer']])
        opti = getattr(mod, self.optimizers_config['optimizer'])
        optimizer = opti(self.parameters(), lr=self.optimizers_config['lr'],
                         betas=(self.optimizers_config['beta1'], self.optimizers_config['beta2']),
                         eps=self.optimizers_config['eps'], weight_decay=self.optimizers_config['weight_decay'])
        return {
            'optimizer': optimizer,
        }

    def denoising_step(self, inputs: torch.Tensor, mask: torch.Tensor, condition: dict, time: torch.Tensor,
                       step: int, schedule_type: str = 'training') -> torch.Tensor:
        """
        Performs a single denoising step during waveform generation.

        Args:
            inputs: Noisy input tensor.
            mask: Mask tensor.
            condition: Condition tensor.
            time: Time tensor.
            step: Current step in the denoising process.
            schedule_type: Type of schedule ('training' or 'inference').

        Returns:
            Denoised waveform.
        """
        noise_prediction = self(inputs, time, mask, condition)
        beta = self.beta_schedule[schedule_type][step]
        alpha = 1 - beta
        alpha_bar = self.alpha_bar_schedule[schedule_type][step]
        delta_mu = beta / (1 - alpha_bar) ** 0.5 * noise_prediction
        denoised_waveform = 1 / alpha ** 0.5 * (inputs - delta_mu)
        if step > 0:
            scaling = ((1 - self.alpha_bar_schedule[schedule_type][step - 1]) / (1 - alpha_bar) * beta) ** 0.5
            step_noise = scaling * torch.randn_like(inputs)
            denoised_waveform = denoised_waveform + step_noise
        return torch.clamp(denoised_waveform, -1.0, 1.0)

    def inference(self, mel: torch.Tensor, scaling_condition: int = 1, output_dim: int = 1,
                  fast_sampling: bool = False) -> torch.Tensor:
        """
        Performs inference to generate waveform from mel spectrogram.

        Args:
            mel: Input mel spectrogram.
            scaling_condition: Scaling factor for condition.
            output_dim: Dimension of the output waveform.
            fast_sampling: Whether to use fast sampling.

        Returns:
            Generated waveform.
        """
        batch_size = 1
        condition = {'loc_cond': self.upsampler(mel), 'g_cond': None}
        output_length = torch.tensor(condition['loc_cond'].shape[-1]) * scaling_condition
        print(f'Generating waveform with length: {output_length}')
        gen_output = torch.randn(batch_size, output_dim, output_length, device=self.device)
        condition = {'loc_cond': self.upsampler(mel), 'g_cond': None}

        if fast_sampling:
            print('Generating waveforms with fast sampling')
            time_steps = self.fast_inference_timesteps
            with torch.no_grad():
                for step in range(len(time_steps)-1, -1, -1):
                    time = torch.ones(batch_size, device=self.device) * time_steps[step]
                    gen_output = self.denoising_step(gen_output, None, condition, time, step, schedule_type='inference')

        else:
            print('Generating waveforms with training sampling')
            with torch.no_grad():
                for step in range(len(self.beta_schedule['training'])-1, -1, -1):
                    time = torch.ones(batch_size, device=self.device).to(torch.long) * step
                    gen_output = self.denoising_step(gen_output, None, condition, time, step, schedule_type='training')

        return gen_output

    def remove_weight_norm(self) -> None:
        """
        Removes weight normalization from all modules in the model.
        """
        remove_weight_norm_recursively(self)


if __name__ == '__main__':
    None
