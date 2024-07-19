import torch
from torch import nn
from utils.modules import TTSModule


class DiffusionEmbedding(TTSModule):
    """
    DiffusionEmbedding generates embeddings for diffusion steps.

    This module creates embeddings for diffusion steps using sinusoidal positional encoding,
    followed by two linear layers with SiLU activation. It's a key component in diffusion-based
    generative models, providing time step information to the model.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the DiffusionEmbedding.

        Args:
            module_config: Configuration dictionary for the module.
            global_config: Global configuration dictionary.
        """
        mandatory_keys = ['max_steps', 'embed_dim', 'output_dimension', 'hidden_channels']
        self.update_keys(mandatory_keys)
        super(DiffusionEmbedding, self).__init__(module_config, global_config)

        # Create and register the embedding vector
        self.register_buffer('embedding', self._create_embedding_vector(self.module_config['max_steps'],
                                                                        self.module_config['embed_dim']),
                             persistent=False)

        # Define the neural network layers
        self.linear_1 = nn.Linear(self.module_config['embed_dim'], self.module_config['hidden_channels'])
        self.linear_2 = nn.Linear(self.module_config['hidden_channels'], self.module_config['output_dimension'])
        self.activation = nn.SiLU()

    @staticmethod
    def _create_embedding_vector(max_steps: int, embed_dim: int) -> torch.Tensor:
        """
        Creates the embedding vector using sinusoidal functions.

        Args:
            max_steps: Maximum number of diffusion steps.
            embed_dim: Dimension of the embedding vector.

        Returns:
            Embedding vector with shape (max_steps, embed_dim).
        """
        time_tensor = torch.arange(max_steps).unsqueeze(1).float()
        dim_tensor = torch.arange(embed_dim//2).unsqueeze(0).float()
        cos_term = torch.cos(time_tensor / 10.**(dim_tensor / (embed_dim//2-1)))
        sin_term = torch.sin(time_tensor / 10.**(dim_tensor / (embed_dim//2-1)))
        return torch.cat([cos_term, sin_term], dim=1)

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DiffusionEmbedding.

        Args:
            diffusion_step: Tensor containing diffusion steps.

        Returns:
            Embedded representation of the diffusion steps.
        """
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._interp_time(diffusion_step)
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        return x

    def _interp_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolates the embedding vectors for non-integer diffusion steps.

        Args:
            t: Tensor containing non-integer diffusion steps.

        Returns:
            Interpolated embedding vectors.
        """
        floor_t = torch.floor(t).long()
        low_enc = self.embedding[floor_t]
        high_enc = self.embedding[floor_t+1]
        return low_enc + (high_enc - low_enc) * (t - floor_t).unsqueeze(-1)

    @staticmethod
    def sample_config() -> dict:
        """
        Provides a sample configuration for the DiffusionEmbedding.
        """
        return {'max_steps': 1000, 'embed_dim': 256, 'output_dimension': 256, 'hidden_channels': 256}