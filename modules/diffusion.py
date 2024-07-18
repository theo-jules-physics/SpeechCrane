import torch
from torch import nn
from utils.modules import TTSModule


class DiffusionEmbedding(TTSModule):
    """
    DiffusionEmbedding generates embeddings for diffusion steps, using sinusoidal positional encoding
    and two linear layers with SiLU activation.

    Args:
        module_config (dict): Dictionary containing the configuration for the module.
            - max_steps (int): Maximum number of diffusion steps.
            - embed_dim (int): Dimension of the embedding vector.
            - hidden_dim (int): Dimension of the hidden layer.
    """
    def __init__(self, module_config, global_config):
        mandatory_keys = ['max_steps', 'embed_dim', 'output_dimension', 'hidden_channels']
        self.update_keys(mandatory_keys)
        super(DiffusionEmbedding, self).__init__(module_config, global_config)
        self.max_steps = module_config['max_steps']
        self.embed_dim = module_config['embed_dim']
        self.hidden_dim = module_config['hidden_channels']
        self.register_buffer('embedding', self._create_embedding_vector(self.module_config['max_steps'],
                                                                        self.module_config['embed_dim']),
                             persistent=False)
        self.linear_1 = nn.Linear(self.module_config['embed_dim'], self.module_config['hidden_channels'])
        self.linear_2 = nn.Linear(self.module_config['hidden_channels'], self.module_config['output_dimension'])
        self.activation = nn.SiLU()

    @staticmethod
    def _create_embedding_vector(max_steps, embed_dim):
        """
        Creates the embedding vector using sinusoidal functions.

        Args:
            max_steps (int): Maximum number of diffusion steps.
            embed_dim (int): Dimension of the embedding vector.

        Returns:
            Tensor: Embedding vector with shape (max_steps, embed_dim).
        """
        time_tensor = torch.arange(max_steps).unsqueeze(1).float()
        dim_tensor = torch.arange(embed_dim//2).unsqueeze(0).float()
        cos_term = torch.cos(time_tensor / 10.**(dim_tensor / (embed_dim//2-1)))
        sin_term = torch.sin(time_tensor / 10.**(dim_tensor / (embed_dim//2-1)))
        return torch.cat([cos_term, sin_term], dim=1)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._interp_time(diffusion_step)
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        return x

    def _interp_time(self, t):
        """
        Interpolates the embedding vectors for non-integer diffusion steps.

        Args:
            t (Tensor): Tensor containing non-integer diffusion steps.

        Returns:
            Tensor: Interpolated embedding vectors.
        """
        floor_t = torch.floor(t).long()
        low_enc = self.embedding[floor_t]
        high_enc = self.embedding[floor_t+1]
        return low_enc + (high_enc - low_enc) * (t - floor_t).unsqueeze(-1)

    @staticmethod
    def sample_config():
        return {'max_steps': 1000, 'embed_dim': 256, 'output_dimension': 256, 'hidden_channels': 256}