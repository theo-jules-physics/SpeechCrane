import os
import torch
import torch.nn.functional as F
import numpy as np
from yaml.loader import SafeLoader
from typing import Union
import yaml


def weighted_average_overlap(outputs: list[np.ndarray], overlap: int) -> np.ndarray:
    """
    Creates a weighted average overlap between segments of outputs to create a smooth transition.

    This function is useful for combining multiple audio segments with overlapping regions,
    ensuring a smooth transition between them.

    Args:
        outputs (List[np.ndarray]): List of output segments, each as a numpy array.
        overlap (int): Number of elements to overlap between segments.

    Returns:
        np.ndarray: Final output array with weighted overlaps applied.

    Example:
        >>> segment1 = np.array([1, 2, 3, 4, 5])
        >>> segment2 = np.array([4, 5, 6, 7, 8])
        >>> result = weighted_average_overlap([segment1, segment2], overlap=2)
        >>> print(result)
        [1, 2, 3, 4, 5, 6, 7, 8]
    """
    window = np.linspace(0, 1, overlap)
    final_output = []

    for i in range(len(outputs)):
        if i == 0:
            # First segment, just append up to the last overlap region
            final_output.append(outputs[i][..., :-overlap])
        else:
            # Weighted average of the overlap
            overlapped_part = (window * outputs[i - 1][..., -overlap:] +
                               (1 - window) * outputs[i][..., :overlap])
            final_output.append(overlapped_part)

            # Append non-overlapping part if it's not the last segment
            if i != len(outputs) - 1:
                final_output.append(outputs[i][..., overlap:-overlap])
            else:
                # Last segment, append all the remaining parts
                final_output.append(outputs[i][..., overlap:])

    # Concatenate all parts to form the final output
    return np.concatenate(final_output, axis=-1)


@torch.jit.script
def fused_gate_op(filter_gate: torch.Tensor) -> torch.Tensor:
    """
    Performs a fused gate operation combining tanh and sigmoid activations.

    Args:
        filter_gate (torch.Tensor): Input tensor containing both filter and gate components.
            Expected shape: (batch_size, channels, ...)

    Returns:
        torch.Tensor: Result of element-wise multiplication of tanh(filter) and sigmoid(gate).
            Has the same shape as the input tensor, but with half the number of channels.

    Note:
        The input tensor is expected to have an even number of channels, which are split
        equally between the filter and gate components.
    """
    filter, gate = filter_gate.split(filter_gate.size(1) // 2, dim=1)
    return torch.tanh(filter) * torch.sigmoid(gate)


def get_mask(lengths: torch.Tensor, max_length: int = None) -> torch.Tensor:
    """
    Generates a binary mask tensor based on the given sequence lengths.

    This function is useful for creating masks for variable-length sequences in a batch,
    often used in attention mechanisms or loss calculations.

    Args:
        lengths (torch.Tensor): 1D tensor containing the lengths of sequences in a batch.
        max_length (int, optional): Maximum length for the mask. If None, uses the maximum
            length in the 'lengths' tensor.

    Returns:
        torch.Tensor: Binary mask tensor of shape (batch_size, max_length).
            1 indicates valid positions, 0 indicates padding.

    Example:
        >>> lengths = torch.tensor([3, 5, 2])
        >>> mask = get_mask(lengths)
        >>> print(mask)
        tensor([[1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0]])
    """
    if max_length is None:
        max_length = lengths.max()
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Generates a path tensor based on a duration tensor and a mask tensor.

    This function is used to create alignments between text and audio features.

    Args:
        duration (torch.Tensor): Duration tensor of shape (batch_size, token_len).
        mask (torch.Tensor): Mask tensor of shape (batch_size, emb_len, token_len).

    Returns:
        torch.Tensor: Path tensor of shape (batch_size, emb_len, token_len).
    """
    batch_size, emb_len, token_len = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(batch_size * token_len)
    path = get_mask(cum_duration_flat, emb_len).to(mask.dtype)
    path = path.view(batch_size, token_len, emb_len)
    shift_path = F.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    path = torch.logical_xor(path, shift_path)
    path = path.transpose(1, 2) * mask
    return path


def correct_noise_scheduler(beta_schedule: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Corrects the noise schedule to be compatible with the diffusion process.

    This function adjusts the beta schedule and calculates related schedules
    used in diffusion models.

    Args:
        beta_schedule (torch.Tensor): Initial beta schedule tensor.
        tau (float): Correction parameter.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - Corrected beta schedule
            - Corrected alpha schedule
            - Corrected alpha bar schedule
            - Corrected square root alpha bar schedule
    """
    alpha_schedule = 1 - beta_schedule
    alpha_bar_schedule = torch.cumprod(alpha_schedule, dim=0)
    sqrt_ab_schedule = alpha_bar_schedule**0.5
    end_sqrt_ab = sqrt_ab_schedule[-1]
    init_sqrt_ab = sqrt_ab_schedule[0]
    nom = init_sqrt_ab * (sqrt_ab_schedule - end_sqrt_ab + tau)
    denom = init_sqrt_ab - end_sqrt_ab + tau
    cor_sqrt_ab_schedule = nom / denom
    corr_ab_schedule = cor_sqrt_ab_schedule**2
    corr_alpha_schedule = corr_ab_schedule[1:] / corr_ab_schedule[:-1]
    corr_alpha_schedule = torch.cat([corr_ab_schedule[:1], corr_alpha_schedule])
    corr_beta_schedule = 1 - corr_alpha_schedule
    return corr_beta_schedule, corr_alpha_schedule, corr_ab_schedule, cor_sqrt_ab_schedule


def find_project_root(start_path: str = None, sentinel: str = ".git") -> str:
    """
    Finds the project root directory by searching for a sentinel file/folder.

    This function traverses up the directory tree from the start path,
    looking for a specified sentinel file or folder.

    Args:
        start_path (str, optional): The starting path for the search.
            If None, uses the current file's directory.
        sentinel (str): The file or folder name to search for as an
            indicator of the project root. Default is ".git".

    Returns:
        str: Path to the project root directory.

    Raises:
        ValueError: If the sentinel is not found in any parent directories.
    """
    if start_path is None:
        start_path = os.path.dirname(os.path.abspath(__file__))

    current_path = start_path

    while os.path.dirname(current_path) != current_path:
        if sentinel in os.listdir(current_path):
            return current_path
        current_path = os.path.dirname(current_path)

    raise ValueError(f"'{sentinel}' not found in any parent directories of {start_path}")


def import_config(dir_run: str) -> dict[str, str | int | float]:
    """
    Import configuration from a YAML file associated with a specific run.

    This function loads and parses a YAML configuration file, handling
    both standard YAML configurations and Weights & Biases (wandb) formats.

    Args:
        dir_run (str): Directory of the run containing the configuration file.

    Returns:
        Dict[str, Union[str, int, float]]: Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file is not found in the specified directory.

    Note:
        If the configuration is in wandb format, it extracts only the 'value' field
        from each parameter and removes the 'wandb_version' key if present.
    """
    config_file = os.path.join(dir_run, 'files', 'config.yaml')
    print(f"Loading configuration file from {config_file}")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} not found.")
    with open(config_file) as f:
        config_data = yaml.load(f, Loader=SafeLoader)
    if 'wandb_version' in config_data:
        del config_data['wandb_version']
        return {k: v['value'] for k, v in config_data.items() if 'value' in v}
    return config_data
