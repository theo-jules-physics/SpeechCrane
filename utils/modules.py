from torch import nn
import torch

LEAKY_RELU_SLOPE = 0.1


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using shift and scale parameters.

    Args:
        x: Input tensor to be modulated.
        shift: Shift parameter for modulation.
        scale: Scale parameter for modulation.

    Returns:
        Modulated tensor.

    Example:
        >>> x = torch.randn(10, 5)
        >>> shift = torch.randn(10, 1)
        >>> scale = torch.randn(10, 1)
        >>> modulated = modulate(x, shift, scale)
    """
    return x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


def remove_weight_norm_recursively(module: nn.Module) -> None:
    """
    Remove weight normalization from all modules in the given module recursively.

    Args:
        module: The root module to start removing weight normalization from.

    Note:
        This function modifies the module in-place.
    """
    stack = [module]
    predefined_types = [nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding]
    while stack:
        children = stack.pop()
        for name, module in children.named_children():
            if isinstance(module, tuple(predefined_types)):
                try:
                    torch.nn.utils.parametrize.remove_parametrizations(module, 'weight')
                    print(f"Weight norm removed from {name}")
                except ValueError:
                    # Catch the error if weight_norm is not applied on this module
                    pass
            else:
                stack.extend(list(module.children()))


def convert_pad_shape(pad_shape: list[list[int]]) -> list[int]:
    """
    Convert the padding shape to the required format.

    Args:
        pad_shape: A list of lists representing padding for each dimension.

    Returns:
        A flattened list of padding values in the order expected by PyTorch padding functions.

    Example:
        >>> convert_pad_shape([[1, 2], [3, 4]])
        [4, 3, 2, 1]
    """
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def get_padding(kernel_size: int | list[int] | tuple[int, ...],
                stride: int | list[int] | tuple[int, ...] | None = None,
                dilation: int | list[int] | tuple[int, ...] | None = None) -> int | list[int]:
    """
    Calculate padding for 1D or multi-dimensional convolution.

    Args:
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Defaults to 1.
        dilation: Dilation of the convolution. Defaults to 1.

    Returns:
        Calculated padding (integer for 1D, list for multi-dimensional).

    Raises:
        ValueError: If kernel_size is not an int, list, or tuple.

    Example:
        >>> get_padding(3)
        1
        >>> get_padding([3, 5], [1, 2], [1, 1])
        [1, 2]
    """
    if isinstance(kernel_size, int):
        if stride is None:
            stride = 1
        if dilation is None:
            dilation = 1
        return ((kernel_size - stride) * dilation) // 2

    elif isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
        if stride is None:
            stride = [1] * len(kernel_size)
        if dilation is None:
            dilation = [1] * len(kernel_size)
        return [get_padding(kernel_size[i], stride[i], dilation[i]) for i in range(len(kernel_size))]

    else:
        raise ValueError(f'Invalid kernel size {kernel_size}')


def select_activation(activation_name: str | None) -> nn.Module:
    """
    Select and return an activation function based on the given name.

    Args:
        activation_name: Name of the activation function.

    Returns:
        An instance of the specified activation function.

    Raises:
        NotImplementedError: If the specified activation is not implemented.

    Example:
        >>> relu = select_activation('relu')
        >>> isinstance(relu, nn.ReLU)
        True
    """
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'leaky_relu':
        return nn.LeakyReLU(LEAKY_RELU_SLOPE)
    elif activation_name == 'gelu':
        return nn.GELU()
    elif activation_name == 'tanh':
        return nn.Tanh()
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'softmax':
        return nn.Softmax(dim=-1)
    elif activation_name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f'Activation {activation_name} not implemented')


class TTSModule(nn.Module):
    """
    Base class for Text-to-Speech modules with configuration management.
    """
    def __init__(self, module_config: dict, global_config: dict):
        """
        Initialize the TTSModule.

        Args:
            module_config: Configuration specific to this module.
            global_config: Global configuration applicable to all modules.
        """
        super(TTSModule, self).__init__()

        self.module_config = self.optional_params.copy() if hasattr(self, 'optional_params') else {}
        self.module_config.update(global_config)
        self.module_config.update(module_config)

        self.module_name = self.__class__.__name__
        self._check_config_keys()

    def _check_config_keys(self) -> None:
        """
        Check if all mandatory configuration keys are present.

        Raises:
            ValueError: If any mandatory keys are missing from the configuration.
        """
        if self.optional_params is not None:
            for key, value in self.optional_params.items():
                self.module_config.setdefault(key, value)
                self.mandatory_keys.remove(key) if key in self.mandatory_keys else None
        self.mandatory_keys = list(set(self.mandatory_keys))
        missing_keys = [key for key in self.mandatory_keys if key not in self.module_config]
        if len(missing_keys) > 0:
            raise ValueError(f'Missing keys {missing_keys} in the module {self.module_name} configuration.')

    def update_keys(self, new_mandatory_keys: list[str] | None = None,
                    new_optional_params: dict | None = None) -> None:
        """
        Update the mandatory keys and optional parameters for the module.

        Args:
            new_mandatory_keys: New mandatory keys to add.
            new_optional_params: New optional parameters to add or update.
        """
        if not hasattr(self, 'mandatory_keys'):
            if new_mandatory_keys is not None:
                self.mandatory_keys = new_mandatory_keys
            else:
                self.mandatory_keys = []
        else:
            if new_mandatory_keys is not None:
                self.mandatory_keys.extend(new_mandatory_keys)

        if not hasattr(self, 'optional_params'):
            if new_optional_params is not None:
                self.optional_params = new_optional_params
            else:
                self.optional_params = {}
        else:
            if new_optional_params is not None:
                for key, value in new_optional_params.items():
                    self.optional_params.setdefault(key, value)

    @staticmethod
    def sample_config() -> dict:
        """
        Provide a sample configuration for the module.

        Returns:
            A dictionary containing a sample configuration.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError('Sample config method not implemented for this module.')

    def remove_weight_norm(self):
        """
        Remove weight normalization from all modules in the module recursively.

        Note:
            This function modifies the module in-place.
        """
        stack = [self]
        predefined_types = [nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding]
        while stack:
            children = stack.pop()
            for name, module in children.named_children():
                if isinstance(module, tuple(predefined_types)):
                    try:
                        torch.nn.utils.parametrize.remove_parametrizations(module, 'weight')
                        print(f"Weight norm removed from {name}")
                    except ValueError:
                        # Catch the error if weight_norm is not applied on this module
                        pass
                else:
                    stack.extend(list(module.children()))
