import numpy as np
import torch

from torch.nn import functional as F


AVAILABLE_ACTIVATION_FUNCTION_OPTIONS = ['arctan_activation',
                                         'softsign_activation',
                                         'sigmoid_activation',
                                         'linear_activation',
                                         'inv_square_root_activation',
                                         'cdf_activation',
                                         'hardtanh_activation',
                                         'no_activation']

def getActivationFunctionByName(last_activation_name):
    if not last_activation_name in AVAILABLE_ACTIVATION_FUNCTION_OPTIONS:
        msg = f"ERROR! Last activation name '{last_activation_name}' is not known!"
        raise Exception(msg)
    return globals()[last_activation_name]

@torch.jit.script
def arctan_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return epsilon + (1 - 2 * epsilon) * (0.5 + torch.arctan(x)/torch.tensor(np.pi))

@torch.jit.script
def softsign_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return (0.5 - epsilon) * F.softsign(x) + 0.5

@torch.jit.script
def sigmoid_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return torch.sigmoid(x)

@torch.jit.script
def linear_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return epsilon + (1 - 2 * epsilon) * (x - x.min())/(x.max() - x.min())

@torch.jit.script
def inv_square_root_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return (0.5 - epsilon) * x * torch.rsqrt(1 + x ** 2) + 0.5

@torch.jit.script
def cdf_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    # https://github.com/IraKorshunova/pytorch/blob/master/torch/autograd/_functions/pointwise.py#L274
    # https://github.com/IraKorshunova/pytorch/blob/master/torch/lib/THC/THCNumerics.cuh#L441
    # https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g3b8115ff34a107f4608152fd943dbf81
    return (0.5 - epsilon) * torch.erf(x/torch.sqrt(torch.tensor(2))) + 0.5

@torch.jit.script
def hardtanh_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return F.hardtanh(x, epsilon, 1.0 - epsilon)

@torch.jit.script
def no_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return x
