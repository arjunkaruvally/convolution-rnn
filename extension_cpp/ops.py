import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

__all__ = ["convrnn_forward", "reference_convrnn", "convrnn_interface"]


def convrnn_interface(x: Tensor, kernel: Tensor, hidden: float):
    return convrnn_forward(x, kernel, hidden)[0]


def convrnn_forward(x: Tensor, kernel: Tensor, hidden: float) -> Tuple[Tensor, Tensor]:
    """Computes the output of conovlutional RNN"""
    return torch.ops.extension_cpp.convrnn_forward.default(x, kernel, hidden)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("extension_cpp::convrnn_forward")
def _(x, kernel, hidden):
    return torch.empty_like(x), torch.empty_like(kernel), torch.empty_like(hidden)


def _backward(ctx, grad_y, grad_y_inter):
    saved_tensors = ctx.saved_tensors
    inputs = saved_tensors[:3]
    y = saved_tensors[3]
    y_inter = saved_tensors[4]
    grad_x, grad_kernel, grad_hidden = torch.ops.extension_cpp.convrnn_backward.default(grad_y, *inputs, y, y_inter)
    return grad_x, grad_kernel, grad_hidden


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs, *output)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "extension_cpp::convrnn_forward", _backward, setup_context=_setup_context)


@torch.library.register_fake("extension_cpp::convrnn_backward")
def _(x, kernel, hidden):
    return torch.empty_like(x)

########################################## Reference Implementations

def reference_convrnn(x: Tensor, kernel: Tensor, hidden:Tensor) -> Tensor:
    """
    Convolution RNN implementation for reference. 

    b - batch size
    k - kernel size
    n - hidden dimension
    l - sequence length
    
    Parameters
    ----------
    x : Tensor (b l n)
    kernel : Tensor (1, 1, k)
    hidden : Tensor (b, 1, n)

    Returns
    -------
    y : Full hidden state trajectory (b l n)
    """

    b, l, n = x.shape
    y = torch.zeros((b, l+1, n), device=x.device)
    y[:, :1, :] = hidden

    hidden_clone = hidden.clone()
    k = kernel.size(2)
    pad_left = k // 2
    pad_right = k - pad_left - 1
    for i in range(l):
        # Circular padding: wrap values from the opposite end
        hidden_padded = F.pad(hidden_clone, (pad_left, pad_right), mode='circular')
        hidden_clone = torch.tanh(F.conv1d(hidden_padded, kernel, bias=None, padding=0) 
                                + x[:, i:i+1, :])
        y[:, i+1:i+2, :] = hidden_clone.clone()

    return y
