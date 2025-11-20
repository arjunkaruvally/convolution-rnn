# CUDA operator implementation of Convolution RNN

An example of writing a C++/CUDA extension for PyTorch. See
[here](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html) for the accompanying tutorial on CUDA operator. 
This repo is a fork of [https://github.com/pytorch/extension-cpp](https://github.com/pytorch/extension-cpp) which is the
repo containing the official pytorch custom operator.

The repo contains a CUDA kernel implementation of convolution RNN. The operator has the following signature

```
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
y : Full hidden state trajectory (b l+1 n)

y[:, 0, :] contains the first hidden state that was passed. 
The hidden state trajectory after each update of convolution RNN is then
stored in the other l indices.
```

## Installation

The examples in this repo work with PyTorch 2.4+.

To build:
```
pip install --no-build-isolation -e .
```

To test:
```
pytest test/
```

## Usage

Details on what the operator does is in the reference implementation `reference_convrnn` in `extension_cpp/ops.py`. 
The parameters that need to be passed are the input, kernel and initial hidden state in the given format.

## Authors

[Arjun Karuvally](https://github.com/arjunkaruvally)
