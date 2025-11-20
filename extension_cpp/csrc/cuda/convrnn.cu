#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

using namespace torch::indexing;

namespace extension_cpp {

std::tuple<at::Tensor, at::Tensor> convrnn_forward_cuda(const at::Tensor& x, 
                                                        const at::Tensor& kernel, 
                                                        const at::Tensor& hidden) {
  namespace F = torch::nn::functional;

  // auto result = torch::zeros_like(x);
  auto b = x.size(0);
  auto l = x.size(1);
  auto n = x.size(2);
  auto k = kernel.size(2);
  auto y = torch::zeros({ b, l+1, n }, x.options());
  auto y_inter = torch::zeros_like(x);
  auto hidden_clone = hidden.clone();
  int padding = k / 2;
  
  // add the hidden state as the first y
  y.index_put_({Slice(), Slice(0, 1, 1), Slice()}, hidden_clone.clone());

  for (int i=0; i<l; i++){
    hidden_clone = at::convolution(hidden_clone, kernel, std::nullopt, {1}, {padding}, {1}, false, {0}, 1);
    y_inter.index_put_({Slice(), Slice(i, i+1, 1), Slice()}, hidden_clone.clone());

    hidden_clone = at::tanh(hidden_clone + x.index({Slice(), Slice(i, i+1), Slice()}));
    y.index_put_({Slice(), Slice(i+1, i+2, 1), Slice()}, hidden_clone.clone());
    
  }

  return {y, y_inter};
}


std::tuple<at::Tensor,at::Tensor,at::Tensor> convrnn_backward_cuda(
                                const at::Tensor& grad_y,
                                const at::Tensor& x, 
                                const at::Tensor& kernel, 
                                const at::Tensor& hidden,
                                const at::Tensor& y,
                                const at::Tensor& y_inter) {
  auto b = x.size(0);
  auto l = x.size(1);
  auto n = x.size(2);
  auto k = kernel.size(2);
  auto grad_x = torch::zeros_like(x);
  auto grad_kernel = torch::zeros_like(kernel);
  auto grad_h_inter = torch::zeros({ b, 1, n }, x.options());
  int padding = k / 2;

  std::array<bool, 3> output_mask = { true, true, true };

  // for (int i=0; i<l; i++){
  //   hidden_clone = at::convolution(hidden_clone, kernel, std::nullopt, {1}, {padding}, {1}, false, {0}, 1);
  //   y_inter.index_put_({Slice(), Slice(i, i+1, 1), Slice()}, hidden_clone.clone());

  //   hidden_clone = at::tanh(hidden_clone + x.index({Slice(), Slice(i, i+1), Slice()}));
  //   y.index_put_({Slice(), Slice(i+1, i+2, 1), Slice()}, hidden_clone.clone());
    
  // }

  // Write backward pass here
  for (int i=l-1; i>=0; i--){
    grad_h_inter = (grad_y.index({Slice(), Slice(i+1, i+2, 1), Slice()}) + grad_h_inter ) * 
                  (1 - at::square(at::tanh(y_inter.index({Slice(), Slice(i, i+1, 1), Slice()}) 
                              + x.index({Slice(), Slice(i, i+1), Slice()})))) ;
      
    grad_x.index_put_({Slice(), Slice(i, i+1), Slice()}, 
                      grad_x.index({Slice(), Slice(i, i+1), Slice()}) + grad_h_inter);

    auto grad_conv_backward = at::convolution_backward(grad_h_inter, // grad_output
                                                       y.index({Slice(), Slice(i, i+1, 1), Slice()}), // input
                                                       kernel, // weight
                                                       std::nullopt, // bias
                                                       {1}, // stride
                                                       {padding}, // padding
                                                       {1}, // dilation
                                                       false, // transposed
                                                       {0}, // output_padding
                                                       1, // groups
                                                       output_mask // output_mask?  
                                                      );
    
    grad_h_inter = std::get<0>(grad_conv_backward);
    grad_kernel = grad_kernel + std::get<1>(grad_conv_backward);
  }

  return { grad_x, grad_kernel, grad_h_inter };
}

// Registers CUDA implementations for convrnn_forward, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("convrnn_forward", &convrnn_forward_cuda);
  m.impl("convrnn_backward", &convrnn_backward_cuda);
}

}
