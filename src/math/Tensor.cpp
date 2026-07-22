#include "Tensor.hpp"
#include <cstddef>
#include <initializer_list>
#include <memory>

Tensor::Tensor(std::initializer_list<size_t> shape) {
  size_t ndims = shape.size();
  shape_stride.dimensions = ndims;

  size_t i = 0;
  for (size_t j : shape) {
    shape_stride.shape[i] = j;
    i++;
  }

  size_t row_size = shape_stride.shape[ndims - 1] * sizeof(float);
  size_t padded_row_size = (row_size + 127) & ~127;

  size_t padded_row_stride = padded_row_size / sizeof(float);

  shape_stride.stride[ndims - 1] = 1;
  shape_stride.stride[ndims - 2] = padded_row_stride;

  for (size_t j = ndims - 2; j > 0; j--) {
    shape_stride.stride[j - 1] =
        shape_stride.stride[j] * shape_stride.shape[j - 1];
  }

  size_t total_tensor_size =
      shape_stride.stride[0] * shape_stride.shape[0] * sizeof(float);

  storage = std::make_shared<TensorStorage>(total_tensor_size);
}
