#pragma once

#include "Tensor.h"

// The linalg library currently only support 2D tensors.

// Flatten the tensor in order to use the linear algebra functions on it.
Tensor flatten(const Tensor& t, size_t start_dim = 1);

// Matrix multiplication of two flattened tensors.
Tensor matmul(const Tensor& A, const Tensor& B);