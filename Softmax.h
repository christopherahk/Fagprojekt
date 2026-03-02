#pragma once
#include "Tensor.h"

struct Softmax {
  Tensor output;

  void forward(const Tensor &inputs);
};
