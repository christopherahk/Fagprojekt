#pragma once
#include "Tensor.h"

struct ReLU {
  Tensor output;
  Tensor dInputs;
  Tensor inputs;

  void forward(const Tensor &inputs);
  void backward(const Tensor &dValues);
};
