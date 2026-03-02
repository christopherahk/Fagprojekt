#pragma once
#include "Tensor.h"

struct Sigmoid {
  Tensor output;
  Tensor dInputs;
  Tensor inputs;

  void forward(const Tensor &inputs);
  void backward(const Tensor &dValues);
};
