#pragma once
#include "Tensor.h"

struct LeakyReLU {
  Tensor output;
  Tensor dInputs;
  Tensor inputs;
  float alpha;

  void forward(const Tensor& inputs, const float alpha = 0.01);
  void backward(const Tensor& dValues);
};