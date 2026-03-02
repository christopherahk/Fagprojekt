#pragma once
#include "Tensor.h"

struct ReLU {
  Tensor output;
  Tensor dInputs;
  Tensor inputs;

  void forward(const Tensor &inputs);
  void backward(const Tensor &dValues);
};

struct Sigmoid {
  Tensor output;
  Tensor dInputs;
  Tensor inputs;

  void forward(const Tensor &inputs);
  void backward(const Tensor &dValues);
};

struct Softmax {
  Tensor output;

  void forward(const Tensor &inputs);
};

struct LeakyReLU {
  Tensor output;
  Tensor dInputs;
  Tensor inputs;

  float alpha;

  void forward(const Tensor &inputs, const float alpha = 0.01);
  void backward(const Tensor &dValues);
};
