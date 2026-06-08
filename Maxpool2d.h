#pragma once
#include "Tensor.h"

struct MaxPool2D {
  Tensor output;
  Tensor dInputs;

  int inChannels;
  int inH, inW;
  int poolH, poolW;
  int outH, outW;

  int *maxIndices;

  MaxPool2D(int inChannels, int inH, int inW, int poolH, int poolW);
  ~MaxPool2D();

  void forward(const Tensor &x);
  void backward(const Tensor &dValues);

  MaxPool2D(const MaxPool2D &) = delete;
  MaxPool2D &operator=(const MaxPool2D &) = delete;
};
