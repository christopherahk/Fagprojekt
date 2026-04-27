#pragma once
#include "Activations.h"
#include "CategoricalCrossEntropyLoss.h"

struct GetLoss {
  Softmax activation;
  CategoricalCrossEntropyLoss loss;

  Tensor dInputs;

  float forward(const Tensor &yPred, const Tensor &yTrue);
  void backward(const Tensor &yTrue);
};
