#pragma once
#include "Softmax.h"
#include "CategoricalCrossEntropyLoss.h"

struct Activation_Sofmax_CategoricalCrossEntropy {
  Softmax activation;
  CategoricalCrossEntropyLoss loss;

  Tensor dInputs;

  float forward(const Tensor& yPred, const Tensor& yTrue);
  void backward(const Tensor& yTrue);
};