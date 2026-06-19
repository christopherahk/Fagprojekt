#pragma once
#include "Activations.h"
#include "FocalLoss.h"

struct GetLoss {
  Softmax activation;
  FocalLoss loss;

  Tensor dInputs;

  GetLoss(float gamma = 2.0f) : loss(gamma) {}

  float forward(const Tensor &yPred, const Tensor &yTrue);
  void backward(const Tensor &yTrue);
};
