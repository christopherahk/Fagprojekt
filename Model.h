#pragma once
#include "Activations.h"
#include "GetLoss.h"
#include "NNLayer.h"
#include "SGD.h"

struct Model {
  NNLayer layer1;
  NNLayer layer2;
  ReLU relu;
  GetLoss lossActivation;

  Model(int nInputs, int nHidden, int batchSize, int nClasses);

  Tensor forward(const Tensor &x);
  float loss(const Tensor &y);
  void backward(const Tensor &y);
};
