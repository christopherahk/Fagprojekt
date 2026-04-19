#pragma once
#include "Activation_Softmax_CategoricalCrossEntropy.h"
#include "NNLayer.h"
#include "ReLU.h"
#include "SGD.h"

struct Model {
  NNLayer layer1;
  NNLayer layer2;
  ReLU relu;
  Activation_Sofmax_CategoricalCrossEntropy lossActivation;

  Model(int nInputs, int nHidden, int batchSize, int nClasses);

  Tensor forward(const Tensor &x);
  float loss(const Tensor &y);
  void backward(const Tensor &y);
};
