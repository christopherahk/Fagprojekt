#pragma once
#include "Tensor.h"

struct DenseLayer {
  Tensor weights;
  Tensor biases;
  const Tensor *inputs;
  Tensor output;
  Tensor dWeights;
  Tensor dBiases;
  Tensor dInputs;

  Tensor mWeights;
  Tensor vWeights;
  Tensor mBiases;
  Tensor vBiases;
  int adamT;

  DenseLayer(int inputCount, int neuronCount);

  void forward(const Tensor &x);
  void backward(const Tensor &dValues, bool computeDInputs = true);
  void update(float learningRate);
};
