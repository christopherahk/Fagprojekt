#pragma once
#include "Tensor.h"

float clampValue(float x, float limit);
void ensureTensorShape(Tensor &t, int rows, int cols);

struct DenseLayer { // i am dense
  Tensor weights;
  Tensor biases;
  const Tensor *inputs;
  Tensor output;
  Tensor dWeights;
  Tensor dBiases;
  Tensor dInputs;

  DenseLayer(int inputCount, int neuronCount);

  void forward(const Tensor &x) void backward(
      const Tensor &dValues,
      bool computeDInputs = true) void update(float learningRate)
};
