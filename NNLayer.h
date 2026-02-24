#pragma once
#include "Tensor.h"

struct NNLayer {
  int nInputs;
  int nNeurons;
  int nBatchSize;

  Tensor weights;
  Tensor biases;
  Tensor output;

  Tensor dWeights;
  Tensor dBiases;
  Tensor dInputs;

  Tensor inputs;

  NNLayer(int inputs_, int neurons, int batchSize);
    
  void initWeights();
  void initBiases();

  void forward(const Tensor& inputs_);
  void backward(const Tensor& dValues);
};