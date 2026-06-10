#pragma once
#include "Tensor.h"

struct Conv2DLayer {
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

  int inH, inW;
  int outChannels;
  int kernelH, kernelW;
  int strideH, strideW;
  int padH, padW;
  int outH, outW;

  Conv2DLayer(int inH, int inW, int outChannels, int kernelH, int kernelW,
              int strideH, int strideW, int padH, int padW);

  void forward(const Tensor &x);
  void backward(const Tensor &dValues, bool computeDInputs = true);
  void update(float learningRate);
};
