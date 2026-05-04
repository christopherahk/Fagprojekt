#include "NNLayer.h"
#include "Tensor.h"
#include <Arduino.h>
#include <cmath>

NNLayer::NNLayer(int inputs_, int neurons, int batchSize)
    : nInputs(inputs_), nNeurons(neurons), nBatchSize(batchSize),
      weights(inputs_, neurons), biases(1, neurons), output(batchSize, neurons),
      dWeights(inputs_, neurons), dBiases(1, neurons),
      dInputs(batchSize, inputs_), inputs(batchSize, inputs_) {
  initWeights();
  initBiases();
}

void NNLayer::initWeights() {
  float scale = 1.0f / sqrtf((float)nInputs);

  for (int i = 0; i < nInputs; i++) {
    for (int j = 0; j < nNeurons; j++) {
      float r = random(-1000, 1000) / 1000.f;
      weights(i, j) = r * scale;
    }
  }
}

void NNLayer::initBiases() {
  for (int i = 0; i < nNeurons; i++) {
    biases(0, i) = 0;
  }
}

void NNLayer::forward(const Tensor &inputs_) {
  inputs = inputs_;

  output = inputs.matmul(weights);

  for (int r = 0; r < output.rowCount; r++) {
    for (int c = 0; c < output.colCount; c++) {
      output(r, c) += biases(0, c);
    }
  }
}

void NNLayer::backward(const Tensor &dValues) {
  Tensor inputsT = inputs.transpose();
  dWeights = inputsT.matmul(dValues);

  Tensor weightsT = weights.transpose();
  Tensor dValuesCopy = dValues;
  dInputs = dValuesCopy.matmul(weightsT);

  dBiases = dValues.sumRows();
}
