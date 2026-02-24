#include "SGD.h"
#include "NNLayer.h"

SGD::SGD(float lr = 0.001)
  : learningRate(lr) {}

void SGD::updateParameters(NNLayer layer) {
  for (int i = 0; i < layer.nInputs; i++) {
    for (int j = 0; j < layer.nNeurons; j++) {
      layer.weights(i, j) -= layer.dWeights(i, j) * learningRate;
    }
  }

  for (int k = 0; k < layer.nNeurons; k++) {
    layer.biases(0, k) -= layer.dBiases(0, k) * learningRate;
  }
}