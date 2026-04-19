#include "Model.h"
#include "Activation_Sofmax_CategoricalCrossEntropy.h"
#include "NNLayer.h"
#include "ReLU.h"
#include "SGD.h"

Model::Model(int nInputs, int nHidden, int batchSize, int nClasses)
    : layer1(nInputs, nHidden, batchSize), layer2(nHidden, nClasses, batchSize),
      relu(), lossActivation() {}

Tensor Model::forward(const Tensor &x) {
  layer1.forward(x);
  relu.forward(layer1.output);
  layer2.forward(relu.output);

  return layer2.output;
}

float Model::loss(const Tensor &y) {
  float loss = lossActivation.forward(layer2.output, y);
  return loss;
}

void Model::backward(const Tensor &y) {
  lossActivation.backward(y);
  layer2.backward(lossActivation.dInputs);
  relu.backward(layer2.dInputs);
  layer1.backward(relu.dInputs);
}
