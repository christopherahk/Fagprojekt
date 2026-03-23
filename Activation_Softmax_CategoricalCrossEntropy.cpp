#include "Activation_Softmax_CategoricalCrossEntropy.h"

float Activation_Softmax_CategoricalCrossEntropy::forward(const Tensor &yPred,
                                                          const Tensor &yTrue) {
  activation.forward(yPred);

  return loss.forward(activation.output, yTrue);
}

void Activation_Softmax_CategoricalCrossEntropy::backward(const Tensor &yTrue) {
  dInputs = activation.output;

  for (int i = 0; i < yTrue.rows; i++) {
    for (int j = 0; j < yTrue.cols; j++) {
      dInputs(i, j) -= yTrue(i, j);
    }
  }

  auto normalizeFunc = [=](const float &x) -> float { return x / yTrue.rows; };
  dInputs = dInputs.applyElementWise(normalizeFunc);
}
