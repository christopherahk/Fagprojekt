#include "Activation_Sofmax_CategoricalCrossEntropy.h"

float Activation_Sofmax_CategoricalCrossEntropy::forward(const Tensor& yPred, const Tensor& yTrue) {
  activation.forward(yPred);

  return loss.calculate(activation.output, yTrue);
}

void Activation_Sofmax_CategoricalCrossEntropy::backward(const Tensor& yTrue) {
  dInputs = activation.output;

  for (int i = 0; i < yTrue.rows; i++) {
    for (int j = 0; j < yTrue.cols; j++) {
      dInputs(i, j) -= yTrue(i, j);
    }
  }

  auto normalizeFunc = [=](const float& x) -> float { return x / yTrue.rows; };
  dInputs = dInputs.applyElementWise(normalizeFunc);
}