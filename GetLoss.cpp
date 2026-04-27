#include "GetLoss.h"

float GetLoss::forward(const Tensor &yPred, const Tensor &yTrue) {
  activation.forward(yPred);
  return loss.calculate(activation.output, yTrue);
}

void GetLoss::backward(const Tensor &yTrue) {
  dInputs = activation.output;

  for (int i = 0; i < yTrue.rows; i++) {
    for (int j = 0; j < yTrue.cols; j++) {
      dInputs(i, j) = (dInputs(i, j) - yTrue(i, j)) / yTrue.rows;
    }
  }
}
