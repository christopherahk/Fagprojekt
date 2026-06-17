#include "GetLoss.h"

float GetLoss::forward(const Tensor &yPred, const Tensor &yTrue) {
  activation.forward(yPred);
  return loss.calculate(activation.output, yTrue);
}

void GetLoss::backward(const Tensor &yTrue) {
  dInputs = activation.output;
  for (int i = 0; i < dInputs.size; i++) {
    dInputs.data[i] = (dInputs.data[i] - yTrue.data[i]) / yTrue.rowCount;
  }
}
