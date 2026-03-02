#include "Sigmoid.h"
#include <cmath>

void Sigmoid::forward(const Tensor &inputs_) {
  inputs = inputs_;

  auto sigmodFunc = [](const float &x) -> float {
    return 1.0f / (1.0f + exp(-x));
  };
  output = inputs_.applyElementWise(sigmodFunc);
}

void Sigmoid::backward(const Tensor &dValues) {
  dInputs = dValues;

  for (int r = 0; r < dValues.rows; r++) {
    for (int c = 0; c < dValues.cols; c++) {
      float sigmoid = 1.0f / (1.0f + exp(-1 * inputs(r, c)));
      dInputs(r, c) = sigmoid * (1 - sigmoid);
    }
  }
}
