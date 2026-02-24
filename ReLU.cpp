#include "ReLU.h"
#include "Tensor.h"
#include <cmath>

void ReLU::forward(const Tensor& inputs_) {
  inputs = inputs_;

  auto relu = [](const float& x) -> float { return fmax(0.0f, x); };
  output = inputs_.applyElementWise(relu);
}

void ReLU::backward(const Tensor& dValues) {
  dInputs = dValues;

  for (int i = 0; i < dValues.rows; i++) {
    for (int j = 0; j < dValues.cols; j++) {
      if (inputs(i, j) <= 0) {
        dInputs(i, j) = 0;
      }
    }
  }
}