#include "Activations.h"
#include <cmath>

void ReLU::forward(const Tensor &inputs_) {
  inputs = inputs_;

  auto relu = [](const float &x) -> float { return fmax(0.0f, x); };
  output = inputs_.applyElementWise(relu);
}

void ReLU::backward(const Tensor &dValues) {
  dInputs = dValues;

  for (int i = 0; i < dValues.rows; i++) {
    for (int j = 0; j < dValues.cols; j++) {
      if (inputs(i, j) <= 0) {
        dInputs(i, j) = 0;
      }
    }
  }
}

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

void Softmax::forward(const Tensor &inputs) {
  Tensor maxValues = inputs.applyMax();

  auto negativeFunc = [](const float &x) -> float { return -x; };
  Tensor negativeMax = maxValues.applyElementWise(negativeFunc);

  Tensor shifted = inputs.addMatAndVec(negativeMax);

  auto expFunc = [](const float &x) -> float { return exp(x); };
  Tensor expValues = shifted.applyElementWise(expFunc);

  Tensor sumExp = expValues.sumRows();

  output = expValues.matDivVecRows(sumExp);
}

void LeakyReLU::forward(const Tensor &inputs_, const float alpha_) {
  inputs = inputs_;
  alpha = alpha_;

  for (int r = 0; r < inputs_.rows; r++) {
    for (int c = 0; c < inputs_.cols; c++) {
      float x = inputs_(r, c);
      output(r, c) = (x < 0.0f) ? alpha_ * x : x;
    }
  }
}

void LeakyReLU::backward(const Tensor &dValues) {
  dInputs = dValues;

  for (int r = 0; r < dValues.rows; r++) {
    for (int c = 0; c < dValues.cols; c++) {
      if (inputs(r, c) < 0) {
        dInputs(r, c) = alpha;
      }
    }
  }
}
