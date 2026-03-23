#include "Activations.h"
#include <math.h>

void ReLU::forward(const Tensor &inputs_) {
  inputs = inputs_;

  auto relu = [](const float &x) -> float { return fmaxf(0.0f, x); };
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
    return 1.0f / (1.0f + expf(-x));
  };
  output = inputs_.applyElementWise(sigmodFunc);
}

void Sigmoid::backward(const Tensor &dValues) {
  dInputs = dValues;

  for (int r = 0; r < dValues.rows; r++) {
    for (int c = 0; c < dValues.cols; c++) {
      float sigmoid = 1.0f / (1.0f + expf(-1 * inputs(r, c)));
      dInputs(r, c) = sigmoid * (1 - sigmoid);
    }
  }
}

void Softmax::forward(const Tensor &inputs) {
  output = Tensor(inputs.rows, inputs.cols);

  for (int r = 0; r < inputs.rows; r++) {
    float rowMax = inputs(r, 0);
    for (int c = 1; c < inputs.cols; c++) {
      if (inputs(r, c) > rowMax) {
        rowMax = inputs(r, c);
      }
    }

    float rowSum = 0.0f;
    for (int c = 0; c < inputs.cols; c++) {
      float e = expf(inputs(r, c) - rowMax);
      output(r, c) = e;
      rowSum += e;
    }

    if (rowSum < 1e-12f) {
      rowSum = 1e-12f;
    }

    for (int c = 0; c < inputs.cols; c++) {
      output(r, c) /= rowSum;
    }
  }
}

void LeakyReLU::forward(const Tensor &inputs_, const float alpha_) {
  inputs = inputs_;
  alpha = alpha_;
  output = Tensor(inputs_.rows, inputs_.cols);

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
        dInputs(r, c) = dValues(r, c) * alpha;
      }
    }
  }
}
