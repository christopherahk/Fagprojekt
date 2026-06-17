#include "Activations.h"
#include <math.h>

void ReLU::forward(Tensor &inputs_) {
  inputs = inputs_;
  for (int i = 0; i < inputs_.size; i++) {
    if (inputs_.data[i] < 0.0f) {
      inputs_.data[i] = 0.0f;
    }
  }
  output = inputs_;
}

void ReLU::backward(const Tensor &dValues) {
  dInputs = dValues;
  for (int i = 0; i < dValues.size; i++) {
    if (inputs.data[i] <= 0) {
      dInputs.data[i] = 0;
    }
  }
}

void Sigmoid::forward(Tensor &inputs_) {
  inputs = inputs_;
  auto sigmodFunc = [](float x) -> float { return 1.0f / (1.0f + expf(-x)); };
  inputs_.apply(sigmodFunc);
  output = inputs_;
}

void Sigmoid::backward(const Tensor &dValues) {
  dInputs = dValues;
  for (int r = 0; r < dValues.rowCount; r++) {
    for (int c = 0; c < dValues.colCount; c++) {
      float sigmoid = 1.0f / (1.0f + expf(-inputs(r, c)));
      dInputs(r, c) = dValues(r, c) * sigmoid * (1.0f - sigmoid);
    }
  }
}

void Softmax::forward(const Tensor &inputs) {
  output = Tensor(inputs.rowCount, inputs.colCount);
  for (int r = 0; r < inputs.rowCount; r++) {
    float rowMax = inputs(r, 0);
    for (int c = 1; c < inputs.colCount; c++) {
      if (inputs(r, c) > rowMax) {
        rowMax = inputs(r, c);
      }
    }
    float rowCountum = 0.0f;
    for (int c = 0; c < inputs.colCount; c++) {
      float e = expf(inputs(r, c) - rowMax);
      output(r, c) = e;
      rowCountum += e;
    }
    if (rowCountum < 1e-12f) {
      rowCountum = 1e-12f;
    }
    for (int c = 0; c < inputs.colCount; c++) {
      output(r, c) /= rowCountum;
    }
  }
}

void LeakyReLU::forward(const Tensor &inputs_, float alpha_) {
  inputs = inputs_;
  alpha = alpha_;
  output = inputs_;
  for (int i = 0; i < output.size; i++) {
    if (output.data[i] < 0.0f) {
      output.data[i] *= alpha_;
    }
  }
}

void LeakyReLU::backward(const Tensor &dValues) {
  dInputs = dValues;
  for (int i = 0; i < dValues.size; i++) {
    if (inputs.data[i] < 0.0f) {
      dInputs.data[i] *= alpha;
    }
  }
}
