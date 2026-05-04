#include "DenseLayer.h"
#include "Tensor.h"
#include <Arduino.h>
#include <math.h>

namespace {
const float kGradClipLimit = 100.0f;
}

static float clampValue(float x, float limit) {
  if (x > limit) {
    return limit;
  }
  if (x < -limit) {
    return -limit;
  }
  return x;
}

static void ensureTensorShape(Tensor &t, int rows, int cols) {
  if (t.rowCount != rows || t.colCount != cols) {
    t = Tensor(rows, cols);
  } else {
    memset(t.data, 0, t.size * sizeof(float));
  }
}

DenseLayer::DenseLayer(int inputCount, int neuronCount)
    : weights(inputCount, neuronCount), biases(1, neuronCount), inputs(nullptr),
      output(), dWeights(), dBiases(), dInputs() {
  for (int i = 0; i < inputCount * neuronCount; i++) {
    weights.data[i] = static_cast<float>(random(-100, 101)) / 1000.0f;
  }
  for (int c = 0; c < neuronCount; c++) {
    biases.data[c] = 0.0f;
  }
}

void DenseLayer::forward(const Tensor &x) {
  inputs = &x;
  ensureTensorShape(output, x.rowCount, weights.colCount);

  for (int r = 0; r < x.rowCount; r++) {
    for (int k = 0; k < x.colCount; k++) {
      float a = x(r, k);
      for (int c = 0; c < weights.colCount; c++) {
        output(r, c) += a * weights(k, c);
      }
    }
  }

  for (int r = 0; r < output.rowCount; r++) {
    for (int c = 0; c < output.colCount; c++) {
      output(r, c) += biases(0, c);
    }
  }
}

void DenseLayer::backward(const Tensor &dValues, bool computeDInputs) {
  if (inputs == nullptr)
    return;

  ensureTensorShape(dWeights, inputs->colCount, dValues.colCount);
  ensureTensorShape(dBiases, 1, dValues.colCount);

  for (int r = 0; r < inputs->colCount; r++) {
    for (int c = 0; c < dValues.colCount; c++) {
      float sum = 0.0f;
      for (int k = 0; k < inputs->rowCount; k++) {
        sum += (*inputs)(k, r) * dValues(k, c);
      }
      dWeights(r, c) = sum;
    }
  }

  if (computeDInputs) {
    ensureTensorShape(dInputs, dValues.rowCount, weights.rowCount);
    for (int r = 0; r < dValues.rowCount; r++) {
      for (int c = 0; c < weights.rowCount; c++) {
        float sum = 0.0f;
        for (int k = 0; k < dValues.colCount; k++) {
          sum += dValues(r, k) * weights(c, k);
        }
        dInputs(r, c) = sum;
      }
    }
  }

  for (int c = 0; c < dValues.colCount; c++) {
    float sum = 0.0f;
    for (int r = 0; r < dValues.rowCount; r++) {
      sum += dValues(r, c);
    }
    dBiases(0, c) = sum;
  }
}

void DenseLayer::update(float learningRate) {
  for (int i = 0; i < weights.size; i++) {
    float grad = isfinite(weights.data[i])
                     ? clampValue(dWeights.data[i], kGradClipLimit)
                     : 0.0f;
    weights.data[i] -= learningRate * grad;
  }

  for (int c = 0; c < biases.colCount; c++) {
    float grad = isfinite(biases.data[c])
                     ? clampValue(dBiases.data[c], kGradClipLimit)
                     : 0.0f;
    biases.data[c] -= learningRate * grad;
  }
}
