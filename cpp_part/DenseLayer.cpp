#include "DenseLayer.h"
#include "Tensor.h"
#include <Arduino.h>
#include <math.h>

namespace {
const float kGradClipLimit = 100.0f;
const float kAdamBeta1 = 0.9f;
const float kAdamBeta2 = 0.999f;
const float kAdamEps = 1e-8f;
} // namespace

static float clampValue(float x, float limit) {
  if (x > limit)
    return limit;
  if (x < -limit)
    return -limit;
  return x;
}

static void ensureTensorShape(Tensor &t, int rows, int cols) {
  if (t.rowCount != rows || t.colCount != cols) {
    t = Tensor(rows, cols);
  } else {
    memset(t.data, 0, t.size * sizeof(float));
  }
}

static void ensureTensorShapeZero(Tensor &t, int rows, int cols) {
  if (t.rowCount != rows || t.colCount != cols) {
    t = Tensor(rows, cols);
    memset(t.data, 0, t.size * sizeof(float));
  }
}

DenseLayer::DenseLayer(int inputCount, int neuronCount)
    : weights(inputCount, neuronCount), biases(1, neuronCount), inputs(nullptr),
      output(), dWeights(), dBiases(), dInputs(), mWeights(), vWeights(),
      mBiases(), vBiases(), adamT(0) {
  float heStd = sqrtf(2.0f / inputCount);
  for (int i = 0; i < inputCount * neuronCount; i++) {
    float u1 = (random(1, 100001)) / 100000.0f;
    float u2 = (random(1, 100001)) / 100000.0f;
    float randn = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    weights.data[i] = randn * heStd;
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

  if (dWeights.rowCount != inputs->colCount ||
      dWeights.colCount != dValues.colCount) {
    dWeights = Tensor(inputs->colCount, dValues.colCount);
    memset(dWeights.data, 0, dWeights.size * sizeof(float));
  }
  if (dBiases.rowCount != 1 || dBiases.colCount != dValues.colCount) {
    dBiases = Tensor(1, dValues.colCount);
    memset(dBiases.data, 0, dBiases.size * sizeof(float));
  }

  for (int r = 0; r < inputs->colCount; r++) {
    for (int c = 0; c < dValues.colCount; c++) {
      float sum = 0.0f;
      for (int k = 0; k < inputs->rowCount; k++) {
        sum += (*inputs)(k, r) * dValues(k, c);
      }
      dWeights(r, c) += sum;
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
    dBiases(0, c) += sum;
  }
}

void DenseLayer::update(float learningRate) {
  ensureTensorShapeZero(mWeights, weights.rowCount, weights.colCount);
  ensureTensorShapeZero(vWeights, weights.rowCount, weights.colCount);
  ensureTensorShapeZero(mBiases, 1, biases.colCount);
  ensureTensorShapeZero(vBiases, 1, biases.colCount);

  adamT++;
  float bc1 = 1.0f - powf(kAdamBeta1, (float)adamT);
  float bc2 = 1.0f - powf(kAdamBeta2, (float)adamT);

  for (int i = 0; i < weights.size; i++) {
    float g = isfinite(dWeights.data[i])
                  ? clampValue(dWeights.data[i], kGradClipLimit)
                  : 0.0f;
    mWeights.data[i] = kAdamBeta1 * mWeights.data[i] + (1.0f - kAdamBeta1) * g;
    vWeights.data[i] =
        kAdamBeta2 * vWeights.data[i] + (1.0f - kAdamBeta2) * g * g;
    float mHat = mWeights.data[i] / bc1;
    float vHat = vWeights.data[i] / bc2;
    weights.data[i] -= learningRate * mHat / (sqrtf(vHat) + kAdamEps);
  }

  for (int c = 0; c < biases.colCount; c++) {
    float g = isfinite(dBiases.data[c])
                  ? clampValue(dBiases.data[c], kGradClipLimit)
                  : 0.0f;
    mBiases.data[c] = kAdamBeta1 * mBiases.data[c] + (1.0f - kAdamBeta1) * g;
    vBiases.data[c] =
        kAdamBeta2 * vBiases.data[c] + (1.0f - kAdamBeta2) * g * g;
    float mHat = mBiases.data[c] / bc1;
    float vHat = vBiases.data[c] / bc2;
    biases.data[c] -= learningRate * mHat / (sqrtf(vHat) + kAdamEps);
  }

  memset(dWeights.data, 0, dWeights.size * sizeof(float));
  memset(dBiases.data, 0, dBiases.size * sizeof(float));
}
