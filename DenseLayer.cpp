#include "DenseLayer.h"
#include "Tensor.h"
#include <Arduino.h>
#include <math.h>

namespace {
const float kGradClipLimit = 100.0f;
}

float clampValue(float x,
                 float limit) { // Makes sure x is between -limit and limit to
                                // prevent exploding gradients.
  if (x > limit) {
    return limit;
  }
  if (x < -limit) {
    return -limit;
  }
  return x;
}

void ensureTensorShape(Tensor &t, int rows,
                       int cols) { // Ensures the tensor has the specified
                                   // shape, reallocating if necessary.
  if (t.rows != rows || t.cols != cols) { // if the shape is different
    t = Tensor(rows, cols);               // reallocate to the new shape
  }
}

DenseLayer::DenseLayer(int inputCount, int neuronCount) //
    : weights(inputCount, neuronCount), biases(1, neuronCount), inputs(nullptr),
      output(), dWeights(), dBiases(), dInputs() {
  // Random small initialization to break symmetry.
  for (int r = 0; r < inputCount; r++) {
    for (int c = 0; c < neuronCount; c++) {
      long rnd = random(-100, 101);
      weights(r, c) = static_cast<float>(rnd) / 1000.0f;
    }
  }

  for (int c = 0; c < neuronCount; c++) {
    biases(0, c) = 0.0f;
  }
}

void DenseLayer::forward(const Tensor &x) {
  inputs = &x;
  ensureTensorShape(output, x.rows, weights.cols);

  for (int r = 0; r < x.rows; r++) {
    for (int c = 0; c < weights.cols; c++) {
      float sum = 0.0f;
      for (int k = 0; k < x.cols; k++) {
        sum += x(r, k) * weights(k, c);
      }
      output(r, c) = sum + biases(0, c);
    }
  }
}

void DenseLayer::backward(const Tensor &dValues, bool computeDInputs) {
  if (inputs == nullptr) {
    // If backward is called before forward,
    // we can't compute gradients, so we just return.
    return;
  }

  ensureTensorShape(dWeights, inputs->cols,
                    dValues.cols); // Ensure dWeights has the correct shape
                                   // (input features x neurons).
  ensureTensorShape(
      dBiases, 1,
      dValues.cols); // Ensure dBiases has the correct shape (1 x neurons).

  for (int r = 0; r < inputs->cols; r++) {
    // Compute dWeights by multiplying inputs^T with dValues.
    for (int c = 0; c < dValues.cols; c++) {
      float sum = 0.0f;
      for (int k = 0; k < inputs->rows; k++) {
        sum += (*inputs)(k, r) * dValues(k, c);
      }
      dWeights(r, c) = sum;
    }
  }

  if (computeDInputs) { // Compute dInputs by multiplying dValues with
                        // weights^T.
    ensureTensorShape(dInputs, dValues.rows, weights.rows);

    for (int r = 0; r < dValues.rows; r++) {
      for (int c = 0; c < weights.rows; c++) {
        float sum = 0.0f;
        for (int k = 0; k < dValues.cols; k++) {
          sum += dValues(r, k) * weights(c, k);
        }
        dInputs(r, c) = sum;
      }
    }
  }

  for (int c = 0; c < dValues.cols; c++) { // Compute dBiases by summing dValues
                                           // across the batch for each neuron.
    float sum = 0.0f;
    for (int r = 0; r < dValues.rows; r++) {
      sum += dValues(r, c);
    }
    dBiases(0, c) = sum;
  }
}

void DenseLayer::update(
    float learningRate) { // Update weights and biases using the
                          // computed gradients, applying gradient
                          // clipping to prevent exploding gradients.
  for (int r = 0; r < weights.rows; r++) {
    for (int c = 0; c < weights.cols; c++) {
      float grad = dWeights(r, c);
      if (!isfinite(grad)) {
        grad = 0.0f;
      }
      grad = clampValue(grad, kGradClipLimit);
      weights(r, c) -= learningRate * grad;
    }
  }

  for (int c = 0; c < biases.cols;
       c++) { // Update biases with gradient clipping.
    float grad = dBiases(0, c);
    if (!isfinite(grad)) {
      grad = 0.0f;
    }
    grad = clampValue(grad, kGradClipLimit);
    biases(0, c) -= learningRate * grad;
  }
};
