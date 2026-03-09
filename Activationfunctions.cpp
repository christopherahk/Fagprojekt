#include "Tensor.h"
#include <cmath>

// Sigmoid activationfunction and derivative

void sigmoid(Tensor &t) {
  for (int r = 0; r < t.rows; r++) {
    for (int c = 0; c < t.cols; c++) {
      float x = t(r, c);

      if (x >= 0.0f) {
        t(r, c) = 1.0f / (1.0f + std::exp(-x));
      } else {
        float z = std::exp(x);
        t(r, c) = z / (1.0f + z);
      }
    }
  }
}

void sigmoid_dev(Tensor &t) {
  for (int r = 0; r < t.rows; r++) {
    for (int c = 0; c < t.cols; c++) {
      t(r, c) = t(r, c) * (1.0f - t(r, c));
    }
  }
}

// Relu activationfunction and derivative
void relu(Tensor &t) {
  for (int r = 0; r < t.rows; r++) {
    for (int c = 0; c < t.cols; c++) {
      if (t(r, c) < 0.0f) {
        t(r, c) = 0.0f;
      }
    }
  }
}

void relu_dev(Tensor &t) {
  for (int r = 0; r < t.rows; r++) {
    for (int c = 0; c < t.cols; c++) {
      if (t(r, c) > 0.0f) {
        t(r, c) = 1.0f;
      } else {
        t(r, c) = 0.0f;
      }
    }
  }
}

// Leaky relu activationfunction and derivative
void leaky_relu(Tensor &t) {
  for (int r = 0; r < t.rows; r++) {
    for (int c = 0; c < t.cols; c++) {
      if (t(r, c) < 0.0f) {
        t(r, c) = 0.01f * t(r, c);
      }
    }
  }
}

void leaky_relu_dev(Tensor &t) {
  for (int r = 0; r < t.rows; r++) {
    for (int c = 0; c < t.cols; c++) {
      if (t(r, c) < 0.0f) {
        t(r, c) = 0.01f;
      } else {
        t(r, c) = 1.0f;
      }
    }
  }
}

// Softmax activationfunction
void softmax(fTensor &t) {
  for (int r = 0; r < t.rows; r++) {
    float sum = 0.0f;

    for (int c = 0; c < t.cols; c++) {
      t(r, c) = std::exp(t(r, c));
      sum += t(r, c);
    }

    for (int c = 0; c < t.cols; c++) {
      t(r, c) /= sum;
    }
  }
}
