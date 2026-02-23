#include "Tensor.h"
#include <algorithm>
#include <Arduino.h>
#include <string>

Tensor::Tensor(const float* data_, int rows_, int cols_)
  : rows(rows_), cols(cols_) {
  // Initializes the data with the specified size and fills the values from the input data.
  int size = rows * cols;
  data = new float[size];
  fill(data_);
}

Tensor::Tensor(int rows_, int cols_)
  : rows(rows_), cols(cols_) {
  /* Constructs a tensor without setting the data values.
  Used for when we just need an empty tensor we will fill later. */
  int size = rows * cols;
  data = new float[size];
}

Tensor::~Tensor() {
  // Destructor - for freeing up memory allocated for the data.
  delete[] data;
}

void Tensor::fill(const float* values) {
  // Fills the values by making a copy.
  if (values != nullptr) {
    std::copy(values, values + (rows * cols), data);
  }
}

void Tensor::printTensor() {
  // Prints the tensor row by row.
  for (int r = 0; r < rows; r++) {
    std::string row = "[";
    for (int c = 0; c < cols; c++) {
      row += std::to_string(this->operator()(r, c));

      if (c != cols - 1) {
        row += ", ";
      } else {
        row += "]";
      }
    }
    Serial.println(row.c_str());
  }
}

float& Tensor::operator()(int row, int col) {
  // Since the data is flattened, this is how we access each index.
  return data[row * cols + col];
}

Tensor Tensor::operator+(const Tensor& B) {
  // Tensor addition. First, an empty tensor is constructed.
  Tensor C(rows, cols);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      int index = r * cols + c;
      // Row by row, column by column, add the indices in the tensors.
      C.data[index] = data[index] + B.data[index];
    }
  }

  return C;
}

Tensor& Tensor::operator=(const Tensor& B) {
  // Assigment operator for setting tensors to each other.
  data = B.data;
  rows = B.rows;
  cols = B.cols;

  return *this;
}

Tensor Tensor::matmul(const Tensor& B) {
  /* Matrix multiplication. Constructs an empty tensor, makes the basic
  algorithm for multiplication, then returns the resulting tensor. */
  Tensor C(rows, B.cols);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < B.cols; j++) {
      float sum = 0.0f;

      for (int k = 0; k < cols; k++) {
        sum += data[i * cols + k] * B.data[k * B.cols + j];
      }

      C.data[i * B.cols + j] = sum;
    }
  }

  return C;
}

Tensor Tensor::transpose() {
  // Transposes the tensor by swapping rows and columns.
  Tensor B(cols, rows);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      B(c, r) = this->operator()(r, c);
    }
  }

  return B;
}