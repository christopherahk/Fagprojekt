#include "Tensor.h"
#include <Arduino.h>
#include <algorithm>
#include <string>

Tensor::Tensor(const float *data_, int rows_, int cols_)
  : rows(rows_), cols(cols_), size(rows_ * cols_) {
  // Initializes the data with the specified size and fills the values from the
  // input data.
  data = new float[size];
  fill(data_);
}

Tensor::Tensor(int rows_, int cols_)
: rows(rows_), cols(cols_), size(rows_ * cols_) {
  /* Constructs a tensor without setting the data values.
  Used for when we just need an empty tensor we will fill later. */
  data = new float[size];
}

Tensor::Tensor(const Tensor &other)
  : rows(other.rows), cols(other.cols), size(other.rows * other.cols) {
  // Copy constructor - deep copies the data array.
  data = new float[size];
  std::copy(other.data, other.data + size, data);
}

Tensor::Tensor()
  : rows(0), cols(0), size(0), data(nullptr) {}

Tensor::~Tensor() {
  // Destructor - for freeing up memory allocated for the data.
  delete[] data;
}

void Tensor::fill(const float *values) {
  // Fills the values by making a copy.
  if (values != nullptr) {
    std::copy(values, values + size, data);
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

float &Tensor::operator()(int row, int col) {
  // Since the data is flattened, this is how we access each index.
  return data[row * cols + col];
}

float Tensor::operator()(int row, int col) const {
  // Since the data is flattened, this is how we access each index.
  return data[row * cols + col];
}

Tensor Tensor::operator+(const Tensor &B) {
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

Tensor Tensor::operator-(const Tensor &B) {
  // Tensor subtraction. First, an empty tensor is constructed.
  Tensor C(rows, cols);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      int index = r * cols + c;
      // Row by row, column by column, subtracts the indices in the tensors.
      C.data[index] = data[index] - B.data[index];
    }
  }

  return C;
}

Tensor &Tensor::operator=(const Tensor &B) {
  // Assigment operator for setting tensors to each other.
  if (this == &B) {
    return *this;
  }

  delete[] data;

  rows = B.rows;
  cols = B.cols;
  size = B.size;

  data = new float[size];
  std::copy(B.data, B.data + size, data);

  return *this;
}

Tensor Tensor::matmul(const Tensor &B) const {
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

Tensor Tensor::transpose() const {
  // Transposes the tensor by swapping rows and columns.
  Tensor B(cols, rows);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      B(c, r) = this->operator()(r, c);
    }
  }

  return B;
}

Tensor Tensor::sumRows() const {
  Tensor C(1, cols);
  
  for (int c = 0; c < cols; c++) {
    float sum = 0.0f;

    for (int r = 0; r < rows; r++) {
      sum += this->operator()(r, c);
    }

    C(0, c) = sum;
  }

  return C;
}

float Tensor::mean() const {
  float result = 0.0f;

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      result += this->operator()(r, c);
    }
  }

  return result / size;
}

Tensor Tensor::clip(float minValue, float maxValue) const {
  Tensor C(rows, cols);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      float x = this->operator()(r, c);

      if (x < minValue) {
        x = minValue;
      }
      else if (x > maxValue) {
        x = maxValue;
      }

      C(r, c) = x;
    }
  }
  
  return C;
}

Tensor Tensor::selectTrueClass(const Tensor& B) const {
  Tensor C(rows, 1);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      if (B(r, c) == 1.0) {
        C(r, 0) = this->operator()(r, c);
        break;
      }
    }
  }

  return C;
}

int Tensor::argmaxRow(int row) const {
  int colMax = 0;
  float maxVal = this->operator()(row, 0);

  for (int c = 0; c < cols; c++) {
    if (this->operator()(row, c) > maxVal) {
      maxVal = this->operator()(row, c);
      colMax = c;
    }
  }

  return colMax;
}

Tensor Tensor::applyMax() const {
  Tensor C(rows, 1);

  for (int r = 0; r < rows; r++) {
    int max = this->argmaxRow(r);
    C(r, 0) = max;
  }
}

Tensor Tensor::addMatAndVec(const Tensor& vec) const {
  Tensor C(rows, cols);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      C(r, c) = this->operator()(r, c) + vec(r, 0);
    }
  }

  return C;
}

Tensor Tensor::matDivVecRows(const Tensor& vec) const {
  Tensor C(rows, cols);

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      C(r, c) = this->operator()(r, c) / vec(r, 0);
    }
  }

  return C;
}
