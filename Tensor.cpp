#include "Tensor.h"
#include <Arduino.h>
#include <algorithm>
#include <string>

Tensor::Tensor(const float *data_, int rows_, int cols_)
    : rows(rows_), cols(cols_), size(rows_ * cols_), owned(true) {
  data = new float[size];
  fill(data_);
}

Tensor::Tensor(int rows_, int cols_)
    : rows(rows_), cols(cols_), size(rows_ * cols_), owned(true) {
  data = new float[size]();
}

Tensor::Tensor(const Tensor &other)
    : rows(other.rows), cols(other.cols), size(other.rows * other.cols),
      owned(true) {
  data = new float[size];
  std::copy(other.data, other.data + size, data);
}

Tensor::Tensor() : rows(0), cols(0), size(0), data(nullptr), owned(false) {}

Tensor::Tensor(float *buffer, int rows_, int cols_)
    : rows(rows_), cols(cols_), size(rows_ * cols_), owned(false) {
  data = buffer;
}

Tensor::~Tensor() {
  if (owned)
    delete[] data;
}

void Tensor::fill(const float *values) {
  if (values != nullptr) {
    std::copy(values, values + size, data);
  }
}

void Tensor::printTensor() {
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

float &Tensor::operator()(int row, int col) { return data[row * cols + col]; }

float Tensor::operator()(int row, int col) const {
  return data[row * cols + col];
}

Tensor Tensor::operator+(const Tensor &B) {
  Tensor C(rows, cols);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      int index = r * cols + c;
      C.data[index] = data[index] + B.data[index];
    }
  }
  return C;
}

Tensor Tensor::operator-(const Tensor &B) {
  Tensor C(rows, cols);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      int index = r * cols + c;
      C.data[index] = data[index] - B.data[index];
    }
  }
  return C;
}

Tensor Tensor::operator*(float scalar) const {
  Tensor C(rows, cols);
  int n = rows * cols;
  for (int i = 0; i < n; i++) {
    C.data[i] = data[i] * scalar;
  }
  return C;
}

Tensor operator*(float scalar, const Tensor &A) { return A * scalar; }

void Tensor::swap(Tensor &B) {
  int tmpRows = rows;
  rows = B.rows;
  B.rows = tmpRows;

  int tmpCols = cols;
  cols = B.cols;
  B.cols = tmpCols;

  int tmpSize = size;
  size = B.size;
  B.size = tmpSize;

  float *tmpData = data;
  data = B.data;
  B.data = tmpData;

  bool tmpOwned = owned;
  owned = B.owned;
  B.owned = tmpOwned;
}

Tensor &Tensor::operator=(Tensor B) {
  swap(B);
  return *this;
}

void Tensor::copyFrom(const Tensor &other) {
  if (this->size == other.size) {
    std::copy(other.data, other.data + size, this->data);
  }
}

Tensor Tensor::matmul(const Tensor &B) const {
  if (cols != B.rows) {
    Serial.println("Error: Incompatible dimensions for matrix multiplication.");
    Serial.print("Tensor A dimensions: ");
    Serial.print(rows);
    Serial.print("x");
    Serial.println(cols);
    Serial.print("Tensor B dimensions: ");
    Serial.print(B.rows);
    Serial.print("x");
    Serial.println(B.cols);
    return Tensor(0, 0);
  }

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

void Tensor::matmulInto(const Tensor &B, Tensor &out) const {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < B.cols; j++) {
      float sum = 0.0f;
      for (int k = 0; k < cols; k++)
        sum += data[i * cols + k] * B.data[k * B.cols + j];
      out.data[i * B.cols + j] = sum;
    }
  }
}

void Tensor::matmulTInto(const Tensor &B, Tensor &out) const {
  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < B.cols; j++) {
      float sum = 0.0f;
      for (int k = 0; k < rows; k++)
        sum += data[k * cols + i] * B.data[k * B.cols + j];
      out.data[i * B.cols + j] = sum;
    }
  }
}

void Tensor::matmulBTInto(const Tensor &B, Tensor &out) const {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < B.rows; j++) {
      float sum = 0.0f;
      for (int k = 0; k < cols; k++)
        sum += data[i * cols + k] * B.data[j * B.cols + k];
      out.data[i * B.rows + j] = sum;
    }
  }
}

Tensor Tensor::transpose() const {
  Tensor B(cols, rows);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      B(c, r) = this->operator()(r, c);
    }
  }
  return B;
}

Tensor Tensor::sumRows() const {
  Tensor C(rows, 1);
  for (int r = 0; r < rows; r++) {
    float sum = 0.0f;
    for (int c = 0; c < cols; c++)
      sum += this->operator()(r, c);
    C(r, 0) = sum;
  }
  return C;
}

Tensor Tensor::sumCols() const {
  Tensor C(1, cols);
  for (int c = 0; c < cols; c++) {
    float sum = 0.0f;
    for (int r = 0; r < rows; r++)
      sum += this->operator()(r, c);
    C(0, c) = sum;
  }
  return C;
}

float Tensor::mean() const {
  float result = 0.0f;
  if (size == 0) {
    return result;
  }
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
      } else if (x > maxValue) {
        x = maxValue;
      }
      C(r, c) = x;
    }
  }
  return C;
}

Tensor Tensor::selectTrueClass(const Tensor &B) const {
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
    float maxVal = this->operator()(r, 0);
    for (int c = 1; c < cols; c++) {
      float v = this->operator()(r, c);
      if (v > maxVal)
        maxVal = v;
    }
    C(r, 0) = maxVal;
  }
  return C;
}

Tensor Tensor::addMatAndVec(const Tensor &vec) const {
  Tensor C(rows, cols);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      C(r, c) = this->operator()(r, c) + vec(r, 0);
    }
  }
  return C;
}

Tensor Tensor::matDivVecRows(const Tensor &vec) const {
  Tensor C(rows, cols);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      C(r, c) = this->operator()(r, c) / vec(r, 0);
    }
  }
  return C;
}

Tensor Tensor::element_wise_multiply(const Tensor &B) {
  if (rows != B.rows || cols != B.cols) {
    Serial.println(
        "Error: Incompatible dimensions for element-wise multiplication.");
    Serial.print("Tensor A dimensions: ");
    Serial.print(rows);
    Serial.print("x");
    Serial.println(cols);
    Serial.print("Tensor B dimensions: ");
    Serial.print(B.rows);
    Serial.print("x");
    Serial.println(B.cols);
    return Tensor(0, 0);
  }
  Tensor C(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      C(i, j) = this->operator()(i, j) * B(i, j);
    }
  }
  return C;
}

void Tensor::apply(float (*func)(float)) {
  if (func == nullptr)
    return;
  int n = rows * cols;
  for (int i = 0; i < n; ++i) {
    data[i] = func(data[i]);
  }
}
