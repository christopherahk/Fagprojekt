#include "Tensor.h"
#include <algorithm>

Tensor::Tensor(int rows, int cols)
    : rowCount(rows), colCount(cols), size(rows * cols) {
  data = new float[size];
}

Tensor::Tensor(const Tensor &other)
    : rowCount(other.rowCount), colCount(other.colCount), size(other.size) {
  data = new float[size];
  std::copy(other.data, other.data + size, data);
}

Tensor::Tensor() : rowCount(0), colCount(0), size(0), data(nullptr) {}

Tensor::~Tensor() { delete[] data; }

float &Tensor::operator()(int row, int col) {
  return data[row * colCount + col];
}

float Tensor::operator()(int row, int col) const {
  return data[row * colCount + col];
}

Tensor &Tensor::operator+=(const Tensor &other) {
  for (int i = 0; i < size; i++)
    data[i] += other.data[i];
  return *this;
}

Tensor &Tensor::operator-=(const Tensor &other) {
  for (int i = 0; i < size; i++)
    data[i] -= other.data[i];
  return *this;
}

Tensor &Tensor::operator*=(float scalar) {
  for (int i = 0; i < size; i++)
    data[i] *= scalar;
  return *this;
}

Tensor &Tensor::operator=(const Tensor &other) {
  if (this != &other) {
    float *newData = new float[other.size];
    std::copy(other.data, other.data + other.size, newData);
    delete[] data;
    data = newData;
    rowCount = other.rowCount;
    colCount = other.colCount;
    size = other.size;
  }
  return *this;
}

Tensor &Tensor::operator=(Tensor &&other) {
  if (this != &other) {
    delete[] data;
    rowCount = other.rowCount;
    colCount = other.colCount;
    size = other.size;
    data = other.data;

    other.data = nullptr;
    other.rowCount = 0;
    other.colCount = 0;
    other.size = 0;
  }
  return *this;
}

Tensor &Tensor::matmul(const Tensor &other) {
  Tensor C(rowCount, other.colCount);
  for (int i = 0; i < rowCount; i++) {
    for (int k = 0; k < colCount; k++) {
      float a = (*this)(i, k);
      for (int j = 0; j < other.colCount; j++) {
        C(i, j) += a * other(k, j);
      }
    }
  }
  *this = std::move(C);
  return *this;
}

Tensor &Tensor::transpose() {
  if (rowCount == colCount) {
    for (int r = 0; r < rowCount; r++)
      for (int c = r + 1; c < colCount; c++)
        std::swap((*this)(r, c), (*this)(c, r));
  } else {
    Tensor B(colCount, rowCount);
    for (int r = 0; r < rowCount; r++)
      for (int c = 0; c < colCount; c++)
        B(c, r) = (*this)(r, c);
    *this = std::move(B);
  }
  return *this;
}

Tensor Tensor::sumRows() const {
  Tensor C(rowCount, 1);
  for (int r = 0; r < rowCount; r++) {
    float sum = 0.0f;
    for (int c = 0; c < colCount; c++)
      sum += (*this)(r, c);
    C(r, 0) = sum;
  }
  return C;
}

Tensor Tensor::sumCols() const {
  Tensor C(1, colCount);
  for (int c = 0; c < colCount; c++) {
    float sum = 0.0f;
    for (int r = 0; r < rowCount; r++)
      sum += (*this)(r, c);
    C(0, c) = sum;
  }
  return C;
}

float Tensor::mean() const {
  if (size == 0) {
    return 0.0f;
  }
  float result = 0.0f;
  for (int i = 0; i < size; i++) {
    result += data[i];
  }
  return result / size;
}

Tensor &Tensor::clip(float minValue, float maxValue) {
  for (int i = 0; i < size; i++) {
    if (data[i] < minValue)
      data[i] = minValue;
    else if (data[i] > maxValue)
      data[i] = maxValue;
  }
  return *this;
}

Tensor Tensor::selectTrueClass(const Tensor &other) const {
  Tensor C(rowCount, 1);
  for (int r = 0; r < rowCount; r++) {
    for (int c = 0; c < colCount; c++) {
      if (other(r, c) == 1.0f) {
        C(r, 0) = (*this)(r, c);
        break;
      }
    }
  }
  return C;
}

int Tensor::argmaxRow(int row) const {
  int colMax = 0;
  float maxVal = (*this)(row, 0);
  for (int c = 1; c < colCount; c++) {
    if ((*this)(row, c) > maxVal) {
      maxVal = (*this)(row, c);
      colMax = c;
    }
  }
  return colMax;
}

void Tensor::apply(float (*func)(float)) {
  if (func == nullptr)
    return;
  for (int i = 0; i < size; i++) {
    data[i] = func(data[i]);
  }
}
