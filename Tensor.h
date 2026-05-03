#pragma once

struct Tensor {
  int rowCount;
  int colCount;
  float *data;
  int size;

  Tensor(int rows, int cols);
  Tensor(const Tensor &other);
  Tensor();
  ~Tensor();

  float &operator()(int row, int col);
  float operator()(int row, int col) const;
  Tensor &operator+=(const Tensor &other);
  Tensor &operator-=(const Tensor &other);
  Tensor &operator*=(float scalar);
  Tensor &operator=(const Tensor &other);
  Tensor &operator=(Tensor &&other);

  Tensor &matmul(const Tensor &B);
  Tensor &transpose();
  Tensor sumRows() const;
  Tensor sumCols() const;
  float mean() const;
  Tensor &clip(float minValue, float maxValue);
  Tensor selectTrueClass(const Tensor &other) const;
  int argmaxRow(int row) const;
  void apply(float (*func)(float));
};
