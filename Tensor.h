#pragma once
// This file only supports 2D Tensors, e.g. matrices, which are flattened to a
// 1D array.

struct Tensor {
  int rows;
  int cols;
  float *data;
  int size;

  // Constructors and destructor.
  Tensor(const float *data_, int rows_,
         int cols_); // Tensor constructor with data, rows and columns.
  Tensor(int rows_, int cols_); // Constructor for empty tensor.
  Tensor(const Tensor &other);  // Copy constructor.
  Tensor();                     // Default constructor.
  ~Tensor();                    // Destructor for the allocated data array.

  // Void functions
  void fill(const float *values); // Used to fill the tensor with data as memory
                                  // for the data array is allocated.
  void printTensor();             // Prints each row of the tensor.

  // Operators
  float &operator()(int row, int col);
  float operator()(int row, int col) const;
  Tensor operator+(const Tensor &B);
  Tensor operator-(const Tensor &B);
  Tensor &operator=(const Tensor &B);

  // Matrix operations
  Tensor matmul(const Tensor &B) const; // Matrix multiplication
  Tensor transpose() const;             // Transpose tensor.
  template <typename Func> Tensor applyElementWise(Func f) const {
    Tensor B(rows, cols);

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        B(r, c) = f(this->operator()(r, c));
      }
    }

    return B;
  }
  Tensor sumRows() const;
  float mean() const;
  Tensor clip(float minValue, float maxValue) const;
  Tensor selectTrueClass(const Tensor &B) const;
  int argmaxRow(int row) const;
  Tensor applyMax() const;
  Tensor addMatAndVec(const Tensor &B) const;
  Tensor matDivVecRows(const Tensor &vec) const;
};
