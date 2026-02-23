#pragma once

// This file only supports 2D Tensors, e.g. matrices, which are flattened to a
// 1D array.

struct Tensor {
  int rows;
  int cols;
  float *data;

  // Constructors and destructor.
  Tensor(const float *data_, int rows_,
         int cols_); // Tensor constructor with data, rows and columns.
  Tensor(int rows_, int cols_); // Constructor for empty tensor.
  Tensor(const Tensor &other);  // Copy constructor.
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

  Tensor matmul(const Tensor &B); // Matrix multiplication
  Tensor transpose();             // Transpose tensor.
  int size() const;               // Get the number of entries in the matrix
  int getrow() const;             // Get the lenght of the tensor
};
