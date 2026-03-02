#include "Tensor.h"
#include <Arduino.h>
#include <algorithm>
#include <string>
#include <utility> // used for std::swap

Tensor::Tensor(const float *data_, int rows_, int cols_)
    : rows(rows_), cols(cols_) {
  // Initializes the data with the specified size and fills the values from the
  // input data.
  int size = rows * cols;
  data = new float[size];
  fill(data_);
}

Tensor::Tensor(int rows_, int cols_) : rows(rows_), cols(cols_) {
  /* Constructs a tensor without setting the data values.
  Used for when we just need an empty tensor we will fill later. */
  int size = rows * cols;
  data = new float[size];
}

Tensor::Tensor(const Tensor &other) : rows(other.rows), cols(other.cols) {
  // Copy constructor - deep copies the data array.
  int size = rows * cols;
  data = new float[size];
  std::copy(other.data, other.data + size, data);
}

Tensor::~Tensor() {
  // Destructor - for freeing up memory allocated for the data.
  delete[] data;
}

void Tensor::fill(const float *values) {
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

void Tensor::swap(Tensor &B) {
  // Swap function for the copy and swap idiom.
  std::swap(rows, B.rows); //
  std::swap(cols, B.cols);
  std::swap(data, B.data);
}

Tensor &Tensor::operator=(Tensor B) { // copy
  swap(B);                            // swap internal data
  return *this;                       // return the modified tensor
}

Tensor Tensor::matmul(const Tensor &B) {
  /* Matrix multiplication. Constructs an empty tensor, makes the basic
  algorithm for multiplication, then returns the resulting tensor. */

  if (cols != B.rows) {
    // Handle error for incompatible dimensions, return an empty tensor or
    // handle as needed
    Serial.println("Error: Incompatible dimensions for matrix multiplication.");
    Serial.print("Tensor A dimensions: ");
    Serial.print(rows);
    Serial.print("x");
    Serial.println(cols);
    Serial.print("Tensor B dimensions: ");
    Serial.print(B.rows);
    Serial.print("x");
    Serial.println(B.cols);
    return Tensor(0, 0); // Return an empty tensor to indicate an error
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

int Tensor::size() const {

  // Get the size of a tensor, i.e how many entries

  return rows * cols;
}

int Tensor::getrow() const {

  // Get the lenght of a tensor

  return rows;
}

Tensor Tensor::operator*(float scalar) { // scalar multiplication
  Tensor C(rows, cols) int n = rows * cols;
  for (int i = 0; i < n; i++) {
    C.data[i] = data[i] * scalar;
  }
  return C;
}

Tensor operator*(float scalar, const Tensor &A) {
  return A * scalar; // Reuse the member operator* for scalar multiplication
}

Tensor Tensor::element_wise_multiply(const Tensor &B) {
  // Element-wise multiplication. Constructs an empty tensor, makes the basic
  // algorithm for element-wise multiplication, then returns the resulting
  // tensor.

  if (rows != B.rows || cols != B.cols) {
    // Handle error for incompatible dimensions, return an empty tensor or
    // handle as needed
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
    return Tensor(0, 0); // Return an empty tensor to indicate an error
  }

  Tensor C(rows, cols);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      C(i, j) = this->operator()(i, j) * B(i, j); // Element-wise multiplication
    }
  }

  return C;
}

void Tensor::apply(
    float (*func)(float)) { // apply a function to each element of the tensor
  if (func == nullptr)
    return; // Handle null function pointer, do nothing or handle as needed

  int n = rows * cols;          // Total number of elements in the tensor
  for (int i = 0; i < n; ++i) { // fricking for loop
    data[i] = func(data[i]); // Apply the function to each element of the tensor
  }
}
