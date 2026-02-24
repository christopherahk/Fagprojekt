#include "Loss.h"
#include "Tensor.h"

float Loss::calculate(const Tensor& output, const Tensor& yTrue) {
  Tensor sampleLosses = forward(output, yTrue);
  
  return sampleLosses.mean();
}