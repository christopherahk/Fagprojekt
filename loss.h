#include "Tensor.h"

struct Loss {
  float calculate(const Tensor& output, const Tensor& yTrue);
  virtual Tensor forward(const Tensor& yPred, const Tensor& yTrue) = 0;
};
