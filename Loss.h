#pragma once
#include "Tensor.h"

struct Loss {
  virtual Tensor forward(const Tensor &yPred, const Tensor &yTrue) = 0;
  virtual ~Loss() = default;
};
