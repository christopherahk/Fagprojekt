#pragma once
#include "Tensor.h"

struct Loss {
  virtual float forward(const Tensor &yPred, const Tensor &yTrue) = 0;
  virtual ~Loss() = default;
};
