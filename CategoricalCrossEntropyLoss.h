#pragma once
#include "Loss.h"

struct CategoricalCrossEntropyLoss : Loss {
  float forward(const Tensor &yPred, const Tensor &yTrue) override;
};
