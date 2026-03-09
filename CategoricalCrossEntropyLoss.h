#pragma once
#include "Loss.h"

struct CategoricalCrossEntropyLoss : Loss {
  Tensor forward(const Tensor &yPred, const Tensor &yTrue) override;
};
