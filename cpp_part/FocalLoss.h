#pragma once
#include "Loss.h"

// Focal Loss (Lin et al., 2017): FL = -(1 - p_t)^gamma * log(p_t)
// Reduces to standard categorical cross-entropy when gamma = 0.
// gamma > 0 down-weights easy/confident examples (p_t close to 1) and
// keeps full gradient signal on hard/misclassified examples (p_t close to 0).
struct FocalLoss : Loss {
  float gamma;

  FocalLoss(float gamma_ = 2.0f) : gamma(gamma_) {}

  Tensor forward(const Tensor &yPred, const Tensor &yTrue) override;
};
