#pragma once

#include "Tensor.h"

class Loss {
public:
  // Calculates the mean loss from outputs and true labels.
  float calculate(const Tensor &output, const Tensor &yTrue);

  // Forward pass — must be implemented by derived loss classes.
  virtual Tensor forward(const Tensor &output, const Tensor &yTrue) = 0;

  virtual ~Loss() = default;
};

// ligegyldig kommentar
