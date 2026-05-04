#pragma once
#include "NNLayer.h"

struct SGD {
  int learningRate;

  explicit SGD(float lr);

  void updateParameters(NNLayer layer);
};
