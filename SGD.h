#pragma once
#include "NNLayer.h"

struct SGD {
  int learningRate;

  SGD(float lr);

  void updateParameters(NNLayer layer);
};