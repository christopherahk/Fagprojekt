#include "FocalLoss.h"
#include "Tensor.h"
#include <math.h>

Tensor FocalLoss::forward(const Tensor &yPred, const Tensor &yTrue) {
  Tensor yPredClipped = yPred;
  yPredClipped.clip(1e-7f, 1.0f - 1e-7f);

  // p_t: predicted probability assigned to the true class, per sample (row).
  Tensor pt = yPredClipped.selectTrueClass(yTrue);

  for (int i = 0; i < pt.size; i++) {
    float p = pt.data[i];
    pt.data[i] = -powf(1.0f - p, gamma) * logf(p);
  }

  return pt;
}
