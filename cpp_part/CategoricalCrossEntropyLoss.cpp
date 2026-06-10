#include "CategoricalCrossEntropyLoss.h"
#include "Tensor.h"
#include <math.h>

Tensor CategoricalCrossEntropyLoss::forward(const Tensor &yPred,
                                            const Tensor &yTrue) {
  Tensor yPredClipped = yPred;
  yPredClipped.clip(1e-7f, 1.0f - 1e-7f);
  Tensor correctConfidences = yPredClipped.selectTrueClass(yTrue);
  correctConfidences.apply([](float x) -> float { return -logf(x); });
  return correctConfidences;
}
