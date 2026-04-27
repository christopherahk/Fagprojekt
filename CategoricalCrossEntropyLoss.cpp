#include "CategoricalCrossEntropyLoss.h"
#include "Tensor.h"
#include <math.h>

Tensor CategoricalCrossEntropyLoss::forward(const Tensor &yPred,
                                            const Tensor &yTrue) {
  Tensor yPredClipped = yPred.clip(1e-7, 1 - 1e-7);
  Tensor correctConfidences = yPredClipped.selectTrueClass(yTrue);
  correctConfidences.apply([](float x) -> float { return -logf(x); });
  return correctConfidences;
}
