#include "CategoricalCrossEntropyLoss.h"
#include "Tensor.h"
#include <cmath>

Tensor CategoricalCrossEntropyLoss::forward(const Tensor& yPred, const Tensor& yTrue) {
  Tensor yPredClipped = yPred.clip(1e-7, 1 - 1e-7);

  Tensor correctConfidences = yPredClipped.selectTrueClass(yTrue);

  auto negativeLogLikelihoodFunc = [](const float& x) -> float { return -log(x); };
  Tensor negativeLogLikelihoods = correctConfidences.applyElementWise(negativeLogLikelihoodFunc);

  return negativeLogLikelihoods;
}