#include "GetLoss.h"
#include <math.h>

float GetLoss::forward(const Tensor &yPred, const Tensor &yTrue) {
  activation.forward(yPred);
  return loss.calculate(activation.output, yTrue);
}

// Combined Softmax + Focal Loss gradient w.r.t. the pre-softmax logits.
//
// For plain CCE+softmax, dL/dz = (p - y). Focal loss reweights that exact
// same direction by a per-sample scalar:
//
//   S = (1 - p_t)^gamma  -  gamma * p_t * (1 - p_t)^(gamma - 1) * ln(p_t)
//   dL/dz_k = S * (p_k - y_k)
//
// where p_t is the predicted probability of the true class for that sample.
// At gamma = 0, S = 1 and this is identical to the original CCE gradient.
void GetLoss::backward(const Tensor &yTrue) {
  const Tensor &probs = activation.output;
  const float gamma = loss.gamma;

  dInputs = probs;

  for (int r = 0; r < yTrue.rowCount; r++) {
    float pt = 0.0f;
    for (int c = 0; c < yTrue.colCount; c++) {
      if (yTrue(r, c) == 1.0f) {
        pt = probs(r, c);
        break;
      }
    }

    if (pt < 1e-7f)
      pt = 1e-7f;
    else if (pt > 1.0f - 1e-7f)
      pt = 1.0f - 1e-7f;

    const float oneMinusPt = 1.0f - pt;
    float scale = powf(oneMinusPt, gamma);
    if (gamma > 0.0f) {
      scale -= gamma * pt * powf(oneMinusPt, gamma - 1.0f) * logf(pt);
    }

    for (int c = 0; c < yTrue.colCount; c++) {
      dInputs(r, c) = scale * (probs(r, c) - yTrue(r, c)) / yTrue.rowCount;
    }
  }
}
