#include "CCEntloss.h"
#include "Tensor.h"
#include <cmath>

float computeCCEntropyLoss(const Tensor &True_vals, const Tensor &Pred_vals) {
  // An implementation of categorical cross entropy loss for classification.
  int N = True_vals.size(); // total entries in the tensor i.e. "N"

  // check if tensors are of same size

  if (N != Pred_vals.size()) {
  }

  float sum = 0.0f;

  // compute -yi log(yhat)

  for (int i = 0; i < N; ++i) {
    float y_true = True_vals(i, 0);
    float y_pred = Pred_vals(i, 0);
    sum += -y_true *
           std::log(y_pred + 1e-15f); // Adding a small value to prevent log(0)
  }

  // Sum/N, static cast N(int) -> N(float)
  return sum / static_cast<float>(N);
}
