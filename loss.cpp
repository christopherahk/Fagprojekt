#include "loss.h"
#include "Tensor.h"

float computeMSEClassLoss(const Tensor &True_vals, const Tensor &Pred_vals) {

  // An implementation of MSE multiclass loss function. Not suited for
  // classification but will be used for testing
  int N = True_vals.size(); // total entries in the tensor i.e. "N"

  // check if tensors are of same size
  if (N != Pred_vals.size()) {
  }
  float sum = 0.0f;
  // compute yi - yhat
  for (int i = 0; i < N; ++i) {
    float error = True_vals(i, 0) - Pred_vals(i, 0);
    // square that shit
    sum += error * error;
  }
  // Sum/N, static cast N(int) -> N(float)
  return sum / static_cast<float>(N); // MUAHAHAHHAHAHAHAHA
}
