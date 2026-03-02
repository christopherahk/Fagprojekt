#include "MSEloss.h"
#include "Tensor.h"
#include <cmath>

float computeMSEClassLoss(const Tensor &True_vals, const Tensor &Pred_vals) {

  int rows = True_vals.rows;
  int cols = True_vals.cols;

  if (rows != Pred_vals.rows || cols != Pred_vals.cols) {
    // håndter fejl
  }

  float sum = 0.0f;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {

      float error = True_vals(i, j) - Pred_vals(i, j);
      sum += error * error;
    }
  }

  return sum / static_cast<float>(rows);
}
