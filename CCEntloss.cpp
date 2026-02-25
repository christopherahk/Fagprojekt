#include "CCEntloss.h"
#include "Tensor.h"
#include <cmath>

float computeCCEntropyLoss(const Tensor &True_vals, const Tensor &Pred_vals) {
  int rows = True_vals.rows; // # samples
  int cols = Pred_vals.cols; // # classes

  if (rows != Pred_vals.rows) {
    // error handling
  }

  float sum = 0.0f;

  for (int i = 0; i < rows; ++i) {

    int true_class = static_cast<int>(True_vals(i, 0));
    float y_pred = Pred_vals(i, true_class);

    if (y_pred < 1e-15f)
      y_pred = 1e-15f;

    sum += -std::log(y_pred);
  }

  return sum / static_cast<float>(rows);
}
