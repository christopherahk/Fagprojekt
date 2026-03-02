#include "CCEntloss.h"
#include "Tensor.h"
#include <cmath>

float computeCCEntropyLossOneHot(const Tensor &True_onehot,
                                 const Tensor &Pred_vals) {
  int rows = Pred_vals.rows;
  int cols = Pred_vals.cols; // num classes
  float total_loss = 0.0f;
  const float epsilon = 1e-7f;

  for (int i = 0; i < rows; ++i) {
    int true_class = -1;

    // One-Hot search
    for (int j = 0; j < cols; ++j) {
      if (True_onehot(i, j) > 0.9f) {
        true_class = j;
        break;
      }
    }

    // Safety check
    if (true_class == -1)
      continue;

    float y_pred = Pred_vals(i, true_class);

    // Clipping
    if (y_pred < epsilon)
      y_pred = epsilon;
    if (y_pred > 1.0f - epsilon)
      y_pred = 1.0f - epsilon;

    total_loss += -std::log(y_pred);
  }

  return total_loss / static_cast<float>(rows);
}
