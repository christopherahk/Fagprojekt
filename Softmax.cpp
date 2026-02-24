#include "Softmax.h"
#include <cmath>

void Softmax::forward(const Tensor& inputs) {
  Tensor maxValues = inputs.applyMax();

  auto negativeFunc = [](const float& x) -> float { return -x; };
  Tensor negativeMax = maxValues.applyElementWise(negativeFunc);

  Tensor shifted = inputs.addMatAndVec(negativeMax);

  auto expFunc = [](const float& x) -> float { return exp(x); };
  Tensor expValues = shifted.applyElementWise(expFunc);

  Tensor sumExp = expValues.sumRows();

  output = expValues.matDivVecRows(sumExp);
}