#include "LeakyReU.h"
#include "Tensor.h"

void LeakyReLU::forward(const Tensor& inputs_, const float alpha_) {
    inputs = inputs_;
    alpha = alpha_;

    for (int r = 0; r < inputs_.rows; r++) {
        for (int c = 0; c < inputs_.cols; c++) {
            float x = inputs_(r, c);
            output(r, c) = (x < 0.0f) ? alpha_ * x : x;
        }
    }
}

void LeakyReLU::backward(const Tensor& dValues) {
    dInputs = dValues;

    for (int r = 0; r < dValues.rows; r++) {
        for (int c = 0; c < dValues.cols; c++) {
            if (inputs(r, c) < 0) {
                dInputs(r, c) = alpha;
            }
        }
    }
}