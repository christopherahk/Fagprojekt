#include "Conv2DLayer.h"
#include "Tensor.h"
#include <Arduino.h>
#include <math.h>

namespace {
const float kGradClipLimit = 100.0f;
}

static float clampValue(float x, float limit) {
  if (x > limit)
    return limit;
  if (x < -limit)
    return -limit;
  return x;
}

static void ensureTensorShape(Tensor &t, int rows, int cols) {
  if (t.rowCount != rows || t.colCount != cols) {
    t = Tensor(rows, cols);
  } else {
    memset(t.data, 0, t.size * sizeof(float));
  }
}

static inline int inIdx(int h, int w, int inW) { return h * inW + w; }

static inline int outIdx(int outCh, int oh, int ow, int outW) {
  return outCh * outW + oh * outW +
         ow; // note: stored as (outChannels, outH*outW)
}

static inline int wIdx(int outCh, int kh, int kw, int kernelW) {
  return outCh * (/* kernelH * */ kernelW) + kh * kernelW + kw;
}

Conv2DLayer::Conv2DLayer(int inH_, int inW_, int outChannels_, int kernelH_,
                         int kernelW_, int strideH_, int strideW_, int padH_,
                         int padW_)
    : weights(outChannels_, kernelH_ * kernelW_), biases(1, outChannels_),
      inputs(nullptr), output(), dWeights(), dBiases(), dInputs(), inH(inH_),
      inW(inW_), outChannels(outChannels_), kernelH(kernelH_),
      kernelW(kernelW_), strideH(strideH_), strideW(strideW_), padH(padH_),
      padW(padW_), outH((inH_ + 2 * padH_ - kernelH_) / strideH_ + 1),
      outW((inW_ + 2 * padW_ - kernelW_) / strideW_ + 1) {
  for (int i = 0; i < weights.size; i++) {
    weights.data[i] = static_cast<float>(random(-100, 101)) / 1000.0f;
  }
  for (int c = 0; c < outChannels; c++) {
    biases.data[c] = 0.0f;
  }
}

void Conv2DLayer::forward(const Tensor &x) {
  inputs = &x;
  ensureTensorShape(output, outChannels, outH * outW);

  for (int outCh = 0; outCh < outChannels; outCh++) {
    for (int oh = 0; oh < outH; oh++) {
      for (int ow = 0; ow < outW; ow++) {
        float sum = biases.data[outCh];

        for (int kh = 0; kh < kernelH; kh++) {
          int ih = oh * strideH + kh - padH;
          if (ih < 0 || ih >= inH)
            continue; // zero-padding

          for (int kw = 0; kw < kernelW; kw++) {
            int iw = ow * strideW + kw - padW;
            if (iw < 0 || iw >= inW)
              continue; // zero-padding

            sum +=
                x.data[inIdx(ih, iw, inW)] *
                weights.data[outCh * (kernelH * kernelW) + kh * kernelW + kw];
          }
        }

        output.data[outCh * (outH * outW) + oh * outW + ow] = sum;
      }
    }
  }
}

void Conv2DLayer::backward(const Tensor &dValues, bool computeDInputs_) {
  if (inputs == nullptr)
    return;

  ensureTensorShape(dWeights, outChannels, kernelH * kernelW);
  ensureTensorShape(dBiases, 1, outChannels);

  for (int outCh = 0; outCh < outChannels; outCh++) {
    float biasGrad = 0.0f;

    for (int oh = 0; oh < outH; oh++) {
      for (int ow = 0; ow < outW; ow++) {
        float dv = dValues.data[outCh * (outH * outW) + oh * outW + ow];
        biasGrad += dv;

        for (int kh = 0; kh < kernelH; kh++) {
          int ih = oh * strideH + kh - padH;
          if (ih < 0 || ih >= inH)
            continue;

          for (int kw = 0; kw < kernelW; kw++) {
            int iw = ow * strideW + kw - padW;
            if (iw < 0 || iw >= inW)
              continue;

            dWeights.data[outCh * (kernelH * kernelW) + kh * kernelW + kw] +=
                dv * inputs->data[inIdx(ih, iw, inW)];
          }
        }
      }
    }

    dBiases.data[outCh] = biasGrad;
  }

  if (computeDInputs_) {
    ensureTensorShape(dInputs, inH, inW);

    for (int outCh = 0; outCh < outChannels; outCh++) {
      for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
          float dv = dValues.data[outCh * (outH * outW) + oh * outW + ow];

          for (int kh = 0; kh < kernelH; kh++) {
            int ih = oh * strideH + kh - padH;
            if (ih < 0 || ih >= inH)
              continue;

            for (int kw = 0; kw < kernelW; kw++) {
              int iw = ow * strideW + kw - padW;
              if (iw < 0 || iw >= inW)
                continue;

              dInputs.data[inIdx(ih, iw, inW)] +=
                  dv *
                  weights.data[outCh * (kernelH * kernelW) + kh * kernelW + kw];
            }
          }
        }
      }
    }
  }
}

void Conv2DLayer::update(float learningRate) {
  for (int i = 0; i < weights.size; i++) {
    float grad = isfinite(weights.data[i])
                     ? clampValue(dWeights.data[i], kGradClipLimit)
                     : 0.0f;
    weights.data[i] -= learningRate * grad;
  }
  for (int c = 0; c < biases.colCount; c++) {
    float grad = isfinite(biases.data[c])
                     ? clampValue(dBiases.data[c], kGradClipLimit)
                     : 0.0f;
    biases.data[c] -= learningRate * grad;
  }
}
