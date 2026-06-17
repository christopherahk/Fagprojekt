#include "Conv2dlayer.h"
#include "Tensor.h"
#include <Arduino.h>
#include <math.h>

namespace {
const float kGradClipLimit = 100.0f;
const float kAdamBeta1 = 0.9f;
const float kAdamBeta2 = 0.999f;
const float kAdamEps = 1e-8f;
} // namespace

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

Conv2DLayer::Conv2DLayer(int inH_, int inW_, int outChannels_, int kernelH_,
                         int kernelW_, int strideH_, int strideW_, int padH_,
                         int padW_)
    : weights(outChannels_, kernelH_ * kernelW_), biases(1, outChannels_),
      inputs(nullptr), output(), dWeights(), dBiases(), dInputs(),
      mWeights(outChannels_, kernelH_ * kernelW_),
      vWeights(outChannels_, kernelH_ * kernelW_), mBiases(1, outChannels_),
      vBiases(1, outChannels_), adamT(0), inH(inH_), inW(inW_),
      outChannels(outChannels_), kernelH(kernelH_), kernelW(kernelW_),
      strideH(strideH_), strideW(strideW_), padH(padH_), padW(padW_),
      outH((inH_ + 2 * padH_ - kernelH_) / strideH_ + 1),
      outW((inW_ + 2 * padW_ - kernelW_) / strideW_ + 1) {

  memset(mWeights.data, 0, mWeights.size * sizeof(float));
  memset(vWeights.data, 0, vWeights.size * sizeof(float));
  memset(mBiases.data, 0, mBiases.size * sizeof(float));
  memset(vBiases.data, 0, vBiases.size * sizeof(float));

  float heStd = sqrtf(2.0f / (kernelH_ * kernelW_ * inH_));
  for (int i = 0; i < weights.size; i++) {
    float u1 = (random(1, 100001)) / 100000.0f;
    float u2 = (random(1, 100001)) / 100000.0f;
    float randn = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    weights.data[i] = randn * heStd;
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
            continue;

          for (int kw = 0; kw < kernelW; kw++) {
            int iw = ow * strideW + kw - padW;
            if (iw < 0 || iw >= inW)
              continue;

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

  if (dWeights.rowCount != outChannels ||
      dWeights.colCount != kernelH * kernelW) {
    dWeights = Tensor(outChannels, kernelH * kernelW);
    memset(dWeights.data, 0, dWeights.size * sizeof(float));
  }
  if (dBiases.rowCount != 1 || dBiases.colCount != outChannels) {
    dBiases = Tensor(1, outChannels);
    memset(dBiases.data, 0, dBiases.size * sizeof(float));
  }

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
    dBiases.data[outCh] += biasGrad;
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
  adamT++;
  float bc1 = 1.0f - powf(kAdamBeta1, (float)adamT);
  float bc2 = 1.0f - powf(kAdamBeta2, (float)adamT);

  for (int i = 0; i < weights.size; i++) {
    float g = isfinite(dWeights.data[i])
                  ? clampValue(dWeights.data[i], kGradClipLimit)
                  : 0.0f;
    mWeights.data[i] = kAdamBeta1 * mWeights.data[i] + (1.0f - kAdamBeta1) * g;
    vWeights.data[i] =
        kAdamBeta2 * vWeights.data[i] + (1.0f - kAdamBeta2) * g * g;
    float mHat = mWeights.data[i] / bc1;
    float vHat = vWeights.data[i] / bc2;
    weights.data[i] -= learningRate * mHat / (sqrtf(vHat) + kAdamEps);
  }

  for (int c = 0; c < biases.colCount; c++) {
    float g = isfinite(dBiases.data[c])
                  ? clampValue(dBiases.data[c], kGradClipLimit)
                  : 0.0f;
    mBiases.data[c] = kAdamBeta1 * mBiases.data[c] + (1.0f - kAdamBeta1) * g;
    vBiases.data[c] =
        kAdamBeta2 * vBiases.data[c] + (1.0f - kAdamBeta2) * g * g;
    float mHat = mBiases.data[c] / bc1;
    float vHat = vBiases.data[c] / bc2;
    biases.data[c] -= learningRate * mHat / (sqrtf(vHat) + kAdamEps);
  }

  memset(dWeights.data, 0, dWeights.size * sizeof(float));
  memset(dBiases.data, 0, dBiases.size * sizeof(float));
}
