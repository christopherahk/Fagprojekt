#include "MaxPool2D.h"
#include "Tensor.h"
#include <float.h>
#include <string.h>

MaxPool2D::MaxPool2D(int inChannels_, int inH_, int inW_, int poolH_,
                     int poolW_)
    : output(), dInputs(), inChannels(inChannels_), inH(inH_), inW(inW_),
      poolH(poolH_), poolW(poolW_), outH(inH_ / poolH_), outW(inW_ / poolW_) {
  maxIndices = new int[inChannels_ * (inH_ / poolH_) * (inW_ / poolW_)];
}

MaxPool2D::~MaxPool2D() { delete[] maxIndices; }

void MaxPool2D::forward(const Tensor &x) {
  if (output.rowCount != inChannels || output.colCount != outH * outW) {
    output = Tensor(inChannels, outH * outW);
  }

  for (int ch = 0; ch < inChannels; ch++) {
    for (int oh = 0; oh < outH; oh++) {
      for (int ow = 0; ow < outW; ow++) {
        float maxVal = -FLT_MAX;
        int maxIdx = 0;

        for (int ph = 0; ph < poolH; ph++) {
          int ih = oh * poolH + ph;
          for (int pw = 0; pw < poolW; pw++) {
            int iw = ow * poolW + pw;
            int flatIn = ih * inW + iw;
            float val = x.data[ch * (inH * inW) + flatIn];
            if (val > maxVal) {
              maxVal = val;
              maxIdx = flatIn;
            }
          }
        }

        int outFlat = ch * (outH * outW) + oh * outW + ow;
        output.data[outFlat] = maxVal;
        maxIndices[outFlat] = maxIdx;
      }
    }
  }
}

void MaxPool2D::backward(const Tensor &dValues) {
  if (dInputs.rowCount != inChannels || dInputs.colCount != inH * inW) {
    dInputs = Tensor(inChannels, inH * inW);
  } else {
    memset(dInputs.data, 0, dInputs.size * sizeof(float));
  }

  for (int ch = 0; ch < inChannels; ch++) {
    for (int oh = 0; oh < outH; oh++) {
      for (int ow = 0; ow < outW; ow++) {
        int outFlat = ch * (outH * outW) + oh * outW + ow;
        int srcFlat = maxIndices[outFlat];
        dInputs.data[ch * (inH * inW) + srcFlat] += dValues.data[outFlat];
      }
    }
  }
}
