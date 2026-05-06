#pragma once
#include <math.h>

// Feature extraction from raw EMG signal windows
// Computes 4 statistical features per channel:
// [k*4 + 0]  mean
// [k*4 + 1]  standard deviation
// [k*4 + 2]  RMS (root mean square)
// [k*4 + 3]  max absolute value
// Input:  float[nChannels * window]  (row-major: channel 0 first)
// Output: float[nChannels * 4]

static void extractFeatures(const float *signal, int nChannels, int window,
                            float *out) {
  for (int ch = 0; ch < nChannels; ch++) {
    const float *seg = signal + ch * window;

    float mean = 0.0f;
    float rms = 0.0f;
    float maxAbs = 0.0f;

    // Single pass for mean, RMS accumulator, and max
    for (int i = 0; i < window; i++) {
      mean += seg[i];
      rms += seg[i] * seg[i];
      float a = seg[i] < 0.0f ? -seg[i] : seg[i];
      if (a > maxAbs)
        maxAbs = a;
    }
    mean /= (float)window;
    rms = sqrtf(rms / (float)window);

    // Second pass for variance
    float var = 0.0f;
    for (int i = 0; i < window; i++) {
      float d = seg[i] - mean;
      var += d * d;
    }
    float std = sqrtf(var / (float)window);

    int base = ch * 4;
    out[base + 0] = mean;
    out[base + 1] = std;
    out[base + 2] = rms;
    out[base + 3] = maxAbs;
  }
}
