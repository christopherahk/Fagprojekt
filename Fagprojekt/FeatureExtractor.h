#pragma once
#include <math.h>

// Feature extraction: RBI (8 bins) + Local Stats (Mean, Max) + Global Spatial
// Stats, tailored for 16-sample RMS windows. 56 channels * 10 features = 560
// features.
// + 2 global spatial features = 562 features total.

static const int N_RBI_BINS = 8;
static const int N_LOCAL_STATS = 2;  // Mean and Max per channel
static const int N_GLOBAL_STATS = 2; // Global Mean and Spatial Variance

static void extractFeatures(const float *signal, int nChannels, int window,
                            float *out) {
  int n_per_ch = N_RBI_BINS + N_LOCAL_STATS; // 10 features per ch
  int bin_size = window / N_RBI_BINS;        // 16 / 8 = 2 samples per bin

  float global_sum = 0.0f;
  float global_max = -1e9f;

  for (int ch = 0; ch < nChannels; ch++) {
    const float *seg = signal + ch * window;
    int base = ch * n_per_ch;

    float ch_sum = 0.0f;
    float ch_max = -1e9f;

    // RBI: integrate per bin, log1p compress
    for (int b = 0; b < N_RBI_BINS; b++) {
      float bin_sum = 0.0f;
      for (int s = 0; s < bin_size; s++) {
        float val = seg[b * bin_size + s];
        float abs_val = val < 0.0f ? -val : val;

        bin_sum += abs_val;
        ch_sum += abs_val;

        if (abs_val > ch_max) {
          ch_max = abs_val;
        }
      }
      out[base + b] = log1p(bin_sum);
    }

    // local Statistical features over the full window (log1p compressed)
    float ch_mean = ch_sum / window;
    out[base + N_RBI_BINS] = log1p(ch_mean);
    out[base + N_RBI_BINS + 1] = log1p(ch_max);

    global_sum += ch_mean;
    if (ch_mean > global_max) {
      global_max = ch_mean;
    }
  }

  // calculate global spatial features
  float global_mean = global_sum / nChannels;
  float spatial_variance = global_max - global_mean;

  // append global features at the very end of the array
  int global_base = nChannels * n_per_ch;
  out[global_base] = log1p(global_mean);
  out[global_base + 1] = log1p(spatial_variance);
}
