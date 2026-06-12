#pragma once
#include <math.h>

// Feature extraction: RBI (4 bins) + FFT power spectrum per channel
// Combines two related feature types:
//
//   RBI (Rectification and Bin-Integration):
//     Captures WHEN amplitude is high within the window.
//     4 bins * 56 channels = 224 features.
//     Computed as: |signal| summed over each bin, log1p compressed. they used
//     this in "Classification of naturally evoked compound action potentials in
//     peripheral nerve spatiotemporal recordings" Ryan G. L. Koh, Adrian I.
//     Nachman & José Zariffa

//   FFT power spectrum:
//     Captures WHICH frequencies are present.
//     17 bins * 56 channels = 952 features.
//     Hanning windowed, log1p compressed.

//   Total: 56 * (4 + 17) = 56 * 21 = 1176 features

// Both use log1p compression to fit Q8.8 fixed-point range [-128, 127].

// Input:  float[nChannels * window]  (row-major: channel 0 first)
// Output: float[nChannels * (N_RBI_BINS + window/2 + 1)]
// No heap allocation -- writes directly into caller's buffer.

static const int N_RBI_BINS = 4;

// Radix-2 DIT FFT for real input.
// N must equal WINDOW (32). Hardcoded size avoids VLAs on Arduino.
static void fft_power(const float *in, int N, float *power_out) {
  float re[32], im[32];

  // Apply Hanning window and copy to work buffers
  for (int i = 0; i < N; i++) {
    float w = 0.5f * (1.0f - cosf(2.0f * 3.14159265f * i / (N - 1)));
    re[i] = in[i] * w;
    im[i] = 0.0f;
  }

  // bit-reversal permutation
  int j = 0;
  for (int i = 1; i < N; i++) {
    int bit = N >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      float ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  // Cooley-Tukey butterfly
  for (int len = 2; len <= N; len <<= 1) {
    float ang = -2.0f * 3.14159265f / len;
    float wRe = cosf(ang);
    float wIm = sinf(ang);
    for (int i = 0; i < N; i += len) {
      float curRe = 1.0f, curIm = 0.0f;
      for (int k = 0; k < len / 2; k++) {
        float uRe = re[i + k];
        float uIm = im[i + k];
        float vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        float vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;
        float newRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newRe;
      }
    }
  }

  // Power spectrum for positive frequencies only, log1p compressed
  int n_bins = N / 2 + 1;
  for (int k = 0; k < n_bins; k++)
    power_out[k] = log1p(re[k] * re[k] + im[k] * im[k]);
}

static void extractFeatures(const float *signal, int nChannels, int window,
                            float *out) {
  int n_fft_bins = window / 2 + 1;        // 17 at window=32
  int n_per_ch = N_RBI_BINS + n_fft_bins; // 4 + 17 = 21
  int bin_size = window / N_RBI_BINS;     // 8 samples per bin

  for (int ch = 0; ch < nChannels; ch++) {
    const float *seg = signal + ch * window;
    int base = ch * n_per_ch;

    // RBI: rectify, integrate per bin, log1p compress
    for (int b = 0; b < N_RBI_BINS; b++) {
      float sum = 0.0f;
      for (int s = 0; s < bin_size; s++)
        sum += seg[b * bin_size + s] < 0.0f ? -seg[b * bin_size + s]
                                            : seg[b * bin_size + s];
      out[base + b] = log1p(sum);
    }

    // FFT power spectrum
    fft_power(seg, window, out + base + N_RBI_BINS);
  }
}
