#pragma once
#include <math.h>
#include <stdint.h>

// ADWIN (ADaptive WINdowing) drift detector
// Splits the window into two halves and tests whether the error rate has
// changed significantly, if so drift happened Memory: O(log N) buckets. At
// MAX_BUCKETS=32:
//   32 * (4 + 2) bytes = 192 bytes per instance

class ADWIN {
public:
  static const int MAX_BUCKETS = 32;

  explicit ADWIN(float delta = 0.002f)
      : delta_(delta), width_(0), bucketCount_(0) {
    for (int i = 0; i < MAX_BUCKETS; i++) {
      buckets_[i].sum = 0.0f;
      buckets_[i].count = 0;
    }
  }

  // Feed one observation (1 = error, 0 = correct)
  // Returns true if drift was detected
  bool update(bool error) {
    addBucket(error ? 1.0f : 0.0f, 1);
    width_++;
    compressBuckets();
    return detectDrift();
  }

  void reset() {
    width_ = 0;
    bucketCount_ = 0;
    for (int i = 0; i < MAX_BUCKETS; i++) {
      buckets_[i].sum = 0.0f;
      buckets_[i].count = 0;
    }
  }

  int width() const { return width_; }
  float mean() const { return width_ > 0 ? totalSum() / width_ : 0.0f; }

private:
  // Each bucket holds a compressed block of observations
  // count is always a power of 2
  struct Bucket {
    float sum;
    int16_t count;
  };

  Bucket buckets_[MAX_BUCKETS];
  int16_t bucketCount_;
  int16_t width_;
  float delta_;

  void addBucket(float sum, int count) {
    if (bucketCount_ >= MAX_BUCKETS) {
      // Drop oldest bucket for memorys sake
      removeBucket();
    }
    buckets_[bucketCount_].sum = sum;
    buckets_[bucketCount_].count = (int16_t)count;
    bucketCount_++;
  }

  // Remove the oldest bucket, idx 0, and shift everything left
  void removeBucket() {
    if (bucketCount_ == 0)
      return;
    width_ -= buckets_[0].count;
    if (width_ < 0)
      width_ = 0;
    for (int i = 0; i < bucketCount_ - 1; i++)
      buckets_[i] = buckets_[i + 1];
    bucketCount_--;
  }

  // Merge adjacent bucket pairs of same count
  void compressBuckets() {
    for (int lvl = 0; lvl < MAX_BUCKETS - 1; lvl++) {
      int16_t levelSize = (int16_t)(1 << lvl);
      int matches = 0;
      for (int i = 0; i < bucketCount_; i++)
        if (buckets_[i].count == levelSize)
          matches++;

      if (matches <= 1)
        break;

      for (int i = 0; i < bucketCount_ - 1; i++) {
        if (buckets_[i].count == levelSize &&
            buckets_[i + 1].count == levelSize) {
          buckets_[i].sum += buckets_[i + 1].sum;
          buckets_[i].count *= 2;
          for (int j = i + 1; j < bucketCount_ - 1; j++)
            buckets_[j] = buckets_[j + 1];
          bucketCount_--;
          break;
        }
      }
    }
  }

  float totalSum() const {
    float s = 0.0f;
    for (int i = 0; i < bucketCount_; i++)
      s += buckets_[i].sum;
    return s;
  }

  // Hoeffding-bound drift test
  // Scans all split points (W0 | W1) of the current window
  // If |mean(W0) - mean(W1)| exceeds the bound, drift is detected and the
  // oldest part of window is dropped
  bool detectDrift() {
    if (width_ < 4)
      return false;

    float total = totalSum();
    float n0 = 0.0f;
    float sum0 = 0.0f;

    for (int i = bucketCount_ - 1; i >= 0; i--) {
      n0 += buckets_[i].count;
      sum0 += buckets_[i].sum;

      float n1 = (float)width_ - n0;
      if (n1 < 1.0f)
        continue;

      float mu0 = sum0 / n0;
      float mu1 = (total - sum0) / n1;

      float m = 1.0f / (1.0f / n0 + 1.0f / n1);
      float eps = sqrtf(1.0f / (2.0f * m) *
                        logf(4.0f * (float)width_ * (float)width_ / delta_));

      if (fabsf(mu0 - mu1) > eps) {
        // Trim the older half of the window
        for (int k = 0; k < (int)n1; k++)
          removeBucket();
        return true;
      }
    }
    return false;
  }
};
