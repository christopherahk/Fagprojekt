#pragma once
#include <math.h>
#include <stdint.h>

// Mondrian Forest -- online random forest for embedded hardware
// =============================================================
//
// Works with RBI+FFT features: 56 * (4 + 17) = 1176 features.
// Memory at N_TREES=10, MAX_NODES=63: ~80 KB -- fits in 256 KB.
//
// Key properties vs HAT:
//   - Splits chosen randomly (Poisson process), not by Gini search.
//     This makes online updates O(depth) with no dataset buffering.
//   - Class counts kept at every node, not just leaves.
//     Prediction uses the leaf's smoothed class distribution.
//   - No Hoeffding bound -- splits happen based on budget (lambda).
//     Higher lambda = deeper trees.
//
// Usage:
//   MondrianForest mf;
//   mf.train(features, trueLabel);
//   int pred = mf.predict(features);
//   mf.predictProba(features, proba);

// ── Configuration ────────────────────────────────────────────────────────────
#ifndef MF_N_TREES
#define MF_N_TREES 10 // ensemble size
#endif
#ifndef MF_MAX_NODES
#define MF_MAX_NODES 63 // depth-5 full binary tree: 2^6 - 1
#endif
#ifndef MF_N_FEATURES
#define MF_N_FEATURES 1176 // 56 * (4 RBI + 17 FFT)
#endif
#ifndef MF_N_CLASSES
#define MF_N_CLASSES 3
#endif
// Mondrian budget: controls max tree depth.
// Higher = deeper trees. Start at 5.0, tune between 3.0 and 10.0.
#ifndef MF_LAMBDA
#define MF_LAMBDA 5.0f
#endif
// Laplace smoothing for probability estimates
#ifndef MF_SMOOTH
#define MF_SMOOTH 1.0f
#endif

// ── Q8.8 fixed-point (range [-128, 127.996], precision 1/256) ────────────────
static inline int16_t mf_toQ88(float f) {
  float c = f < -128.0f ? -128.0f : (f > 127.996f ? 127.996f : f);
  return (int16_t)(c * 256.0f);
}
static inline float mf_fromQ88(int16_t q) { return q / 256.0f; }

// ── LCG random number generator (no stdlib needed) ───────────────────────────
static uint32_t mf_rng_ = 12345u;
static inline void mf_seed(uint32_t s) { mf_rng_ = s; }
static inline float mf_rand() {
  mf_rng_ = mf_rng_ * 1664525u + 1013904223u;
  return (float)(mf_rng_ >> 8) / (float)(1u << 24);
}
static inline float mf_rand_exp(float rate) {
  float u = mf_rand();
  if (u < 1e-7f)
    u = 1e-7f;
  return -logf(u) / rate;
}

// ── Node ─────────────────────────────────────────────────────────────────────
struct MFNode {
  int16_t splitDim; // feature index, -1 = unsplit leaf
  float splitThreshold;
  float tau; // Mondrian split time
  int16_t left, right, parent;
  float classCounts[MF_N_CLASSES];
  float nSamples;
};

// ── Tree ─────────────────────────────────────────────────────────────────────
struct MFTree {
  MFNode nodes[MF_MAX_NODES];
  int16_t nNodes;
  // Feature bounding box at root level (Q8.8 to save memory)
  int16_t fMin[MF_N_FEATURES];
  int16_t fMax[MF_N_FEATURES];
};

// ── MondrianForest
// ────────────────────────────────────────────────────────────
class MondrianForest {
public:
  MondrianForest() : totalSamples_(0) {
    for (int t = 0; t < MF_N_TREES; t++) {
      trees_[t].nNodes = 0;
      for (int f = 0; f < MF_N_FEATURES; f++) {
        trees_[t].fMin[f] = mf_toQ88(127.0f);
        trees_[t].fMax[f] = mf_toQ88(-128.0f);
      }
      allocNode(t, -1);
    }
  }

  // Train on one labelled sample (test-then-train should be done by caller)
  void train(const float *features, int label) {
    for (int t = 0; t < MF_N_TREES; t++) {
      // Different seed per tree per sample for diversity
      mf_seed((uint32_t)(t * 1664525u ^ totalSamples_ * 22695477u));
      updateRanges(t, features);
      updateTree(t, 0, features, label, 0.0f);
    }
    totalSamples_++;
  }

  int predict(const float *features) const {
    float p[MF_N_CLASSES] = {};
    predictProba(features, p);
    int best = 0;
    for (int c = 1; c < MF_N_CLASSES; c++)
      if (p[c] > p[best])
        best = c;
    return best;
  }

  void predictProba(const float *features, float *proba) const {
    for (int c = 0; c < MF_N_CLASSES; c++)
      proba[c] = 0.0f;
    float tp[MF_N_CLASSES];
    for (int t = 0; t < MF_N_TREES; t++) {
      treePredict(t, features, tp);
      for (int c = 0; c < MF_N_CLASSES; c++)
        proba[c] += tp[c];
    }
    for (int c = 0; c < MF_N_CLASSES; c++)
      proba[c] /= MF_N_TREES;
  }

  int totalNodes() const {
    int s = 0;
    for (int t = 0; t < MF_N_TREES; t++)
      s += trees_[t].nNodes;
    return s;
  }
  int samplesSeen() const { return totalSamples_; }

private:
  MFTree trees_[MF_N_TREES];
  int totalSamples_;

  int allocNode(int t, int parent) {
    MFTree &tree = trees_[t];
    if (tree.nNodes >= MF_MAX_NODES)
      return -1;
    int idx = tree.nNodes++;
    MFNode &n = tree.nodes[idx];
    n.splitDim = -1;
    n.splitThreshold = 0.0f;
    n.tau = 0.0f;
    n.left = -1;
    n.right = -1;
    n.parent = (int16_t)parent;
    n.nSamples = 0.0f;
    for (int c = 0; c < MF_N_CLASSES; c++)
      n.classCounts[c] = 0.0f;
    return idx;
  }

  void updateRanges(int t, const float *features) {
    MFTree &tree = trees_[t];
    for (int f = 0; f < MF_N_FEATURES; f++) {
      if (features[f] < mf_fromQ88(tree.fMin[f]))
        tree.fMin[f] = mf_toQ88(features[f]);
      if (features[f] > mf_fromQ88(tree.fMax[f]))
        tree.fMax[f] = mf_toQ88(features[f]);
    }
  }

  // Recursive Mondrian tree update.
  // At each node: update counts, then either recurse or try to split.
  void updateTree(int t, int nodeIdx, const float *features, int label,
                  float parentTau) {
    if (nodeIdx < 0 || nodeIdx >= trees_[t].nNodes)
      return;
    MFTree &tree = trees_[t];
    MFNode &node = tree.nodes[nodeIdx];

    // Update class counts on the path
    node.classCounts[label] += 1.0f;
    node.nSamples += 1.0f;

    if (node.splitDim < 0) {
      // Leaf: try Mondrian extension
      trySplit(t, nodeIdx, features, label, parentTau);
    } else {
      // Internal: recurse into appropriate child
      int next = features[node.splitDim] <= node.splitThreshold ? node.left
                                                                : node.right;
      updateTree(t, next, features, label, node.tau);
    }
  }

  void trySplit(int t, int leafIdx, const float *features, int label,
                float parentTau) {
    MFTree &tree = trees_[t];
    if (tree.nNodes + 2 > MF_MAX_NODES)
      return;

    // Compute extension of this point outside current bounding box
    float totalExt = 0.0f;
    for (int f = 0; f < MF_N_FEATURES; f++) {
      float lo = mf_fromQ88(tree.fMin[f]);
      float hi = mf_fromQ88(tree.fMax[f]);
      if (features[f] < lo)
        totalExt += lo - features[f];
      else if (features[f] > hi)
        totalExt += features[f] - hi;
    }
    if (totalExt < 1e-6f)
      return;

    // Sample split time -- only split if within budget
    float splitTime = parentTau + mf_rand_exp(totalExt);
    if (splitTime > MF_LAMBDA)
      return;

    // Choose split dimension proportional to extension
    float u = mf_rand() * totalExt;
    int splitDim = 0;
    float cumExt = 0.0f;
    for (int f = 0; f < MF_N_FEATURES; f++) {
      float lo = mf_fromQ88(tree.fMin[f]);
      float hi = mf_fromQ88(tree.fMax[f]);
      float ext = 0.0f;
      if (features[f] < lo)
        ext = lo - features[f];
      else if (features[f] > hi)
        ext = features[f] - hi;
      cumExt += ext;
      if (cumExt >= u) {
        splitDim = f;
        break;
      }
    }

    // Threshold: uniformly in the extension region
    float lo = mf_fromQ88(tree.fMin[splitDim]);
    float hi = mf_fromQ88(tree.fMax[splitDim]);
    float ext = 0.0f;
    float threshold;
    if (features[splitDim] < lo) {
      ext = lo - features[splitDim];
      threshold = features[splitDim] + mf_rand() * ext;
    } else if (features[splitDim] > hi) {
      ext = features[splitDim] - hi;
      threshold = hi + mf_rand() * ext;
    } else {
      threshold = lo + mf_rand() * (hi - lo);
    }

    // Allocate children
    int leftIdx = allocNode(t, leafIdx);
    int rightIdx = allocNode(t, leafIdx);
    if (leftIdx < 0 || rightIdx < 0)
      return;

    // Distribute existing counts to children
    MFNode &leaf = tree.nodes[leafIdx];
    for (int c = 0; c < MF_N_CLASSES; c++) {
      float newCount = (c == label) ? 1.0f : 0.0f;
      float oldCount = leaf.classCounts[c] - newCount;
      if (oldCount < 0.0f)
        oldCount = 0.0f;

      if (features[splitDim] <= threshold) {
        tree.nodes[leftIdx].classCounts[c] = newCount;
        tree.nodes[rightIdx].classCounts[c] = oldCount;
      } else {
        tree.nodes[rightIdx].classCounts[c] = newCount;
        tree.nodes[leftIdx].classCounts[c] = oldCount;
      }
    }
    float oldN = leaf.nSamples - 1.0f;
    if (oldN < 0.0f)
      oldN = 0.0f;
    if (features[splitDim] <= threshold) {
      tree.nodes[leftIdx].nSamples = 1.0f;
      tree.nodes[rightIdx].nSamples = oldN;
    } else {
      tree.nodes[rightIdx].nSamples = 1.0f;
      tree.nodes[leftIdx].nSamples = oldN;
    }

    // Convert leaf to internal node
    leaf.splitDim = (int16_t)splitDim;
    leaf.splitThreshold = threshold;
    leaf.tau = splitTime;
    leaf.left = (int16_t)leftIdx;
    leaf.right = (int16_t)rightIdx;
  }

  void treePredict(int t, const float *features, float *proba) const {
    const MFTree &tree = trees_[t];
    int nodeIdx = 0;
    while (nodeIdx >= 0 && nodeIdx < tree.nNodes) {
      const MFNode &n = tree.nodes[nodeIdx];
      if (n.splitDim < 0)
        break;
      nodeIdx = features[n.splitDim] <= n.splitThreshold ? n.left : n.right;
      if (nodeIdx < 0 || nodeIdx >= tree.nNodes) {
        nodeIdx = 0;
        break;
      }
    }
    const MFNode &leaf = tree.nodes[nodeIdx];
    float total = leaf.nSamples + MF_SMOOTH * MF_N_CLASSES;
    for (int c = 0; c < MF_N_CLASSES; c++)
      proba[c] = (leaf.classCounts[c] + MF_SMOOTH) / total;
  }
};
