#pragma once
#include "ADWIN.h"
#include <Arduino.h>
#include <math.h>
#include <stdint.h>

// Hoeffding Adaptive Tree (HAT)
// learns from one sample at a time with no buffering
// internal nodes and leaf nodes are stored in seperate static arrays

class HoeffdingAdaptiveTree {
public:
  // nFeatures : features per sample  (N_CHANNELS * 4 = 224)
  // nClasses  : number of classes    (3)
  // delta     : Hoeffding confidence. Lower = splits more cautiously 0.05 = 95%
  // confidence tau       : tie breaking threshold. If the top two candidate
  // features are within tau of each other in Gini gain, we split when epsilon
  // falls below tau.
  HoeffdingAdaptiveTree(int nFeatures, int nClasses, float delta = 0.05f,
                        float tau = 0.05f)
      : nFeatures_(nFeatures), nClasses_(nClasses), delta_(delta), tau_(tau),
        nInternals_(0), nLeaves_(0), internals_{}, leaves_{} {
    // single root start idx 0
    allocLeaf(NO_PARENT);
  }
  int resetCount() const { return resetCount_; }

  // train on one labelled sample
  // predict first
  // then update the leaf statistics and try split
  void train(const float *features, int trueLabel) {
    int li = traverseToLeaf(features);

    Leaf &lf = leaves_[li];

    // update class counts
    lf.classCounts[trueLabel] += 1.0f;
    lf.samplesAtLeaf++;

    // drift check, feed whether the current majority vote is wrong
    // Testing disabled ADWIN
    int pred = majorityClass(li);
    bool drifted = lf.adwin.update(pred != trueLabel);

    if (drifted) {
      resetLeaf(li);
      return;
    }

    // update online feature means
    updateFeatureMeans(li, features, trueLabel);

    //
    if (lf.samplesAtLeaf >=
        MIN_SAMPLES_SPLIT) { // removed && lf.samplesAtLeaf % 50 == 0 to check
                             // for split more often
      trySplit(li);
    } // comment this to make it a naive bayes classifier without splits
  }

  // Return the predicted class index for a sample.
  int predict(const float *features) const {
    return majorityClass(traverseToLeaf(features));
  }

  // fill proba[nClasses] with normalised class probabilities.
  void predictProba(const float *features, float *proba) const {
    int li = traverseToLeaf(features);
    const Leaf &lf = leaves_[li];
    float total = 0.0f;
    for (int c = 0; c < nClasses_; c++)
      total += lf.classCounts[c];
    for (int c = 0; c < nClasses_; c++)
      proba[c] = (total > 0.0f) ? lf.classCounts[c] / total : 0.0f;
  }

  int leafCount() const { return nLeaves_; }
  int internalCount() const { return nInternals_; }

  void exportSnapshot(const char *treeName) const {
    Serial.print("# Tree snapshot: ");
    Serial.println(treeName);

    Serial.print("# nFeatures=");
    Serial.print(nFeatures_);
    Serial.print(" nClasses=");
    Serial.print(nClasses_);
    Serial.print(" nLeaves=");
    Serial.print(nLeaves_);
    Serial.print(" nInternals=");
    Serial.println(nInternals_);

    for (int i = 0; i < nInternals_; i++) {
      const Internal &nd = internals_[i];
      Serial.print("INTERNAL ");
      Serial.print(i);
      Serial.print(" splitFeature=");
      Serial.print(nd.splitFeature);
      Serial.print(" splitThreshold=");
      Serial.print(nd.splitThreshold, 7);
      Serial.print(" left=");
      Serial.print(nd.left);
      Serial.print(" right=");
      Serial.print(nd.right);
      Serial.print(" leftIsLeaf=");
      Serial.print(nd.leftIsLeaf ? 1 : 0);
      Serial.print(" rightIsLeaf=");
      Serial.println(nd.rightIsLeaf ? 1 : 0);
    }

    for (int i = 0; i < nLeaves_; i++) {
      const Leaf &lf = leaves_[i];
      Serial.print("LEAF ");
      Serial.print(i);
      Serial.print(" parentInternal=");
      Serial.print(lf.parentInternal);
      Serial.print(" isRightChild=");
      Serial.print(lf.isRightChild ? 1 : 0);
      Serial.print(" samplesAtLeaf=");
      Serial.print(lf.samplesAtLeaf);
      Serial.print(" classCounts=");
      for (int c = 0; c < nClasses_; c++) {
        Serial.print(lf.classCounts[c], 7);
        if (c < nClasses_ - 1)
          Serial.print(",");
      }
      Serial.print(" featureMeanFirstClass=");
      for (int f = 0; f < nFeatures_; f++) {
        Serial.print(fromQ88(lf.featureMean[f][0]), 7);
        if (f < nFeatures_ - 1)
          Serial.print(",");
      }
      Serial.println();
    }
  }

private:
  // LIMITED!!!!!!!!! POWEEEEERRRRRRR!!!!!!!
  // less leaves/internals for small ensemble
  static const int MAX_LEAVES = 12;    // i think 20 is enough but testing 30
  static const int MAX_INTERNALS = 11; // should be 1 less than max leaves
  static const int MAX_FEATURES = 336;
  static const int MAX_CLASSES = 3;
  static const int NO_PARENT = -1;
  static const int MIN_SAMPLES_SPLIT = 30; // changed from 50
  int resetCount_ = 0;

  static int16_t toQ88(float f) {
    float clamped = f < -128.0f ? -128.0f : (f > 127.996f ? 127.996f : f);
    return (int16_t)(clamped * 256.0f);
  }
  static float fromQ88(int16_t q) { return q / 256.0f; }

  //
  struct Internal {
    int16_t splitFeature;
    float splitThreshold;
    int16_t left;
    int16_t right;
    int16_t parent; // internal index of parent, or NO_PARENT for root
    bool leftIsLeaf;
    bool rightIsLeaf;
  };

  //
  struct Leaf {
    int16_t parentInternal;
    bool isRightChild;

    float classCounts[MAX_CLASSES];
    int16_t samplesAtLeaf;

    int16_t featureMean[MAX_FEATURES][MAX_CLASSES];
    int16_t classCount[MAX_CLASSES]; // samples seen per class

    ADWIN adwin;
  };

  // static arrays for internals and leaves
  Internal internals_[MAX_INTERNALS];
  Leaf leaves_[MAX_LEAVES];

  int nInternals_;
  int nLeaves_;

  int nFeatures_;
  int nClasses_;
  float delta_;
  float tau_;

  // returns leaf index, or 0 (root) if the pool is full.
  int allocLeaf(int parentInternal, bool isRightChild = false) {
    if (nLeaves_ >= MAX_LEAVES)
      return 0;
    int li = nLeaves_++;
    Leaf &lf = leaves_[li];

    lf.parentInternal = (int16_t)parentInternal;
    lf.isRightChild = isRightChild;
    lf.samplesAtLeaf = 0;

    for (int c = 0; c < nClasses_; c++) {
      lf.classCounts[c] = 0.0f;
      lf.classCount[c] = 0;
    }
    for (int f = 0; f < nFeatures_; f++)
      for (int c = 0; c < nClasses_; c++)
        lf.featureMean[f][c] = 0;

    return li;
  }

  // allocate a new internal node
  // returns internal index, or -1 if pool is full
  int allocInternal(int parentInternal) {
    if (nInternals_ >= MAX_INTERNALS)
      return -1;
    int ii = nInternals_++;
    internals_[ii].splitFeature = -1;
    internals_[ii].parent = (int16_t)parentInternal;
    return ii;
  }

  // walk the tree from the root and return leaf index that the feature vector
  // routes to
  int traverseToLeaf(const float *features) const {
    // If the tree has no internal nodes yet
    if (nInternals_ == 0)
      return 0;

    // Start at internal node idx 0
    int ii = 0;
    while (true) {
      const Internal &nd = internals_[ii];
      bool goLeft = features[nd.splitFeature] <= nd.splitThreshold;

      if (goLeft) {
        if (nd.leftIsLeaf)
          return nd.left;
        ii = nd.left;
      } else {
        if (nd.rightIsLeaf)
          return nd.right;
        ii = nd.right;
      }
    }
  }

  int majorityClass(int li) const {
    const Leaf &lf = leaves_[li];
    int best = 0;
    for (int c = 1; c < nClasses_; c++)
      if (lf.classCounts[c] > lf.classCounts[best])
        best = c;
    return best;
  }

  void updateFeatureMeans(int li, const float *features, int label) {
    Leaf &lf = leaves_[li];
    lf.classCount[label]++;
    int n = lf.classCount[label];

    for (int f = 0; f < nFeatures_; f++) {
      float oldMean = fromQ88(lf.featureMean[f][label]);
      float newMean = oldMean + (features[f] - oldMean) / (float)n;
      lf.featureMean[f][label] = toQ88(newMean);
    }
  }

  // (Gini impurity)
  // Estimate the Gini gain for a single feature using the per-class mean
  // values, as split-point candidates (midpoint between each pair of class
  // means). Returns the best Gini impurity found and writes the corresponding
  // threshold to bestThreshold.
  float giniForFeature(int li, int f, float &bestThreshold) const {
    const Leaf &lf = leaves_[li];

    float total = 0.0f;
    for (int c = 0; c < nClasses_; c++)
      total += lf.classCounts[c];
    if (total < 2.0f) {
      bestThreshold = 0.0f;
      return 1.0f;
    }

    float bestGini = 1.0f;
    bestThreshold = 0.0f;

    // try the midpoint between each pair of class means as a threshold
    for (int c1 = 0; c1 < nClasses_; c1++) {
      for (int c2 = c1 + 1; c2 < nClasses_; c2++) {
        float mu1 = fromQ88(lf.featureMean[f][c1]);
        float mu2 = fromQ88(lf.featureMean[f][c2]);
        float threshold = 0.5f * (mu1 + mu2);

        // count samples going left/right for each class
        float leftCounts[MAX_CLASSES] = {};
        float rightCounts[MAX_CLASSES] = {};
        for (int c = 0; c < nClasses_; c++) {
          float mu = fromQ88(lf.featureMean[f][c]);
          if (mu <= threshold)
            leftCounts[c] = lf.classCounts[c];
          else
            rightCounts[c] = lf.classCounts[c];
        }

        float leftTotal = 0.0f, rightTotal = 0.0f;
        for (int c = 0; c < nClasses_; c++) {
          leftTotal += leftCounts[c];
          rightTotal += rightCounts[c];
        }
        if (leftTotal < 1.0f || rightTotal < 1.0f)
          continue;

        float gL = 1.0f, gR = 1.0f;
        for (int c = 0; c < nClasses_; c++) {
          float pl = leftCounts[c] / leftTotal;
          float pr = rightCounts[c] / rightTotal;
          gL -= pl * pl;
          gR -= pr * pr;
        }

        float gini = (leftTotal * gL + rightTotal * gR) / total;
        if (gini < bestGini) {
          bestGini = gini;
          bestThreshold = threshold;
        }
      }
    }
    return bestGini;
  }

  // Hoeffding bound split decision

  void trySplit(int li) {
    if (nInternals_ >= MAX_INTERNALS)
      return;
    if (nLeaves_ + 1 >= MAX_LEAVES)
      return;

    float best1 = 1.0f, best2 = 1.0f;
    int bestF = -1;
    float bestThresh = 0.0f;

    // Evaluate only sqrt(nFeatures_) randomly chosen features
    // Uses a simple LCG random number generator -- no stdlib needed
    int nCandidates = 1;
    while (nCandidates * nCandidates < nFeatures_)
      nCandidates++; // ceil(sqrt)

    uint32_t rng =
        (uint32_t)(leaves_[li].samplesAtLeaf * 1664525u + 1013904223u);

    for (int k = 0; k < nCandidates; k++) {
      rng = rng * 1664525u + 1013904223u;
      int f = (int)(rng >> 16) % nFeatures_;

      float thresh;
      float g = giniForFeature(li, f, thresh);
      if (g < best1) {
        best2 = best1;
        best1 = g;
        bestF = f;
        bestThresh = thresh;
      } else if (g < best2) {
        best2 = g;
      }
    }

    if (bestF < 0)
      return;

    // Hoeffding bound: epsilon = sqrt(R^2 * ln(1/delta) / 2n)
    // R = 1 for Gini (range [0,1])
    float n = (float)leaves_[li].samplesAtLeaf;
    float eps = sqrtf(logf(1.0f / delta_) / (2.0f * n));

    float gap = best2 - best1;
    if (gap > eps || eps < tau_) {
      doSplit(li, bestF, bestThresh);
    }
  }

  // Convert the current leaf into an internal split node.
  // Allocates two new leaves as children and transfers the class count
  // distribution from the old leaf to the children.
  void doSplit(int li, int feature, float threshold) {
    Leaf &oldLeaf = leaves_[li];

    // allocate new internal node
    int ii = allocInternal(oldLeaf.parentInternal);
    if (ii < 0)
      return; // pool full, skip split

    Internal &nd = internals_[ii];
    nd.splitFeature = (int16_t)feature;
    nd.splitThreshold = threshold;

    // allocate two new leaves
    int leftLi = allocLeaf(ii, false);
    int rightLi = allocLeaf(ii, true);

    nd.left = (int16_t)leftLi;
    nd.right = (int16_t)rightLi;
    nd.leftIsLeaf = true;
    nd.rightIsLeaf = true;

    // distribute the old leafs class counts to the new children using the split
    // threshold on each class mean feature value
    for (int c = 0; c < nClasses_; c++) {
      float mu = fromQ88(oldLeaf.featureMean[feature][c]);
      if (mu <= threshold) {
        leaves_[leftLi].classCounts[c] = oldLeaf.classCounts[c];
        leaves_[leftLi].classCount[c] = oldLeaf.classCount[c];
      } else {
        leaves_[rightLi].classCounts[c] = oldLeaf.classCounts[c];
        leaves_[rightLi].classCount[c] = oldLeaf.classCount[c];
      }
    }

    // wire the parent internal node to point to the new internal node instead
    // of the old leaf, if old leaf had parent
    int parentII = oldLeaf.parentInternal;
    if (parentII != NO_PARENT && parentII >= 0) {
      Internal &parent = internals_[parentII];
      if (parent.leftIsLeaf && parent.left == li) {
        parent.left = (int16_t)ii;
        parent.leftIsLeaf = false;
      } else if (parent.rightIsLeaf && parent.right == li) {
        parent.right = (int16_t)ii;
        parent.rightIsLeaf = false;
      }
    }
  }

  // When ADWIN detects drift in a leaf, zero out all statistics so the leaf
  // starts learning from scratch with the new distribution the leaf stays in
  // the tree at the same position, no change needed, which keeps implementation
  // simple
  void resetLeaf(int li) {
    Leaf &lf = leaves_[li];
    lf.adwin.reset();
    lf.samplesAtLeaf = 0;
    resetCount_++;

    for (int c = 0; c < nClasses_; c++) {
      lf.classCounts[c] = 0.0f;
      lf.classCount[c] = 0;
    }
    for (int f = 0; f < nFeatures_; f++)
      for (int c = 0; c < nClasses_; c++)
        lf.featureMean[f][c] = 0;
  }
};
