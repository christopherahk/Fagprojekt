#pragma once
#include <stdint.h>
#include <math.h>
#include <string.h>
// represents one DT node.
// splt_feature: -1 -> leaf node, pred is class index
const int HT_N_CLASSES = 3; // number of classes in the classification task
const int HT_N_FEATURES = 224; // number of features extracted from the signal
const int HT_MAX_NODES = 64;    // maximum number of nodes in a tree 
const int HT_MAX_LEAVES = 20;   // 
const int HT_GRACE_PERIOD = 50; // how many samples to observe before trying to split a leaf
const float HT_DELTA = 0.01f;   // confidence level for Hoeffding bound (1 - delta is confidence)
const float HT_TIE_THRESHOLD = 0.05f;

struct LeafStats {
    uint32_t nSamples[HT_N_CLASSES];
    float sum[HT_N_FEATURES][HT_N_CLASSES]; 
    float sumSq[HT_N_FEATURES][HT_N_CLASSES]; // for variance
    uint32_t totalSamples;

    void reset() {
        memset(this, 0, sizeof(LeafStats));
    }
};
struct HoeffdingNode {
    int16_t split_feature = -1; // -1 = Blad
    float threshold = 0.0f;
    int16_t left = -1;
    int16_t right = -1;
    LeafStats* stats = nullptr;
};

class HoeffdingTree {
private:
    HoeffdingNode nodes[HT_MAX_NODES];
    LeafStats leafPool[HT_MAX_LEAVES];
    int nodeCount = 1;
    int leafCount = 0;

    // compute Hoeffding Bound: epsilon = sqrt( (R^2 * ln(1/delta)) / (2 * n) )
    float calculateHoeffdingBound(uint32_t n) {
        float R = 1.0f; // For normalized features.
        return sqrtf((R * R * logf(1.0f / HT_DELTA)) / (2.0f * n));
    }

    // Gini Impurity for a leaf node (split evaluation)
    float calculateGini(const LeafStats* s) {
        if (s->totalSamples == 0) return 1.0f;
        float impurity = 1.0f;
        for (int i = 0; i < HT_N_CLASSES; i++) {
            float p = (float)s->nSamples[i] / s->totalSamples;
            impurity -= p * p;
        }
        return impurity;
    }

    float calculateEstimatedGain(const LeafStats* s, int featureIdx, float threshold, float currentGini) {
        uint32_t leftCount = 0;
        uint32_t rightCount = 0; // Her sikrer vi, at den er defineret
        uint32_t leftSamples[HT_N_CLASSES];
        uint32_t rightSamples[HT_N_CLASSES];

        // Initialiser arrays til nul
        for(int i = 0; i < HT_N_CLASSES; i++) {
            leftSamples[i] = 0;
            rightSamples[i] = 0;
        }

        for (int c = 0; c < HT_N_CLASSES; c++) {
            if (s->nSamples[c] == 0) continue;

            // Simpel heuristik: hvor ligger klassens gennemsnit i forhold til tærsklen?
            float classMean = s->sum[featureIdx][c] / s->nSamples[c];
            
            // Vi estimerer en fordeling (fx 80% til den dominerende side)
            float ratioLeft = (classMean <= threshold) ? 0.8f : 0.2f;
            
            leftSamples[c] = (uint32_t)(s->nSamples[c] * ratioLeft);
            rightSamples[c] = s->nSamples[c] - leftSamples[c];
            
            leftCount += leftSamples[c];
            rightCount += rightSamples[c];
        }

        // Tjek om splittet overhovedet deler dataen
        if (leftCount == 0 || rightCount == 0) return 0.0f;

        float giniLeft = 1.0f;
        float giniRight = 1.0f;

        for (int c = 0; c < HT_N_CLASSES; c++) {
            float pL = (float)leftSamples[c] / leftCount;
            float pR = (float)rightSamples[c] / rightCount;
            giniLeft -= pL * pL;
            giniRight -= pR * pR;
        }

        // Vægtet Gini-gennemsnit for de to nye potentielle grene
        float weightedGini = ((float)leftCount / s->totalSamples) * giniLeft + 
                             ((float)rightCount / s->totalSamples) * giniRight;

        return currentGini - weightedGini;
    }

public:
    HoeffdingTree() {
        nodes[0].stats = &leafPool[leafCount++];
        nodes[0].stats->reset();
    }

    // prediction (Inference)
    uint8_t predict(const float* features) {
        int curr = 0;
        while (nodes[curr].split_feature != -1) {
            if (features[nodes[curr].split_feature] <= nodes[curr].threshold) {
                curr = nodes[curr].left;
            } else {
                curr = nodes[curr].right;
            }
        }

        // find most common class in the leaf
        uint8_t bestClass = 0;
        for (int c = 1; c < HT_N_CLASSES; c++) {
            if (nodes[curr].stats->nSamples[c] > nodes[curr].stats->nSamples[bestClass]) {
                bestClass = c;
            }
        }
        return bestClass;
    }

    // Live Training
    void train(const float* features, uint8_t label) {
        int curr = 0;
        // 1. Find correct leaf for the sample
        while (nodes[curr].split_feature != -1) {
            if (features[nodes[curr].split_feature] <= nodes[curr].threshold) {
                curr = nodes[curr].left;
            } else {
                curr = nodes[curr].right;
            }
        }

        // 2. Update statistics in the leaf
        LeafStats* s = nodes[curr].stats;
        s->nSamples[label]++;
        s->totalSamples++;
        for (int i = 0; i < HT_N_FEATURES; i++) {
            s->sum[i][label] += features[i];
            s->sumSq[i][label] += features[i] * features[i];
        }

        // 3. check for split post grace period
        if (s->totalSamples >= HT_GRACE_PERIOD && leafCount < HT_MAX_LEAVES && nodeCount < HT_MAX_NODES - 2) {
            attemptSplit(curr);
        }
    }
    




    void attemptSplit(int nodeIdx) {
    LeafStats* s = nodes[nodeIdx].stats;
    float epsilon = calculateHoeffdingBound(s->totalSamples);
    
    int bestFeature = -1;
    int secondBestFeature = -1;
    float bestGiniGain = -1.0f;
    float secondBestGiniGain = -1.0f;
    float currentGini = calculateGini(s);

    for (int f = 0; f < HT_N_FEATURES; f++) {
        // Vi approksimerer tærsklen som middelværdien af featuren i dette blad
        float thresholdAttempt = 0;
        for(int c=0; c < HT_N_CLASSES; c++) thresholdAttempt += s->sum[f][c];
        thresholdAttempt /= s->totalSamples;

        // Beregn Gini Gain for denne feature (meget forenklet)
        // I praksis skal du bruge statistikken til at estimere hvor mange 
        // samples der falder hhv. over og under middelværdien.
        float gain = calculateEstimatedGain(s, f, thresholdAttempt, currentGini);

        if (gain > bestGiniGain) {
            secondBestGiniGain = bestGiniGain;
            secondBestFeature = bestFeature;
            bestGiniGain = gain;
            bestFeature = f;
        } else if (gain > secondBestGiniGain) {
            secondBestGiniGain = gain;
            secondBestFeature = f;
        }
    }

    // Hoeffding Bound Check: Er den bedste feature statistisk signifikant bedre?
    if (bestFeature != -1 && (bestGiniGain - secondBestGiniGain > epsilon || epsilon < HT_TIE_THRESHOLD)) {
        float finalThresh = 0;
        for(int c=0; c < HT_N_CLASSES; c++) finalThresh += s->sum[bestFeature][c];
        splitNode(nodeIdx, bestFeature, finalThresh / s->totalSamples);
    }
}

    void splitNode(int idx, int feature, float thresh) {
        int leftIdx = nodeCount++;
        int rightIdx = nodeCount++;

        nodes[idx].split_feature = feature;
        nodes[idx].threshold = thresh;
        nodes[idx].left = leftIdx;
        nodes[idx].right = rightIdx;

        // Tildel nye blade fra poolen
        nodes[leftIdx].stats = &leafPool[leafCount++];
        nodes[leftIdx].stats->reset();
        nodes[rightIdx].stats = &leafPool[leafCount++];
        nodes[rightIdx].stats->reset();
        
        // Bemærk: Den gamle statistik i nodes[idx].stats "tabes" her for at spare RAM,
        // da vi nu bruger de to nye blade i stedet.
    }
};
