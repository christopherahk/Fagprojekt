#include "FeatureExtractor.h"
#include "HoeffdingTree.h"
#include <Arduino.h>

// signal dimensions
const int N_CHANNELS = 56;
const int WINDOW = 32;
const int N_CLASSES = 3;
const int N_FLOATS = N_CHANNELS * WINDOW;
const int N_FEATURES = N_CHANNELS * 6;

const int BYTES_NEEDED = N_FLOATS * sizeof(float);
const int CHUNK_SIZE = 256;

// Class labels for serial output
const char *CLASS_NAMES[N_CLASSES] = {"dorsi", "plantar", "none"};

// Raw signal window
static float values[N_FLOATS];

// Feature vector
static float features[N_FEATURES];

// Probability output
static float proba[3][N_CLASSES];
static float avgProba[N_CLASSES];

static uint32_t labeledSamples = 0;
static uint32_t correctLabeledSamples = 0;

// model setup
// delta = 0.05 : split when 95% statistically confident
// tau   = 0.05 : also split if the top two features are within 0.05 of each
// testing less conservative measures
// other + epsilon has fallen below tau
// Ensemble: 3 HAT instances (prototype small ensemble)
HoeffdingAdaptiveTree hat0(N_FEATURES, N_CLASSES, /*delta=*/0.1f,
                           /*tau=*/0.1f);
HoeffdingAdaptiveTree hat1(N_FEATURES, N_CLASSES, /*delta=*/0.1f,
                           /*tau=*/0.1f);
HoeffdingAdaptiveTree hat2(N_FEATURES, N_CLASSES, /*delta=*/0.1f,
                           /*tau=*/0.1f);

// helpers to iterate
HoeffdingAdaptiveTree *hats[3] = {&hat0, &hat1, &hat2};

// Python side sends raw float bytes in CHUNK_SIZE chunks
// void get_data() {
//   int received = 0;
//   uint8_t *buf = reinterpret_cast<uint8_t *>(values);
//   int chunk_count = 0;

//   while (received < BYTES_NEEDED) {
//     int to_read = min(CHUNK_SIZE, BYTES_NEEDED - received);
//     unsigned long start = millis();

//     while (Serial.available() < to_read) {
//       if (millis() - start > 5000) {
//         Serial.print("TIMEOUT at chunk ");
//         Serial.println(chunk_count);
//         return;
//       }
//     }

//     for (int i = 0; i < to_read; i++)
//       buf[received++] = Serial.read();

//     chunk_count++;
//     Serial.println("ACK");
//   }

void get_data() {
  int received = 0;
  uint8_t *buf = reinterpret_cast<uint8_t *>(values);

  while (received < BYTES_NEEDED) {
    int to_read = min(CHUNK_SIZE, BYTES_NEEDED - received);
    unsigned long start = millis();

    while (Serial.available() < to_read) {
      if (millis() - start > 5000) {
        Serial.print("TIMEOUT at chunk ");
        Serial.println(received);
        return;
      }
    }

    for (int i = 0; i < to_read; i++)
      buf[received++] = Serial.read();

    // Serial.println("ACK"); <-- FJERNET! Sparer oceaner af tid.
  }

  // Read the mode byte after the payload:
  // 'L' = labeled sample, 'U' = unlabeled inference-only sample.
  unsigned long modeStart = millis();
  while (!Serial.available()) {
    if (millis() - modeStart > 2000) {
      processWindow(values, -1);
      Serial.println("INFER");
      return;
    }
  }

  char mode = (char)Serial.read();

  if (mode == 'U') {
    processWindow(values, -1);
    Serial.println("INFER");
    return;
  }

  if (mode != 'L') {
    processWindow(values, -1);
    Serial.println("INVALID_MODE");
    return;
  }
  Serial.print("DBG mode=");
  Serial.println(mode);

  int label[N_CLASSES];
  for (int i = 0; i < N_CLASSES; i++) {
    unsigned long start = millis();
    while (!Serial.available()) {
      if (millis() - start > 2000) {
        processWindow(values, -1);
        Serial.println("INVALID_LABEL");
        return;
      }
    }
    label[i] = Serial.read();
  }

  int labelIdx = -1;
  for (int i = 0; i < N_CLASSES; i++) {
    if (label[i] == 1) {
      labelIdx = i;
      break;
    }
  }

  if (labelIdx < 0) {
    processWindow(values, -1);
    Serial.println("INVALID_LABEL");
    return;
  }

  processWindow(values, labelIdx);
  Serial.println("TRAIN");
}

void processWindow(const float *data, int labelIdx) {
  // feature extraction
  Serial.print("DBG sample ");
  Serial.println(labelIdx);
  extractFeatures(data, N_CHANNELS, WINDOW, features);

  // prediction before training: per-tree
  int preds[3];
  for (int t = 0; t < 3; t++) {
    hats[t]->predictProba(features, proba[t]);
    preds[t] = hats[t]->predict(features);
  }

  // average probabilities
  for (int c = 0; c < N_CLASSES; c++) {
    avgProba[c] = 0.0f;
    for (int t = 0; t < 3; t++)
      avgProba[c] += proba[t][c];
    avgProba[c] /= 3.0f;
  }

  // majority vote
  int votes[N_CLASSES] = {0};
  for (int t = 0; t < 3; t++)
    votes[preds[t]]++;

  int ensemblePred = 0;
  for (int c = 1; c < N_CLASSES; c++)
    if (votes[c] > votes[ensemblePred])
      ensemblePred = c;

  bool isLabeled = labelIdx >= 0 && labelIdx < N_CLASSES;
  if (isLabeled) {
    static uint8_t bagIdx = 0;

    for (int t = 0; t < 3; t++) {

      if (t != bagIdx % 3) {
        hats[t]->train(features, labelIdx);
      }
    }
    bagIdx++;

    labeledSamples++;
    if (ensemblePred == labelIdx)
      correctLabeledSamples++;
  }

  //
  Serial.print("Probs: ");
  for (int i = 0; i < N_CLASSES; i++) {
    Serial.print(avgProba[i], 4);
    if (i < N_CLASSES - 1)
      Serial.print(", ");
  }
  Serial.println();
  Serial.print("Pred: ");
  Serial.print(CLASS_NAMES[ensemblePred]);
  if (isLabeled) {
    Serial.print(" | True: ");
    Serial.println(CLASS_NAMES[labelIdx]);
  } else {
    Serial.println(" | True: <none>");
  }

  // Tree size diagnostics
  Serial.print("Leaves: ");
  Serial.print(hat0.leafCount());
  Serial.print(",");
  Serial.print(hat1.leafCount());
  Serial.print(",");
  Serial.print(hat2.leafCount());
  Serial.print(" | Internals: ");
  Serial.print(hat0.internalCount());
  Serial.print(",");
  Serial.print(hat1.internalCount());
  Serial.print(",");
  Serial.println(hat2.internalCount());
  Serial.print("Resets: ");
  Serial.print(hat0.resetCount());
  Serial.print(",");
  Serial.print(hat1.resetCount());
  Serial.print(",");
  Serial.println(hat2.resetCount());

  if (isLabeled) {
    float runningAcc =
        labeledSamples > 0
            ? 100.0f * (float)correctLabeledSamples / (float)labeledSamples
            : 0.0f;
    Serial.print("Acc: ");
    Serial.print(runningAcc, 2);
    Serial.print("% (");
    Serial.print(correctLabeledSamples);
    Serial.print("/");
    Serial.print(labeledSamples);
    Serial.println(")");
    Serial.println(ensemblePred == labelIdx ? "CORRECT" : "WRONG");
  } else {
    Serial.println("UNLABELED");
  }
}

void setup() {
  Serial.begin(115200); // same as streamer.py BAUD

  unsigned long usb_timeout = millis();
  while (!Serial) {
    if (millis() - usb_timeout > 4000)
      break;
  }

  delay(1000);
  while (Serial.available()) {
    Serial.read();
  }

  while (true) {
    if (Serial.available() > 0) {
      char c = Serial.read();
      if (c == 'X') {

        Serial.println("READY");
        break;
      }
    }
    delay(10);
  }

  while (Serial.available()) {
    Serial.read();
  }
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'D') {

      delay(2);
      get_data();
    } else if (command == 'E') {

      unsigned long start = millis();
      while (!Serial.available()) {
        if (millis() - start > 200)
          break;
      }
      if (Serial.available() && Serial.read() == 'X') {
        Serial.println("START_EXPORT");
        Serial.println("#ifndef TRAINED_WEIGHTS_H");
        Serial.println("#define TRAINED_WEIGHTS_H");
        hat0.exportSnapshot("ensemble_tree_0");
        hat1.exportSnapshot("ensemble_tree_1");
        hat2.exportSnapshot("ensemble_tree_2");
        Serial.println("#endif");
        Serial.println("END_EXPORT");
        while (true) {
          if (Serial.available() && Serial.read() == 'R')
            break;
        }
      }
    }
  }
}

// fork + knife
