#include "FeatureExtractor.h"
#include "HoeffdingTree.h"
#include <Arduino.h>

// signal dimensions
const int N_CHANNELS = 56;
const int WINDOW = 20;
const int N_CLASSES = 3;
const int N_FLOATS = N_CHANNELS * WINDOW;
const int N_FEATURES = N_CHANNELS * 4;

const int BYTES_NEEDED = N_FLOATS * sizeof(float);
const int CHUNK_SIZE = 64;

// Class labels for serial output
const char *CLASS_NAMES[N_CLASSES] = {"dorsi", "plantar", "none"};

// Raw signal window
static float values[N_FLOATS];

// Feature vector
static float features[N_FEATURES];

// Probability output
static float proba[N_CLASSES];

static uint32_t labeledSamples = 0;
static uint32_t correctLabeledSamples = 0;

// model setup
// delta = 0.05 : split when 95% statistically confident
// tau   = 0.05 : also split if the top two features are within 0.05 of each
// other + epsilon has fallen below tau
HoeffdingAdaptiveTree hat(N_FEATURES, N_CLASSES,
                          /*delta=*/0.05f,
                          /*tau=*/0.05f);

// Python side sends raw float bytes in CHUNK_SIZE chunks
void get_data() {
  int received = 0;
  uint8_t *buf = reinterpret_cast<uint8_t *>(values);
  int chunk_count = 0;

  while (received < BYTES_NEEDED) {
    int to_read = min(CHUNK_SIZE, BYTES_NEEDED - received);
    unsigned long start = millis();

    while (Serial.available() < to_read) {
      if (millis() - start > 5000) {
        Serial.print("TIMEOUT at chunk ");
        Serial.println(chunk_count);
        return;
      }
    }

    for (int i = 0; i < to_read; i++)
      buf[received++] = Serial.read();

    chunk_count++;
    Serial.println("ACK");
  }

  // Read the optional one-hot label (3 bytes: e.g. [1, 0, 0] for dorsi).
  // If no label arrives, treat the window as unlabeled inference-only data.
  int label[N_CLASSES];
  bool hasLabel = true;
  for (int i = 0; i < N_CLASSES; i++) {
    unsigned long start = millis();
    while (!Serial.available()) {
      if (millis() - start > 5000) {
        hasLabel = false;
        break;
      }
    }
    if (!hasLabel)
      break;
    label[i] = Serial.read();
  }

  if (!hasLabel) {
    processWindow(values, -1);
    Serial.println("INFER");
    return;
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
  extractFeatures(data, N_CHANNELS, WINDOW, features);

  // prediction before training
  hat.predictProba(features, proba);
  int pred = hat.predict(features);

  bool isLabeled = labelIdx >= 0 && labelIdx < N_CLASSES;

  if (isLabeled) {
    // train on instance
    hat.train(features, labelIdx);
    labeledSamples++;
    if (pred == labelIdx)
      correctLabeledSamples++;
  }

  // serial output
  Serial.print("Probs: ");
  for (int i = 0; i < N_CLASSES; i++) {
    Serial.print(proba[i], 4);
    if (i < N_CLASSES - 1)
      Serial.print(", ");
  }
  Serial.println();

  Serial.print("Pred: ");
  Serial.print(CLASS_NAMES[pred]);
  if (isLabeled) {
    Serial.print(" | True: ");
    Serial.println(CLASS_NAMES[labelIdx]);
  } else {
    Serial.println(" | True: <none>");
  }

  // Tree size diagnostics
  Serial.print("Leaves: ");
  Serial.print(hat.leafCount());
  Serial.print(" | Internals: ");
  Serial.println(hat.internalCount());

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
    Serial.println(pred == labelIdx ? "CORRECT" : "WRONG");
  } else {
    Serial.println("UNLABELED");
  }
}

void setup() {
  Serial.begin(115200);
  // waitin on ptyhon
  while (true) {
    Serial.println("READY");
    delay(500);
    if (Serial.available()) {
      while (Serial.available())
        Serial.read(); // flush
      break;
    }
  }
}

void loop() {
  Serial.println("SEND");
  get_data();
}
