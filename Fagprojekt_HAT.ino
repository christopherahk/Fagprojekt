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

  // Read the one-hot label (3 bytes: e.g. [1, 0, 0] for dorsi)
  int label[N_CLASSES];
  for (int i = 0; i < N_CLASSES; i++) {
    while (!Serial.available())
      ;
    label[i] = Serial.read();
  }

  processWindow(values, label);
  Serial.println("TRAIN");
}

void processWindow(const float *data, const int label[N_CLASSES]) {
  // Decode one-hot label to class index
  int labelIdx = -1;
  for (int i = 0; i < N_CLASSES; i++) {
    if (label[i] == 1) {
      labelIdx = i;
      break;
    }
  }
  if (labelIdx < 0)
    return; // invalid skip

  // feature extraction
  extractFeatures(data, N_CHANNELS, WINDOW, features);

  // prediction before training
  hat.predictProba(features, proba);
  int pred = hat.predict(features);

  // train on instance
  hat.train(features, labelIdx);

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
  Serial.print(" | True: ");
  Serial.println(CLASS_NAMES[labelIdx]);

  // Tree size diagnostics
  Serial.print("Leaves: ");
  Serial.print(hat.leafCount());
  Serial.print(" | Internals: ");
  Serial.println(hat.internalCount());

  Serial.println(pred == labelIdx ? "CORRECT" : "WRONG");
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
