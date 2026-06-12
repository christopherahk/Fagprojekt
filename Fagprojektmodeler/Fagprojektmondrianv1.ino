#include "FeatureExtractor.h"
#define MF_LAMBDA 6.0f  // Højere = dybere træer til  1176 features
#define MF_N_TREES 24   //
#define MF_MAX_NODES 63 //
#include "MondrianForest.h"
#include <Arduino.h>

const int N_CHANNELS = 56;
const int WINDOW = 32;
const int N_CLASSES = 3;
const int N_FLOATS = N_CHANNELS * WINDOW;                          // 1792
const int N_FEATURES = N_CHANNELS * (N_RBI_BINS + WINDOW / 2 + 1); // 56*21=1176

const int BYTES_NEEDED = N_FLOATS * sizeof(float); // 7168 bytes
const int CHUNK_SIZE = 256;

// Class names must match Python ARDUINO_CLASS_NAMES order
const char *CLASS_NAMES[N_CLASSES] = {"dorsi", "plantar", "none"};

// Static buffers
static float values[N_FLOATS];
static float features[N_FEATURES];
static float proba[N_CLASSES];

//
bool testing = false; // false = TRAIN mode, true = VALIDATION/TEST mode

static int windowCount = 0;
static float total_loss = 0.0f;
static int total_correct = 0;

MondrianForest mf;

void processWindow(const float *data, const int label[N_CLASSES], int wCount);

// Serial protocol
void get_data() {
  int received = 0;
  uint8_t *buf = reinterpret_cast<uint8_t *>(values);

  while (received < BYTES_NEEDED) {
    int to_read = min(CHUNK_SIZE, BYTES_NEEDED - received);
    unsigned long start = millis();
    while (Serial.available() < to_read) {
      if (millis() - start > 5000)
        return;
    }
    for (int i = 0; i < to_read; i++) {
      buf[received++] = Serial.read();
    }

    Serial.println("ACK");
  }

  int label[N_CLASSES];
  for (int i = 0; i < N_CLASSES; i++) {
    while (!Serial.available())
      ;
    label[i] = Serial.read();
  }

  processWindow(values, label, windowCount++);

  if (!testing) {
    Serial.println("TRAIN");
  } else {
    Serial.println("INFER");
  }
}

void processWindow(const float *data, const int label[N_CLASSES], int wCount) {

  extractFeatures(data, N_CHANNELS, WINDOW, features);

  mf.predictProba(features, proba);
  int pred = mf.predict(features);

  int labelIdx = -1;
  for (int i = 0; i < N_CLASSES; i++) {
    if (label[i] == 1) {
      labelIdx = i;
    }
  }

  float loss = 0.0f;
  if (labelIdx >= 0 && labelIdx < N_CLASSES) {
    float p = proba[labelIdx];
    if (p < 1e-7f)
      p = 1e-7f;
    loss = -logf(p);
  }

  if (!testing) {
    if (labelIdx >= 0 && labelIdx < N_CLASSES) {
      mf.train(features, labelIdx);
    }
  }

  if (!testing) {
    Serial.print("Probs: ");
    for (int i = 0; i < N_CLASSES; i++) {
      Serial.print(proba[i], 4);
      if (i < N_CLASSES - 1)
        Serial.print(", ");
    }
    Serial.print(" | Label: ");
    Serial.print(labelIdx);
    Serial.print(" | Loss: ");
    Serial.println(loss, 4);

    Serial.print("Pred: ");
    Serial.println(CLASS_NAMES[pred]);
  }

  if (testing) {
    total_loss += loss;
    if (pred == labelIdx) {
      total_correct++;
    }
  }
}

void setup() {

  Serial.begin(1000000);
  while (true) {
    Serial.println("READY");
    delay(500);
    if (Serial.available()) {
      while (Serial.available())
        Serial.read();
      break;
    }
  }
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == 'E') {
      unsigned long start = millis();
      while (!Serial.available()) {
        if (millis() - start > 100)
          break;
      }
      if (Serial.available() && Serial.read() == 'X') {

        Serial.println("START_EXPORT");
        Serial.println("#ifndef TRAINED_WEIGHTS_H");
        Serial.println("#define TRAINED_WEIGHTS_H\n");

        Serial.print("# Mondrian Forest: ");
        Serial.print(MF_N_TREES);
        Serial.print(" trees, lambda=");
        Serial.print(MF_LAMBDA);
        Serial.print(", totalNodes=");
        Serial.println(mf.totalNodes());

        Serial.println("\n#endif");
        Serial.println("END_EXPORT");

        while (true) {
          if (Serial.available() && Serial.read() == 'R')
            break;
        }
        return;
      }
    }

    if (c == 'V') {

      testing = true;
      total_loss = 0.0f;
      total_correct = 0;
      windowCount = 0;
    }

    if (c == 'D') {

      Serial.print("VAL_LOSS:");
      Serial.println(windowCount > 0 ? (float)total_loss / windowCount : 0.0f,
                     4);
      Serial.print("VAL_ACC:");
      Serial.println(
          windowCount > 0 ? (float)total_correct / windowCount : 0.0f, 4);

      testing = false;
      total_loss = 0.0f;
      total_correct = 0;
      windowCount = 0;
    }
  }

  Serial.println("SEND");
  get_data();
}
