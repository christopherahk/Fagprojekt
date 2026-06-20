#include "FeatureExtractor.h"

#define MF_LAMBDA 18.0f
#define MF_N_TREES 10
#define MF_MAX_NODES 511
#include "MondrianForest.h"
#include <Arduino.h>

const int N_CHANNELS = 56;
const int WINDOW = 16;
const int N_CLASSES = 3;
const int N_FLOATS = N_CHANNELS * WINDOW;
// N_FEATURES: 56 channels * 10 local + 2 global
const int N_FEATURES = (N_CHANNELS * (N_RBI_BINS + 2)) + 2;
const int BYTES_NEEDED = N_FLOATS * sizeof(float);
const int CHUNK_SIZE = 256;

const char *CLASS_NAMES[N_CLASSES] = {"dorsi", "plantar", "none"};

static float values[N_FLOATS];
static float features[N_FEATURES];
static float proba[N_CLASSES];

static int val_total = 0;
static int val_correct = 0;
static float val_loss = 0.0f;

MondrianForest mf;

bool readSignal() {
  int received = 0;
  uint8_t *buf = reinterpret_cast<uint8_t *>(values);

  while (received < BYTES_NEEDED) {
    int to_read = min(CHUNK_SIZE, BYTES_NEEDED - received);
    unsigned long start = millis();
    while (Serial.available() < to_read) {
      if (millis() - start > 5000) {
        Serial.println("TIMEOUT");
        return false;
      }
    }
    for (int i = 0; i < to_read; i++) {
      buf[received++] = Serial.read();
    }
    Serial.println("ACK");
  }
  return true;
}

int readLabel() {
  unsigned long s = millis();
  while (!Serial.available()) {
    if (millis() - s > 2000)
      return -1;
  }
  char modeByte = Serial.read(); // reads L byte from streamer

  int label[N_CLASSES];
  for (int i = 0; i < N_CLASSES; i++) {
    unsigned long labelStart = millis();
    while (!Serial.available()) {
      if (millis() - labelStart > 2000)
        return -1;
    }
    label[i] = Serial.read();
  }
  for (int i = 0; i < N_CLASSES; i++) {
    if (label[i] == 1)
      return i;
  }
  return -1;
}

void processWindow(int labelIdx, bool doTrain, bool doValidate) {
  extractFeatures(values, N_CHANNELS, WINDOW, features);

  mf.predictProba(features, proba);
  int pred = mf.predict(features);

  float loss = 0.0f;
  if (labelIdx >= 0) {
    float p = proba[labelIdx];
    if (p < 1e-7f)
      p = 1e-7f;
    loss = -logf(p);
  }

  if (doTrain && labelIdx >= 0)
    mf.train(features, labelIdx);

  if (doValidate && labelIdx >= 0) {
    val_total++;
    val_loss += loss;
    if (pred == labelIdx)
      val_correct++;
  }

  Serial.print("Probs: ");
  for (int i = 0; i < N_CLASSES; i++) {
    Serial.print(proba[i], 4);
    if (i < N_CLASSES - 1)
      Serial.print(", ");
  }
  Serial.println();
  Serial.print("Pred: ");
  Serial.println(CLASS_NAMES[pred]);
}

void setup() {
  Serial.begin(1000000);
  while (true) {
    Serial.println("READY");
    delay(500);
    if (Serial.available() > 0) {
      char c = Serial.read();
      if (c == 'G') {
        while (Serial.available())
          Serial.read();
        break;
      }
      while (Serial.available())
        Serial.read();
    }
  }
  while (Serial.available())
    Serial.read();
}

void loop() {
  while (Serial.available() == 0) {
    Serial.println("SEND");
    delay(10);
  }

  char cmd = Serial.read();

  if (cmd == 'D') {
    if (!readSignal())
      return;
    int labelIdx = readLabel();
    processWindow(labelIdx, true, false);
    Serial.println("TRAIN");

  } else if (cmd == 'V') {
    val_total = 0;
    val_correct = 0;
    val_loss = 0.0f;

    while (true) {
      while (Serial.available() == 0) {
        Serial.println("SEND");
        delay(10);
      }
      char vcmd = Serial.read();

      if (vcmd == 'F')
        break;

      if (vcmd == 'D') {
        if (!readSignal())
          continue;
        int labelIdx = readLabel();
        processWindow(labelIdx, false, true);
        Serial.println("INFER");
      }
    }

    float avgLoss = val_total > 0 ? val_loss / val_total : 0.0f;
    float acc = val_total > 0 ? (float)val_correct / val_total : 0.0f;
    Serial.print("VAL_LOSS:");
    Serial.println(avgLoss, 4);
    Serial.print("VAL_ACC:");
    Serial.println(acc, 4);

  } else if (cmd == 'E') {
    unsigned long start = millis();
    while (!Serial.available() && millis() - start < 200)
      ;
    if (Serial.available() && Serial.read() == 'X') {
      Serial.println("START_EXPORT");
      Serial.print("# MondrianForest: trees=");
      Serial.print(MF_N_TREES);
      Serial.print(" lambda=");
      Serial.print(MF_LAMBDA);
      Serial.print(" nodes=");
      Serial.println(mf.totalNodes());
      Serial.println("END_EXPORT");
      while (true)
        if (Serial.available() && Serial.read() == 'R')
          break;
    }
  }
}
