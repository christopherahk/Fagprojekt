#include "FeatureExtractor.h"
#define MF_LAMBDA 6.0f
#define MF_N_TREES 18
#define MF_MAX_NODES 127
#include "MondrianForest.h"
#include <Arduino.h>

bool testing = false;

const int N_CHANNELS = 56;
const int WINDOW = 32;
const int N_CLASSES = 3;
const int N_FLOATS = N_CHANNELS * WINDOW;
const int N_FEATURES = N_CHANNELS * (N_RBI_BINS + WINDOW / 2 + 1);
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

bool get_data() {
  int received = 0;
  uint8_t *buf = reinterpret_cast<uint8_t *>(values);
  while (received < BYTES_NEEDED) {
    int to_read = min(CHUNK_SIZE, BYTES_NEEDED - received);
    unsigned long start = millis();
    while (Serial.available() < to_read) {
      if (millis() - start > 5000) {
        return false;
      }
    }
    for (int i = 0; i < to_read; i++)
      buf[received++] = Serial.read();
    Serial.println("ACK");
  }
  return true;
}

int readLabel(char &mode) {
  unsigned long start = millis();
  while (!Serial.available()) {
    if (millis() - start > 2000)
      return -1;
  }
  mode = (char)Serial.read();

  int label[N_CLASSES];
  for (int i = 0; i < N_CLASSES; i++) {
    unsigned long s = millis();
    while (!Serial.available()) {
      if (millis() - s > 2000)
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

  if (doTrain && labelIdx >= 0 && !doValidate) {
    mf.train(features, labelIdx);
  }

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
  if (labelIdx >= 0) {
    Serial.print(" | Correct: ");
    Serial.print(pred == labelIdx ? 1 : 0);
  }
  Serial.println();

  Serial.print("Pred: ");
  Serial.println(CLASS_NAMES[pred]);

  if (doTrain && !doValidate) {
    Serial.println("TRAIN");
  } else {
    Serial.println("INFER");
  }
}

void setup() {
  Serial.begin(1000000);
  while (true) {
    Serial.println("READY");
    delay(200);
    if (Serial.available() >= 2) {
      char buf[3] = {};
      Serial.readBytes(buf, 2);
      if (strncmp(buf, "G", 1) == 0)
        break;
      while (Serial.available())
        Serial.read();
    }
  }
  while (Serial.available())
    Serial.read();
}

void loop() {
  Serial.println("SEND");
  while (!Serial.available())
    ;
  char cmd = Serial.read();

  if (cmd == 'D') {
    if (get_data()) {
      char mode = 'U';
      while (!Serial.available())
        ;
      char cmd2 = Serial.read();
      if (cmd2 == 'L') {
        int labelIdx = readLabel(mode);
        bool doValidate = testing;
        bool doTrain = (mode == 'L');
        processWindow(labelIdx, doTrain, doValidate);
      }
    }
  } else if (cmd == 'V') {
    testing = true;
    val_total = 0;
    val_correct = 0;
    val_loss = 0.0f;
  } else if (cmd == 'F') {
    Serial.print("VAL_LOSS:");
    Serial.println(val_total > 0 ? (val_loss / val_total) : 0.0f, 4);
    Serial.print("VAL_ACC:");
    Serial.println(val_total > 0 ? ((float)val_correct / val_total) : 0.0f, 4);
    testing = false;
  }
}
