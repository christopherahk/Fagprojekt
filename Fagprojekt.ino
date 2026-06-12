#include "FeatureExtractor.h"
#define MF_LAMBDA 6.0f
#define MF_N_TREES 24
#define MF_MAX_NODES 63
#include "MondrianForest.h"
#include <Arduino.h>

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

// Validation counters -- reset each validation pass
static int val_total = 0;
static int val_correct = 0;
static float val_loss = 0.0f;

MondrianForest mf;

// ── Read one window of raw signal data ───────────────────────────────────────
// Returns true on success, false on timeout.
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
    for (int i = 0; i < to_read; i++)
      buf[received++] = Serial.read();
    // No ACK -- streamer.py does not expect it and it clogs the buffer
  }
  return true;
}

// ── Read mode byte then one-hot label ────────────────────────────────────────
// Mode 'L' = labeled (train), 'U' = unlabeled (val/infer only).
// Returns label index 0-2, or -1 for unlabeled/error.
int readLabel() {
  unsigned long start = millis();
  while (!Serial.available()) {
    if (millis() - start > 2000)
      return -1;
  }
  char mode = (char)Serial.read();

  // Read the 3-byte one-hot regardless of mode
  int label[N_CLASSES];
  for (int i = 0; i < N_CLASSES; i++) {
    unsigned long s = millis();
    while (!Serial.available()) {
      if (millis() - s > 2000)
        return -1;
    }
    label[i] = Serial.read();
  }

  if (mode == 'U')
    return -1; // unlabeled -- infer only

  for (int i = 0; i < N_CLASSES; i++)
    if (label[i] == 1)
      return i;
  return -1;
}

// ── Process one window: extract features, predict, optionally train
// ───────────
void processWindow(int labelIdx, bool doTrain, bool doValidate) {
  extractFeatures(values, N_CHANNELS, WINDOW, features);

  mf.predictProba(features, proba);
  int pred = mf.predict(features);

  // Cross-entropy loss (used for val reporting and early stopping)
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

  if (!doValidate) {
    // Training mode -- send full output for Python logging
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
}

// ── Setup
// ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(1000000);

  // Wait for "GO\n" handshake from Python
  while (true) {
    Serial.println("READY");
    delay(200);
    if (Serial.available() >= 3) {
      char buf[4] = {};
      Serial.readBytes(buf, 3);
      if (strncmp(buf, "GO\n", 3) == 0)
        break;
      // Flush unexpected bytes and retry
      while (Serial.available())
        Serial.read();
    }
  }
  while (Serial.available())
    Serial.read();
}

// ── Main loop
// ─────────────────────────────────────────────────────────────────
void loop() {
  // Signal Python that we are ready for next window
  Serial.println("SEND");

  // Wait for command byte
  while (!Serial.available())
    ;
  char cmd = Serial.read();

  if (cmd == 'D') {
    // ── Training window ───────────────────────────────────────────────
    if (!readSignal())
      return;
    int labelIdx = readLabel();
    processWindow(labelIdx, /*doTrain=*/true, /*doValidate=*/false);
    Serial.println("TRAIN");

  } else if (cmd == 'V') {
    // ── Start validation pass ─────────────────────────────────────────
    // Python sends 'V' then streams windows with 'D' + 'U' + label.
    // After all val windows, Python sends 'F' to finish.
    val_total = 0;
    val_correct = 0;
    val_loss = 0.0f;

    while (true) {
      Serial.println("SEND");
      while (!Serial.available())
        ;
      char vcmd = Serial.read();

      if (vcmd == 'F')
        break; // end of validation

      if (vcmd == 'D') {
        if (!readSignal())
          continue;
        int labelIdx = readLabel();
        processWindow(labelIdx, /*doTrain=*/false, /*doValidate=*/true);
        Serial.println("INFER");
      }
    }

    // Report validation results
    float avgLoss = val_total > 0 ? val_loss / val_total : 0.0f;
    float acc = val_total > 0 ? (float)val_correct / val_total : 0.0f;
    Serial.print("VAL_LOSS:");
    Serial.println(avgLoss, 4);
    Serial.print("VAL_ACC:");
    Serial.println(acc, 4);

  } else if (cmd == 'E') {
    // ── Export model snapshot ─────────────────────────────────────────
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
