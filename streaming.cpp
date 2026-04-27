#include "streaming.h"

#include "Activations.h"
#include "DenseLayer.h"
#include "GetLoss.h"
#include "Tensor.h"

#include <Arduino.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

namespace {
const int kInputRows = 56;
const int kInputCols = 100;
const int kHiddenNeurons = 16;
const int kOutputClasses = 3;
const float kLearningRate = 0.045f;
const unsigned long kLabelWaitMs = 20;

DenseLayer *gDense1 = nullptr;
DenseLayer *gDense2 = nullptr;
LeakyReLU gAct1;
GetLoss gLossActivation;

Tensor gInputTensor(kInputRows, kInputCols);
Tensor gTargetTensor(kInputRows, kOutputClasses);
Tensor gFrameProbs(kInputRows, kOutputClasses);

int gCurrentRow = 0;
bool gTensorReady = false;
bool gLabelReady = false;
bool gWaitingForLabel = false;
unsigned long gLabelWaitStartMs = 0;

String gPendingDataLine;
bool gHasPendingDataLine = false;

bool parseFloatToken(const char *token, float &out) {
  if (token == nullptr) {
    return false;
  }

  errno = 0;
  char *endPtr = nullptr;
  float value = strtof(token, &endPtr);
  if (endPtr == token || errno != 0 || !isfinite(value)) {
    return false;
  }

  while (endPtr != nullptr && *endPtr != '\0') {
    if (!isspace(static_cast<unsigned char>(*endPtr))) {
      return false;
    }
    endPtr++;
  }

  out = value;
  return true;
}

bool parseIntToken(const char *token, int &out) {
  if (token == nullptr) {
    return false;
  }

  errno = 0;
  char *endPtr = nullptr;
  long value = strtol(token, &endPtr, 10);
  if (endPtr == token || errno != 0) {
    return false;
  }

  while (endPtr != nullptr && *endPtr != '\0') {
    if (!isspace(static_cast<unsigned char>(*endPtr))) {
      return false;
    }
    endPtr++;
  }

  if (value < -2147483647L - 1L || value > 2147483647L) {
    return false;
  }

  out = static_cast<int>(value);
  return true;
}

bool parseFeatureRowInto(const String &line, int row) {
  if (row < 0 || row >= kInputRows) {
    return false;
  }

  char buffer[2048];
  line.toCharArray(buffer, sizeof(buffer));

  int col = 0;
  char *context = nullptr;
  char *token = strtok_r(buffer, ",", &context);
  if (token != nullptr && strlen(token) >= 2 &&
      ((token[0] == 'c' || token[0] == 'C') &&
       (token[1] == 'h' || token[1] == 'H'))) {
    token = strtok_r(nullptr, ",", &context);
  }

  while (token != nullptr) {
    if (col >= kInputCols) {
      return false;
    }

    float value = 0.0f;
    if (!parseFloatToken(token, value)) {
      return false;
    }
    gInputTensor(row, col) = value;
    col++;
    token = strtok_r(nullptr, ",", &context);
  }

  return col == kInputCols;
}

bool parseOneHotLabel(const String &line) {
  char buffer[128];
  line.toCharArray(buffer, sizeof(buffer));

  int values[kOutputClasses] = {0, 0, 0};
  int count = 0;

  char *context = nullptr;
  char *token = strtok_r(buffer, ",", &context);
  if (token == nullptr) {
    return false;
  }

  if (strcmp(token, "LABEL") == 0 || strcmp(token, "label") == 0) {
    token = strtok_r(nullptr, ",", &context);
  }

  while (token != nullptr) {
    if (count >= kOutputClasses) {
      return false;
    }

    int v = 0;
    if (!parseIntToken(token, v)) {
      return false;
    }
    if (v != 0 && v != 1) {
      return false;
    }

    values[count] = v;
    count++;
    token = strtok_r(nullptr, ",", &context);
  }

  if (count != kOutputClasses) {
    return false;
  }

  int sum = values[0] + values[1] + values[2];
  if (sum != 1) {
    return false;
  }

  for (int r = 0; r < kInputRows; r++) {
    for (int c = 0; c < kOutputClasses; c++) {
      gTargetTensor(r, c) = static_cast<float>(values[c]);
    }
  }

  return true;
}

void ingestLine(const String &line) {
  if (line.length() == 0 || line.startsWith("#META")) {
    return;
  }

  if (gTensorReady) {
    if (parseOneHotLabel(line)) {
      gLabelReady = true;
      gWaitingForLabel = false;
      return;
    }

    gPendingDataLine = line;
    gHasPendingDataLine = true;
    return;
  }

  if (!parseFeatureRowInto(line, gCurrentRow)) {
    gCurrentRow = 0;
    gTensorReady = false;
    gLabelReady = false;
    gWaitingForLabel = false;
    Serial.println("ERR_FRAME_RESET");
    return;
  }

  gCurrentRow++;
  if (gCurrentRow == kInputRows) {
    gTensorReady = true;
    gWaitingForLabel = true;
    gLabelWaitStartMs = millis();
  }
}
} // namespace

void initNetwork() {
  if (gDense1 == nullptr) {
    gDense1 = new DenseLayer(kInputCols, kHiddenNeurons);
  }
  if (gDense2 == nullptr) {
    gDense2 = new DenseLayer(kHiddenNeurons, kOutputClasses);
  }

  gCurrentRow = 0;
  gTensorReady = false;
  gLabelReady = false;
  gWaitingForLabel = false;
  gLabelWaitStartMs = 0;
  gHasPendingDataLine = false;
  gPendingDataLine = "";

  Serial.println("NET_READY");
}

void get_input() {
  if (gTensorReady && gLabelReady) {
    return;
  }

  if (gHasPendingDataLine && !gTensorReady) {
    String pending = gPendingDataLine;
    gPendingDataLine = "";
    gHasPendingDataLine = false;
    ingestLine(pending);
  }

  while (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.length() == 0) {
      continue;
    }

    ingestLine(line);

    if (gTensorReady && gLabelReady) {
      return;
    }

    if (gTensorReady && gHasPendingDataLine) {
      return;
    }
  }
}

bool input_is_ready() {
  if (!gTensorReady) {
    return false;
  }

  if (gLabelReady) {
    return true;
  }

  if (gWaitingForLabel) {
    unsigned long elapsed = millis() - gLabelWaitStartMs;
    if (elapsed < kLabelWaitMs) {
      return false;
    }
    gWaitingForLabel = false;
  }

  return true;
}

void forwardpass() {
  if (!gTensorReady || gDense1 == nullptr || gDense2 == nullptr) {
    return;
  }

  gDense1->forward(gInputTensor);
  gAct1.forward(gDense1->output, 0.01f);
  gDense2->forward(gAct1.output);
  gLossActivation.activation.forward(gDense2->output);
  gFrameProbs = gLossActivation.activation.output;
}

void output() {
  if (!gTensorReady) {
    return;
  }

  float avg[kOutputClasses] = {0.0f, 0.0f, 0.0f};

  for (int r = 0; r < kInputRows; r++) {
    for (int c = 0; c < kOutputClasses; c++) {
      avg[c] += gFrameProbs(r, c);
    }
  }

  int predClass = 0;
  float best = -1.0f;
  for (int c = 0; c < kOutputClasses; c++) {
    avg[c] /= static_cast<float>(kInputRows);
    if (avg[c] > best) {
      best = avg[c];
      predClass = c;
    }
  }

  Serial.print("OUT,avg=");
  Serial.print(avg[0], 6);
  Serial.print(",");
  Serial.print(avg[1], 6);
  Serial.print(",");
  Serial.print(avg[2], 6);
  Serial.print(",pred=");
  Serial.println(predClass);
}

bool labels_available() { return gTensorReady && gLabelReady; }

void update_weights() {
  if (!labels_available() || gDense1 == nullptr || gDense2 == nullptr) {
    return;
  }

  gLossActivation.backward(gTargetTensor);
  gDense2->backward(gLossActivation.dInputs, true);
  gAct1.backward(gDense2->dInputs);
  gDense1->backward(gAct1.dInputs, false);

  gDense1->update(kLearningRate);
  gDense2->update(kLearningRate);

  Serial.println("TRAIN,sgd=1");
}

void set_input_empty() {
  gTensorReady = false;
  gLabelReady = false;
  gWaitingForLabel = false;
  gLabelWaitStartMs = 0;
  gCurrentRow = 0;
}
