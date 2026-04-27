
// How to use (LOOCV test firmware):
// 1) Compile this file as the active streaming implementation with
// Fagprojekt.ino. 2) Open serial at 115200 and confirm it prints NET_READY. 3)
// Run python_files/loocv_coordinator.py to send START_FOLD/START_EVAL commands.
// 4) Firmware replies with RESULT,... and CONFUSION,... lines per fold.

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

#if defined(STREAMING_USE_LOOCV)

namespace {
const int kInputRows = 56;
const int kInputCols = 100;
const int kHiddenNeurons = 16;
const int kOutputClasses = 3;
const float kLearningRate = 0.045f;
const unsigned long kLabelWaitMs = 20;

// Mode: 0=inactive, 1=training, 2=evaluation
int gMode = 0;
int gCurrentFold = -1;

DenseLayer *gDense1 = nullptr;
DenseLayer *gDense2 = nullptr;
LeakyReLU gAct1;
GetLoss gLossActivation;

Tensor gInputTensor(kInputRows, kInputCols);
Tensor gTargetTensor(kInputRows, kOutputClasses);
Tensor gFrameProbs(kInputRows, kOutputClasses);

// Tracking for LOOCV
int gTrainSampleCount = 0;
int gTestSampleCount = 0;
float gAccumulatedLoss = 0.0f;
int gCorrectPredictions = 0;
int gConfusionMatrix[3][3] = {{0}};

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

void resetFrameState() {
  gCurrentRow = 0;
  gTensorReady = false;
  gLabelReady = false;
  gWaitingForLabel = false;
  gLabelWaitStartMs = 0;
  gHasPendingDataLine = false;
  gPendingDataLine = "";
}

void ingestLine(const String &line) {
  if (line.length() == 0) {
    return;
  }

  // Protocol commands.
  if (line.startsWith("START_FOLD")) {
    int fold = -1;
    sscanf(line.c_str(), "START_FOLD %d", &fold);
    gCurrentFold = fold;
    gMode = 1;
    gTrainSampleCount = 0;
    gAccumulatedLoss = 0.0f;
    gCorrectPredictions = 0;
    memset(gConfusionMatrix, 0, sizeof(gConfusionMatrix));
    resetFrameState();

    Serial.print("FOLD_START,fold=");
    Serial.println(fold);
    return;
  }

  if (line.startsWith("START_EVAL")) {
    gMode = 2;
    gTestSampleCount = 0;
    gAccumulatedLoss = 0.0f;
    gCorrectPredictions = 0;
    memset(gConfusionMatrix, 0, sizeof(gConfusionMatrix));
    resetFrameState();

    Serial.println("EVAL_START");
    return;
  }

  if (line.startsWith("GET_RESULTS")) {
    report_fold_results();
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
    resetFrameState();
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

int getTrueClassFromTarget() {
  for (int c = 0; c < kOutputClasses; c++) {
    if (gTargetTensor(0, c) > 0.5f) {
      return c;
    }
  }
  return -1;
}

int getPredClassFromProbs() {
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
  return predClass;
}
} // namespace

void initNetwork() {
  if (gDense1 == nullptr) {
    gDense1 = new DenseLayer(kInputCols, kHiddenNeurons);
  }
  if (gDense2 == nullptr) {
    gDense2 = new DenseLayer(kHiddenNeurons, kOutputClasses);
  }

  gMode = 0;
  gCurrentFold = -1;
  gTrainSampleCount = 0;
  gTestSampleCount = 0;
  gAccumulatedLoss = 0.0f;
  gCorrectPredictions = 0;
  memset(gConfusionMatrix, 0, sizeof(gConfusionMatrix));
  resetFrameState();

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

  int predClass = getPredClassFromProbs();

  Serial.print("OUT,pred=");
  Serial.println(predClass);
}

bool labels_available() { return gTensorReady && gLabelReady; }

void update_weights() {
  if (!labels_available() || gDense1 == nullptr || gDense2 == nullptr) {
    return;
  }

  if (gMode == 1) {
    // Training: update weights.
    gLossActivation.backward(gTargetTensor);
    gDense2->backward(gLossActivation.dInputs, true);
    gAct1.backward(gDense2->dInputs);
    gDense1->backward(gAct1.dInputs, false);

    gDense1->update(kLearningRate);
    gDense2->update(kLearningRate);

    float loss = gLossActivation.forward(gDense2->output, gTargetTensor);
    gAccumulatedLoss += loss;
    gTrainSampleCount++;

    Serial.print("TRAIN,sample=");
    Serial.print(gTrainSampleCount);
    Serial.print(",loss=");
    Serial.println(loss, 6);
  } else if (gMode == 2) {
    // Evaluation: metrics only, no updates.
    float loss = gLossActivation.forward(gDense2->output, gTargetTensor);
    gAccumulatedLoss += loss;
    gTestSampleCount++;

    int trueClass = getTrueClassFromTarget();
    int predClass = getPredClassFromProbs();

    if (trueClass >= 0 && predClass >= 0) {
      gConfusionMatrix[trueClass][predClass]++;
      if (trueClass == predClass) {
        gCorrectPredictions++;
      }
    }

    Serial.print("EVAL,sample=");
    Serial.print(gTestSampleCount);
    Serial.print(",loss=");
    Serial.print(loss, 6);
    Serial.print(",true=");
    Serial.print(trueClass);
    Serial.print(",pred=");
    Serial.println(predClass);
  }
}

void set_input_empty() { resetFrameState(); }

void report_fold_results() {
  if (gMode != 2 || gTestSampleCount == 0) {
    Serial.println("RESULT,loss=0.000000,acc=0.000000,correct=0,total=0");
    return;
  }

  float meanLoss = gAccumulatedLoss / static_cast<float>(gTestSampleCount);
  float accuracy = static_cast<float>(gCorrectPredictions) /
                   static_cast<float>(gTestSampleCount);

  Serial.print("RESULT,loss=");
  Serial.print(meanLoss, 6);
  Serial.print(",acc=");
  Serial.print(accuracy, 6);
  Serial.print(",correct=");
  Serial.print(gCorrectPredictions);
  Serial.print(",total=");
  Serial.println(gTestSampleCount);

  Serial.print("CONFUSION,");
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Serial.print(gConfusionMatrix[i][j]);
      if (i < 2 || j < 2) {
        Serial.print(",");
      }
    }
  }
  Serial.println();
}

#endif // STREAMING_USE_LOOCV
