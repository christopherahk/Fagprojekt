#define SERIAL_RX_BUFFER_SIZE 256

#include <Arduino.h>
#include "tree_weights.h" // Keep this at the top

#include "Activations.h"
#include "CategoricalCrossEntropyLoss.h"
#include "Conv2dlayer.h"
#include "DenseLayer.h"
#include "GetLoss.h"
#include "trained_weights.h"

bool testing = true;

// Bayesian-optimised CNN weight: final = alpha*CNN + (1-alpha)*forest
const float ENSEMBLE_ALPHA = 0.99f;

// ── Network dimensions
// ────────────────────────────────────────────────────────
const int N_CHANNELS = 56;
const int SEQ_LEN = 16;
const int N_CLASSES = 3;
const int N_FLOATS = N_CHANNELS * SEQ_LEN;
const int BYTES_NEEDED = N_FLOATS * sizeof(float);
const int CHUNK_SIZE = 256;

const int BATCH_SIZE = 64;
const bool FREEZE_CONV = false;
const float INITIAL_LR = 0.002f;
const float LR_DECAY = 0.95f;

// ── Conv-1
// ────────────────────────────────────────────────────────────────────
const int C1_FILTERS = 16;
const int C1_KH = N_CHANNELS;
const int C1_KW = 3;
const int C1_SH = N_CHANNELS;
const int C1_SW = 1;
const int C1_PAD_H = 0;
const int C1_PAD_W = 1;
const int C1_OUT_H = (N_CHANNELS + 2 * C1_PAD_H - C1_KH) / C1_SH + 1;
const int C1_OUT_W = (SEQ_LEN + 2 * C1_PAD_W - C1_KW) / C1_SW + 1;

// ── Conv-2
// ────────────────────────────────────────────────────────────────────
const int C2_FILTERS = 32;
const int C2_KH = C1_FILTERS;
const int C2_KW = 3;
const int C2_SH = C1_FILTERS;
const int C2_SW = 1;
const int C2_PAD_H = 0;
const int C2_PAD_W = 1;
const int C2_OUT_H = (C1_FILTERS + 2 * C2_PAD_H - C2_KH) / C2_SH + 1;
const int C2_OUT_W = (C1_OUT_W + 2 * C2_PAD_W - C2_KW) / C2_SW + 1;

const int POOL_OUT = C2_FILTERS;
const int HIDDEN_SIZE = 16;
const int OUTPUT_SIZE = N_CLASSES;

// ── Static buffers
// ────────────────────────────────────────────────────────────
static float values[N_FLOATS];
static Tensor flat(1, POOL_OUT);
static Tensor dFlat(1, POOL_OUT);
static Tensor input(N_CHANNELS, SEQ_LEN);
static Tensor yTrue(1, OUTPUT_SIZE);

static float blendedProbs[N_CLASSES];

// ── Layers
// ────────────────────────────────────────────────────────────────────
Conv2DLayer conv1(N_CHANNELS, SEQ_LEN, C1_FILTERS, C1_KH, C1_KW, C1_SH, C1_SW,
                  C1_PAD_H, C1_PAD_W);
ReLU reluConv1;
Conv2DLayer conv2(C1_FILTERS, C1_OUT_W, C2_FILTERS, C2_KH, C2_KW, C2_SH, C2_SW,
                  C2_PAD_H, C2_PAD_W);
ReLU reluConv2;

DenseLayer layer1(POOL_OUT, HIDDEN_SIZE);
ReLU relu1;
DenseLayer layer2(HIDDEN_SIZE, OUTPUT_SIZE);
GetLoss getLoss;

static int windowCount = 0;
static int batchCounter = 0;
static float currentLR = INITIAL_LR;
static float total_loss = 0.0f;
static int total_correct = 0;

// ══════════════════════════════════════════════════════════════════════════════
// Weight I/O
// ══════════════════════════════════════════════════════════════════════════════

void loadWeights() {
  memcpy(conv1.weights.data, conv1_w, conv1.weights.size * sizeof(float));
  memcpy(conv1.biases.data, conv1_b, conv1.biases.size * sizeof(float));
  memcpy(conv2.weights.data, conv2_w, conv2.weights.size * sizeof(float));
  memcpy(conv2.biases.data, conv2_b, conv2.biases.size * sizeof(float));
  memcpy(layer1.weights.data, layer1_w, layer1.weights.size * sizeof(float));
  memcpy(layer1.biases.data, layer1_b, layer1.biases.size * sizeof(float));
  memcpy(layer2.weights.data, layer2_w, layer2.weights.size * sizeof(float));
  memcpy(layer2.biases.data, layer2_b, layer2.biases.size * sizeof(float));
}

void exportModel() {
  Serial.println("START_EXPORT");
  Serial.println("#ifndef TRAINED_WEIGHTS_H");
  Serial.println("#define TRAINED_WEIGHTS_H\n");

  auto printTensor = [](const char *name, Tensor &t) {
    Serial.print("float ");
    Serial.print(name);
    Serial.print("[] = {");
    for (int i = 0; i < t.size; i++) {
      Serial.print(t.data[i], 7);
      if (i < t.size - 1)
        Serial.print(", ");
      if (i % 8 == 7)
        Serial.print("\n    ");
    }
    Serial.println("};");
  };

  printTensor("conv1_w", conv1.weights);
  printTensor("conv1_b", conv1.biases);
  printTensor("conv2_w", conv2.weights);
  printTensor("conv2_b", conv2.biases);
  printTensor("layer1_w", layer1.weights);
  printTensor("layer1_b", layer1.biases);
  printTensor("layer2_w", layer2.weights);
  printTensor("layer2_b", layer2.biases);

  Serial.println("\n#endif");
  Serial.println("END_EXPORT");
}

// ── Logging helpers
// ───────────────────────────────────────────────────────────

void printProbs(int labelIdx, float loss) {
  const Tensor &probs = getLoss.activation.output;
  Serial.print("Probs: ");
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    Serial.print(probs.data[i], 2);
    if (i < OUTPUT_SIZE - 1)
      Serial.print(", ");
  }
  Serial.print(" | Label: ");
  Serial.print(labelIdx);
  Serial.print(" | Loss: ");
  Serial.println(loss, 2);
}

void printBlendedProbs(int labelIdx, float loss, const float *blended) {
  Serial.print("Ensemble: ");
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    Serial.print(blended[i], 2);
    if (i < OUTPUT_SIZE - 1)
      Serial.print(", ");
  }
  Serial.print(" | Label: ");
  Serial.print(labelIdx);
  Serial.print(" | CNN_Loss: ");
  Serial.println(loss, 2);
}

// ── Core window processing
// ────────────────────────────────────────────────────

void processWindow(const float *data, const int label[N_CLASSES], int wCount) {
  for (int i = 0; i < N_FLOATS; i++)
    input.data[i] = data[i];
  int labelIdx = -1;
  for (int i = 0; i < N_CLASSES; i++) {
    yTrue.data[i] = (label[i] == 1) ? 1.0f : 0.0f;
    if (label[i] == 1)
      labelIdx = i;
  }

  // ── CNN forward pass ───────────────────────────────────────────────────────
  conv1.forward(input);
  reluConv1.forward(conv1.output);
  conv2.forward(reluConv1.output);
  reluConv2.forward(conv2.output);

  for (int c = 0; c < C2_FILTERS; c++) {
    float sum = 0.0f;
    for (int t = 0; t < C2_OUT_W; t++)
      sum += reluConv2.output.data[c * C2_OUT_W + t];
    flat.data[c] = sum / C2_OUT_W;
  }

  layer1.forward(flat);
  relu1.forward(layer1.output);
  layer2.forward(relu1.output);

  float loss = getLoss.forward(layer2.output, yTrue);

  // ── Ensemble blend ─────────────────────────────────────────────────────────
  // Get predictions from the Random Forest (defined in tree_weights.h)
  float forestProbs[TREE_N_CLASSES];
  treeForestPredict(input.data, forestProbs);

  // Blend CNN and Forest probabilities
  for (int i = 0; i < N_CLASSES; i++) {
    blendedProbs[i] = (ENSEMBLE_ALPHA * getLoss.activation.output.data[i]) +
                      ((1.0f - ENSEMBLE_ALPHA) * forestProbs[i]);
  }

  // ── Backward / update (CNN only; forest is frozen) ────────────────────────
  if (!testing) {
    getLoss.backward(yTrue);
    layer2.backward(getLoss.dInputs);
    relu1.backward(layer2.dInputs);
    layer1.backward(relu1.dInputs);

    memcpy(dFlat.data, layer1.dInputs.data, POOL_OUT * sizeof(float));

    if (reluConv2.dInputs.rowCount != C2_FILTERS ||
        reluConv2.dInputs.colCount != C2_OUT_W) {
      reluConv2.dInputs = Tensor(C2_FILTERS, C2_OUT_W);
    }

    for (int c = 0; c < C2_FILTERS; c++) {
      float g = dFlat.data[c] / C2_OUT_W;
      for (int t = 0; t < C2_OUT_W; t++) {
        reluConv2.dInputs.data[c * C2_OUT_W + t] = g;
      }
    }

    reluConv2.backward(reluConv2.dInputs);
    if (!FREEZE_CONV) {
      conv2.backward(reluConv2.dInputs, true);
      reluConv1.backward(conv2.dInputs);
      conv1.backward(reluConv1.dInputs, true);
    }

    batchCounter++;
    if (batchCounter >= BATCH_SIZE) {
      float scaledLR = currentLR / BATCH_SIZE;
      if (!FREEZE_CONV) {
        conv1.update(scaledLR);
        conv2.update(scaledLR);
      }
      layer1.update(scaledLR);
      layer2.update(scaledLR);
      batchCounter = 0;
    }

    if (wCount % 100 == 0) {
      printProbs(labelIdx, loss);
    }
  }

  // ── Evaluation: blended ensemble decides the class ────────────────────────
  if (testing) {
    total_loss += loss;

    int pred = 0;
    float best_val = blendedProbs[0];
    for (int i = 1; i < N_CLASSES; i++) {
      if (blendedProbs[i] > best_val) {
        best_val = blendedProbs[i];
        pred = i;
      }
    }

    if (pred == labelIdx)
      total_correct++;
    printBlendedProbs(labelIdx, loss, blendedProbs);
  }
}

// ── Serial I/O
// ────────────────────────────────────────────────────────────────

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
    for (int i = 0; i < to_read; i++)
      buf[received++] = Serial.read();
    Serial.println("ACK");
  }
  int label[N_CLASSES];
  for (int i = 0; i < N_CLASSES; i++) {
    while (!Serial.available())
      ;
    label[i] = Serial.read();
  }
  processWindow(values, label, windowCount++);
  Serial.println("TRAIN");
}

// ── Arduino entry points
// ──────────────────────────────────────────────────────

void setup() {
  Serial.begin(1000000);
  loadWeights();
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
        exportModel();
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
      currentLR *= LR_DECAY;
      return;
    }

    if (c == 'D') {
      Serial.print("VAL_LOSS:");
      Serial.println((float)total_loss / windowCount, 4);
      Serial.print("VAL_ACC:");
      Serial.println((float)total_correct / windowCount, 4);
      testing = false;
      total_loss = 0.0f;
      total_correct = 0;
      windowCount = 0;
      return;
    }

    if (c == 'T') {
      testing = true;
      total_loss = 0.0f;
      total_correct = 0;
      windowCount = 0;
      return;
    }
  }
  Serial.println("SEND");
  get_data();
}
