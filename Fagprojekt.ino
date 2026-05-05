#include "Activations.h"
#include "CategoricalCrossEntropyLoss.h"
#include "Conv2DLayer.h"
#include "DenseLayer.h"
#include "GetLoss.h"
#include "MaxPool2D.h"
#include <Arduino.h>

const int N_CHANNELS = 56;
const int WINDOW = 50;
const int N_CLASSES = 3;
const int N_FLOATS = N_CHANNELS * WINDOW;
const int BYTES_NEEDED = N_FLOATS * sizeof(float);
const int CHUNK_SIZE = 256;

const int BATCH_SIZE = 8;
const bool FREEZE_CONV = false;
const float INITIAL_LR = 0.001f;
const float LR_DECAY = 0.95f;
const int DECAY_STEP = 100;

const int CONV_FILTERS = 8;
const int CONV_KERNEL_H = 3;
const int CONV_KERNEL_W = 3;
const int CONV_STRIDE_H = 3;
const int CONV_STRIDE_W = 4;
const int CONV_PAD_H = 0;
const int CONV_PAD_W = 0;
const int CONV_OUT_H =
    (N_CHANNELS + 2 * CONV_PAD_H - CONV_KERNEL_H) / CONV_STRIDE_H + 1;
const int CONV_OUT_W =
    (WINDOW + 2 * CONV_PAD_W - CONV_KERNEL_W) / CONV_STRIDE_W + 1;

const int POOL_H = 3;
const int POOL_W = 3;
const int POOL_OUT_H = CONV_OUT_H / POOL_H;
const int POOL_OUT_W = CONV_OUT_W / POOL_W;

const int CONV_OUT_SIZE = CONV_FILTERS * POOL_OUT_H * POOL_OUT_W;
const int HIDDEN_SIZE = 32;
const int OUTPUT_SIZE = N_CLASSES;

static float values[N_FLOATS];
static Tensor flat(1, CONV_OUT_SIZE);
static Tensor dFlat(CONV_FILTERS, POOL_OUT_H *POOL_OUT_W);
static Tensor input(N_CHANNELS, WINDOW);
static Tensor yTrue(1, OUTPUT_SIZE);

Conv2DLayer conv(N_CHANNELS, WINDOW, CONV_FILTERS, CONV_KERNEL_H, CONV_KERNEL_W,
                 CONV_STRIDE_H, CONV_STRIDE_W, CONV_PAD_H, CONV_PAD_W);
ReLU reluConv;
MaxPool2D pool(CONV_FILTERS, CONV_OUT_H, CONV_OUT_W, POOL_H, POOL_W);
DenseLayer layer1(CONV_OUT_SIZE, HIDDEN_SIZE);
ReLU relu1;
DenseLayer layer2(HIDDEN_SIZE, OUTPUT_SIZE);
GetLoss getLoss;

static int windowCount = 0;
static int batchCounter = 0;
static int globalBatchCount = 0;
static float currentLR = INITIAL_LR;

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

  printTensor("conv_w", conv.weights);
  printTensor("conv_b", conv.biases);
  printTensor("layer1_w", layer1.weights);
  printTensor("layer1_b", layer1.biases);
  printTensor("layer2_w", layer2.weights);
  printTensor("layer2_b", layer2.biases);

  Serial.println("\n#endif");
  Serial.println("END_EXPORT");
}

void processWindow(const float *data, const int label[N_CLASSES], int wCount) {
  for (int i = 0; i < N_FLOATS; i++)
    input.data[i] = data[i];
  int labelIdx = -1;
  for (int i = 0; i < N_CLASSES; i++) {
    yTrue.data[i] = (label[i] == 1) ? 1.0f : 0.0f;
    if (label[i] == 1)
      labelIdx = i;
  }
  conv.forward(input);
  reluConv.forward(conv.output);
  pool.forward(reluConv.output);

  memcpy(flat.data, pool.output.data, CONV_OUT_SIZE * sizeof(float));

  layer1.forward(flat);
  relu1.forward(layer1.output);
  layer2.forward(relu1.output);

  float loss = getLoss.forward(layer2.output, yTrue);

  getLoss.backward(yTrue);
  layer2.backward(getLoss.dInputs);
  relu1.backward(layer2.dInputs);
  layer1.backward(relu1.dInputs);

  memcpy(dFlat.data, layer1.dInputs.data, CONV_OUT_SIZE * sizeof(float));

  pool.backward(dFlat);
  reluConv.backward(pool.dInputs);
  conv.backward(reluConv.dInputs, true);

  batchCounter++;
  if (batchCounter >= BATCH_SIZE) {
    float scaledLR = currentLR / BATCH_SIZE;
    globalBatchCount++;
    if (globalBatchCount % DECAY_STEP == 0)
      currentLR *= LR_DECAY;
    if (!FREEZE_CONV)
      conv.update(scaledLR);
    layer1.update(scaledLR);
    layer2.update(scaledLR);
    batchCounter = 0;
  }
  if (wCount % 10 == 0) {
    Tensor probs = getLoss.activation.output;
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
}

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

void setup() {
  Serial.begin(500000);
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
  }
  Serial.println("SEND");
  get_data();
}
