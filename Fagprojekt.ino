#include "Activations.h"
#include "CategoricalCrossEntropyLoss.h"
#include "DenseLayer.h"
#include "GetLoss.h"
#include <Arduino.h>

const int N_CHANNELS = 56;
const int WINDOW = 20;
const int N_CLASSES = 3;
const int N_FLOATS = N_CHANNELS * WINDOW;
const int BYTES_NEEDED = N_FLOATS * sizeof(float);
const int CHUNK_SIZE = 64;

static float values[N_FLOATS];

const int INPUT_SIZE = 1120;
const int HIDDEN_SIZE = 4;
const int OUTPUT_SIZE = 3;
const float LEARNING_RATE = 0.01f;

DenseLayer layer1(INPUT_SIZE, HIDDEN_SIZE);
DenseLayer layer2(HIDDEN_SIZE, OUTPUT_SIZE);
ReLU relu;
GetLoss getLoss;

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
  Tensor input(1, INPUT_SIZE);
  for (int i = 0; i < INPUT_SIZE; i++)
    input.data[i] = data[i];

  int labelIdx = -1;
  for (int i = 0; i < N_CLASSES; i++) {
    if (label[i] == 1) {
      labelIdx = i;
      break;
    }
  }

  Tensor yTrue(1, OUTPUT_SIZE);
  for (int i = 0; i < OUTPUT_SIZE; i++)
    yTrue.data[i] = (i == labelIdx) ? 1.0f : 0.0f;

  layer1.forward(input);
  relu.forward(layer1.output);
  layer2.forward(relu.output);
  float loss = getLoss.forward(layer2.output, yTrue);

  getLoss.backward(yTrue);
  layer2.backward(getLoss.dInputs);
  relu.backward(layer2.dInputs);
  layer1.backward(relu.dInputs);

  layer1.update(LEARNING_RATE);
  layer2.update(LEARNING_RATE);

  Tensor probs = getLoss.activation.output;
  Serial.print("Probs: ");
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    Serial.print(probs.data[i], 4);
    if (i < OUTPUT_SIZE - 1)
      Serial.print(", ");
  }
  Serial.println();

  Serial.print("TRAIN loss=");
  Serial.println(loss, 4);
}

void setup() {
  Serial.begin(115200);
  while (true) {
    Serial.println("READY");
    delay(500);
    if (Serial.available()) {
      while (Serial.available()) {
        Serial.read();
      }
      break;
    }
  }
}

void loop() {
  Serial.println("SEND");
  get_data();
}
