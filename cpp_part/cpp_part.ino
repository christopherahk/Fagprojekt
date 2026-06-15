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

// Validation counters -- reset each validation pass
static int val_total = 0;
static int val_correct = 0;
static float val_loss = 0.0f;

MondrianForest mf;

static int windowCount = 0;
static int batchCounter = 0;
static int globalBatchCount = 0;
static float currentLR = INITIAL_LR;

static float total_loss = 0.0f;
static int total_correct = 0;

/*void loadWeights() {
  memcpy(conv1.weights.data, conv1_w, conv1.weights.size * sizeof(float));
  memcpy(conv1.biases.data, conv1_b, conv1.biases.size * sizeof(float));
  memcpy(conv1.mWeights.data, conv1_mw, conv1.mWeights.size * sizeof(float));
  memcpy(conv1.mBiases.data, conv1_mb, conv1.mBiases.size * sizeof(float));
  memcpy(conv2.weights.data, conv2_w, conv2.weights.size * sizeof(float));
  memcpy(conv2.biases.data, conv2_b, conv2.biases.size * sizeof(float));
  memcpy(conv2.mWeights.data, conv2_mw, conv2.mWeights.size * sizeof(float));
  memcpy(conv2.mBiases.data, conv2_mb, conv2.mBiases.size * sizeof(float));
  memcpy(layer1.weights.data, layer1_w, layer1.weights.size * sizeof(float));
  memcpy(layer1.biases.data, layer1_b, layer1.biases.size * sizeof(float));
  memcpy(layer1.mWeights.data, layer1_mw, layer1.mWeights.size * sizeof(float));
  memcpy(layer1.mBiases.data, layer1_mb, layer1.mBiases.size * sizeof(float));
  memcpy(layer2.weights.data, layer2_w, layer2.weights.size * sizeof(float));
  memcpy(layer2.biases.data, layer2_b, layer2.biases.size * sizeof(float));
  memcpy(layer2.mWeights.data, layer2_mw, layer2.mWeights.size * sizeof(float));
  memcpy(layer2.mBiases.data, layer2_mb, layer2.mBiases.size * sizeof(float));
}*/

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
  printTensor("conv1_mw", conv1.mWeights);
  printTensor("conv1_mb", conv1.mBiases);
  printTensor("conv2_w", conv2.weights);
  printTensor("conv2_b", conv2.biases);
  printTensor("conv2_mw", conv2.mWeights);
  printTensor("conv2_mb", conv2.mBiases);
  printTensor("layer1_w", layer1.weights);
  printTensor("layer1_b", layer1.biases);
  printTensor("layer1_mw", layer1.mWeights);
  printTensor("layer1_mb", layer1.mBiases);
  printTensor("layer2_w", layer2.weights);
  printTensor("layer2_b", layer2.biases);
  printTensor("layer2_mw", layer2.mWeights);
  printTensor("layer2_mb", layer2.mBiases);

  Serial.println("\n#endif");
  Serial.println("END_EXPORT");
}

void printProbs(int labelIdx, float loss) {
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

void processWindow(const float *data, const int label[N_CLASSES], int wCount) {
  for (int i = 0; i < N_FLOATS; i++)
    input.data[i] = data[i];

  int labelIdx = -1;
  for (int i = 0; i < N_CLASSES; i++) {
    yTrue.data[i] = (label[i] == 1) ? 1.0f : 0.0f;
    if (label[i] == 1)
      labelIdx = i;
  }

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

  if (!testing) {
    getLoss.backward(yTrue);
    layer2.backward(getLoss.dInputs);
    relu1.backward(layer2.dInputs);
    layer1.backward(relu1.dInputs);

    memcpy(dFlat.data, layer1.dInputs.data, POOL_OUT * sizeof(float));
    for (int c = 0; c < C2_FILTERS; c++) {
      float g = dFlat.data[c] / C2_OUT_W;
      for (int t = 0; t < C2_OUT_W; t++)
        reluConv2.output.data[c * C2_OUT_W + t] = g;
    }

    reluConv2.backward(reluConv2.output);
    conv2.backward(reluConv2.dInputs, true);

    reluConv1.backward(conv2.dInputs);
    conv1.backward(reluConv1.dInputs, true);

    batchCounter++;
    if (batchCounter >= BATCH_SIZE) {
      float scaledLR = currentLR / BATCH_SIZE;
      globalBatchCount++;
      if (globalBatchCount % DECAY_STEP == 0)
        currentLR *= LR_DECAY;
      if (!FREEZE_CONV) {
        conv1.update(scaledLR);
        conv2.update(scaledLR);
      }
      layer1.update(scaledLR);
      layer2.update(scaledLR);
      batchCounter = 0;
    }
  }

  if ((wCount % 100 == 0 && !testing)) {
    printProbs(labelIdx, loss);
  }

  if (testing) {
    total_loss += loss;

    Tensor probs = getLoss.activation.output;
    int pred = 0;
    for (int i = 1; i < N_CLASSES; i++)
      if (probs.data[i] > probs.data[pred]) {
        pred = i;
      }
    if (pred == labelIdx) {
      total_correct++;
    }
  }
}

void get_data() {
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
  }
  return true;
}

// one hot
// Mode L = labeled (train), U = unlabeled (val/infer only).
// Returns label index 0-2, or -1 for unlabeled/error.
int readLabel() {
  unsigned long start = millis();
  while (!Serial.available()) {
    if (millis() - start > 2000)
      return -1;
  }
  char mode = (char)Serial.read();

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
    return -1; // unlabeled infer only

  for (int i = 0; i < N_CLASSES; i++)
    if (label[i] == 1)
      return i;
  return -1;
}

// Process one window, extract features, predict, optionally train

void processWindow(int labelIdx, bool doTrain, bool doValidate) {
  extractFeatures(values, N_CHANNELS, WINDOW, features);

  mf.predictProba(features, proba);
  int pred = mf.predict(features);

  // Cross-entropy loss
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
    // Training mode
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

void setup() {
  Serial.begin(1000000);
  if (testing) {
  }
  while (true) {
    Serial.println("READY");
    delay(200);
    if (Serial.available() >= 3) {
      char buf[4] = {};
      Serial.readBytes(buf, 3);
      if (strncmp(buf, "GO\n", 3) == 0)
        break;

      while (Serial.available())
        Serial.read();
    }
  }
  while (Serial.available())
    Serial.read();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'E') {
      unsigned long start = millis();
      while (!Serial.available())
        if (millis() - start > 100)
          break;
      if (Serial.available() && Serial.read() == 'X') {
        exportModel();
        while (true)
          if (Serial.available() && Serial.read() == 'R')
            break;
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
      Serial.println((float)total_loss / windowCount, 4);
      Serial.print("VAL_ACC:");
      Serial.println((float)total_correct / windowCount, 4);
      testing = false;
      total_loss = 0.0f;
      total_correct = 0;
      windowCount = 0;
      currentLR = INITIAL_LR;
    }
  }
}
