#include "Tensor.h"
#include "NNLayer.h"
#include "ReLU.h"
#include "Activation_Sofmax_CategoricalCrossEntropy.h"
#include "SGD.h"
#include <algorithm>

float* oneHot(Tensor& batchY, int row, int label, int nClasses) {
  for (int c = 0; c < nClasses; c++) {
    batchY(row, c) = 0.0f;
  }

  batchY(row, label) = 1.0f;
}

void copyBatch(
  const float* train_data,
  const float* train_labels,
  int startIndex,
  Tensor& batchX,
  Tensor& batchY,
  int nInputs,
  int nClasses,
  int batchSize)
{
  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < nInputs; j++) {
      batchX(i, j) = train_data[startIndex + i];
    }

    int label = (int)train_labels[startIndex + i];
    oneHot(batchY, i, label, nClasses);
  }
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  randomSeed(analogRead(A0));

  float train_data[6] = {0, 1, 2, 3, 4, 5};
  float train_labels[6] = {1, 0, 1, 0, 1, 0};

  int N_INPUTS = 1;
  int TRAINING_SAMPLES = 6;
  int N_HIDDEN = 100;
  int N_CLASSES = 2;
  int BATCH_SIZE = 2;
  int EPOCHS = 5;

  NNLayer layer1(N_INPUTS, N_HIDDEN, BATCH_SIZE);
  ReLU relu;
  NNLayer layer2(N_HIDDEN, N_CLASSES, BATCH_SIZE);
  Activation_Sofmax_CategoricalCrossEntropy lossActivation;

  SGD optimizer1(0.005);
  SGD optimizer2(0.005);

  Tensor batchX(BATCH_SIZE, N_INPUTS);
  Tensor batchY(BATCH_SIZE, N_CLASSES);

  Serial.print("Starting training for ");
  Serial.print(EPOCHS);
  Serial.println(" epochs.");

  for (int epoch = 1; epoch <= EPOCHS; epoch++) {
    float epochLoss = 0.0f;
    int numBatches = TRAINING_SAMPLES / BATCH_SIZE;

    for (int batch = 0; batch < numBatches; batch++) {
      Serial.print("Doing batch number: ");
      Serial.print(batch);
      Serial.print("/");
      Serial.print(numBatches);
      Serial.print(" | Epoch: ");
      Serial.println(epoch);

      int startIndex = batch * BATCH_SIZE;
      copyBatch(train_data, train_labels, startIndex, batchX, batchY, N_INPUTS, N_CLASSES, BATCH_SIZE);

      layer1.forward(batchX);
      relu.forward(layer1.output);
      layer2.forward(relu.output);
      float loss = lossActivation.forward(layer2.output, batchY);
      epochLoss += loss;

      lossActivation.backward(batchY);
      layer2.backward(lossActivation.dInputs);
      relu.backward(layer2.dInputs);
      layer1.backward(relu.dInputs);

      optimizer2.updateParameters(layer2);
      optimizer1.updateParameters(layer1);
    }

    Serial.print("Epoch: ");
    Serial.print(epoch);
    Serial.print(" | Avg. loss: ");
    Serial.println(epochLoss / numBatches);
  }
}

void loop() {}
