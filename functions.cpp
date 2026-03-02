#include "functions.h"
#include "Model.h"
#include "SGD.h"
#include "Tensor.h"
#include <Arduino.h>
#include <algorithm>
#include <cmath>

void oneHot(Tensor &batchY, int row, int label, int nClasses) {
  for (int c = 0; c < nClasses; c++) {
    batchY(row, c) = 0.0f;
  }

  batchY(row, label) = 1.0f;
}

void copyBatch(const float *train_data, const float *train_labels,
               int startIndex, Tensor &batchX, Tensor &batchY, int nInputs,
               int nClasses, int batchSize) {
  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < nInputs; j++) {
      batchX(i, j) = train_data[(startIndex + i) * nInputs + j];
    }

    int label = (int)train_labels[startIndex + i];
    oneHot(batchY, i, label, nClasses);
  }
}

float *set_labels(const float *data, float *labels, const int rows,
                  const int cols) {
  for (int channel = 0; channel < rows; channel++) {
    for (int index = 0; index < cols; index++) {
      int idx = channel * cols + index;

      labels[idx] = (float)(abs(data[idx]) >= 25);
    }
  }
}

void training_loop(int epochs, int trainingSamples, int batchSize,
                   float *train_data, float *train_labels, Tensor &batchX,
                   Tensor &batchY, int nInputs, int nClasses, Model &model,
                   SGD &optimizer1, SGD &optimizer2) {
  /*
  Serial.print("Starting training for ");
  Serial.print(epochs);
  Serial.println(" epochs.");
  */

  for (int epoch = 1; epoch <= epochs; epoch++) {
    float epochLoss = 0.0f;
    int numBatches = trainingSamples / batchSize;

    for (int batch = 0; batch < numBatches; batch++) {
      /*
      Serial.print("Doing batch number: ");
      Serial.print(batch + 1);
      Serial.print("/");
      Serial.print(numBatches);
      Serial.print(" | Epoch: ");
      Serial.println(epoch);
      */
      int startIndex = batch * batchSize;
      copyBatch(train_data, train_labels, startIndex, batchX, batchY, nInputs,
                nClasses, batchSize);

      model.forward(batchX);
      float loss = model.loss(batchY);
      epochLoss += loss;

      model.backward(batchY);

      optimizer2.updateParameters(model.layer2);
      optimizer1.updateParameters(model.layer1);
    }
    /*
    Serial.print("Epoch: ");
    Serial.print(epoch);
    Serial.print(" | Avg. loss: ");
    */
    Serial.println(epochLoss / numBatches);
  }
}
