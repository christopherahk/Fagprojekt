#include "Model.h"
#include "SGD.h"
#include "Tensor.h"
#include "functions.h"

#define N_INPUTS 10
#define N_CHANNELS 56
#define N_CLASSES 2
#define BATCH_SIZE 1
#define N_HIDDEN 64

Model model(N_INPUTS, N_HIDDEN, BATCH_SIZE, N_CLASSES);
SGD optimizer1(0.01);
SGD optimizer2(0.01);

Tensor batchX(N_CHANNELS, N_INPUTS);
Tensor batchY(1, N_CLASSES);

String inputLine;

void setup() {
  Serial.begin(115200);
  while (!Serial)
    ;
}

void loop() {
  // Wait until a full line arrives
  if (Serial.available()) {
    inputLine = Serial.readStringUntil('\n');

    if (inputLine.length() == 0)
      return;

    int startIndex = 0;
    int spaceIndex = inputLine.indexOf(' ');
    float label = 0.0f;

    while (spaceIndex != -1) {
      String word = inputLine.substring(startIndex, spaceIndex);

      if (startIndex = N_INPUTS * N_CHANNELS) {
        label = word.toFloat();
        break;
      }

      batchX(startIndex / 56, startIndex % 56) = word.toFloat();

      startIndex = spaceIndex + 1;
      spaceIndex = inputLine.indexOf(' ', startIndex);
    }

    for (int c = 0; c < N_CLASSES; c++) {
      batchY(0, c) = 0.0f;
    }
    batchY(0, label) = 1.0f;

    // Train step
    model.forward(batchX);

    float loss = model.loss(batchY);

    model.backward(batchY);

    optimizer2.updateParameters(model.layer2);
    optimizer1.updateParameters(model.layer1);

    // Send loss back to Python
    Serial.println(loss, 6);
  }
}
