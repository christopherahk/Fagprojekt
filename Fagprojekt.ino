#include "Activations.h"
#include "DenseLayer.h"
#include "GetLoss.h"
#include "Tensor.h"
#include <Arduino.h>
#include <math.h>

// Global variables and constants
const int INPUT_ROWS = 56;
const int INPUT_COLS = 100;
const int HIDDEN_NEURONS = 16;
const int OUTPUT_CLASSES = 3;
const float LEARNING_RATE = 0.045f;
const float GRAD_CLIP = 100.0f;
String line;
Tensor inputTensor(56, 100);
int currentRow = 0;
bool tensorReady = false;

void readSerialIntoTensor() {
  if (tensorReady)
    return;
  if (!Serial.available())
    return; // if no data, skip

  line = Serial.readStringUntil('\n'); // read a line of input
  line.trim();                         // remove any leading/trailing whitespace

  if (line.length() == 0)
    return; // skip empty lines
  if (line.startsWith("#META"))
    return; // skip metadata lines

  char buffer[2048]; // safer than dynamic stack
  line.toCharArray(buffer,
                   sizeof(buffer)); // convert String to char array for strtok

  int col = 0;
  char *token = strtok(buffer, ","); // split by comma

  while (token != NULL &&
         col < inputTensor.cols) { // ensure we don't exceed column bounds
    inputTensor(currentRow, col) = atof(token);
    token = strtok(NULL, ",");
    col++;
  }

  if (col != INPUT_COLS || token != NULL) {
    currentRow = 0;
    tensorReady = false;
    Serial.println("ERR_FRAME_RESET");
    return;
  }

  currentRow++;

  if (currentRow == INPUT_ROWS) {
    tensorReady = true; // nu må loop køre processFrame
  }
}

void initNetwork() {
  // initialize the neural network layers and activation functions here
  // we want to layers first with LeakyRelu and the second layer with softmax
  // activation we also want to initialize the weights and biases for the
  // layers, and set up any necessary data structures for training the output
  // should be a tensor of size 3, representing the probabilities for each class
}

bool input_is_ready() {
  // see if get input is full, and ready for forward pass.
  return tensorReady;
}

void set_input_empty() {
  // reset input tensor after training to make sure input is ready continues to
  // work correctly

  tensorReady = false;
  currentRow = 0;
}

bool labels_available() {
  // check if labels are available for training
  return true;
}

void setup() {
  Serial.begin(115200);
  initNetwork();
}

void update_weights() {
  // get the global dWeights and dBiases from the layers, and update the weights
  // and biases using the learning rate and gradient clipping use a set
}

void loop() {

  get_input(); // build the input tensor from USB data
  if (input_is_ready()) {

    forwardpass();
    output();
    if (labels_available()) {

      update_weights();
    }

    set_input_empty();
  }
}

const optimizer = "SGD";
optmizerconst optimizer = "SGD"; // can be "SGD", "ADAM", or "RMSPROP" - if it
                                 // is neither defualt is SGD
if (optimizer == "SGD") {
  // SGD update
  // w -= lr * clip(grad)
} else if (optimizer == "ADAM") {
  // Adam update
  // use m, v, t state tensors/counter
} else if (optimizer == "RMSPROP") {
  // RMSProp update
  for (int i = 0; i < 2; i++) {
    DenseLayer *L = layers[i];

    Tensor *vW = (i == 0) ? &vw1 : &vw2;
    Tensor *vB = (i == 0) ? &vb1 : &vb2;

    rmspropUpdateTensor(L->weights, L->dWeights, *vW, LEARNING_RATE, beta2,
                        eps);
    rmspropUpdateTensor(L->biases, L->dBiases, *vB, LEARNING_RATE, beta2, eps);
  }

} else {
  // fallback
  Serial.println("Unknown optimizer, defaulting to SGD");
  // run SGD
  for (int i = 0; i < 2; i++) {
    DenseLayer *L = layers[i];
    sgdUpdateTensor(L->weights, L->dWeights, LEARNING_RATE);
    sgdUpdateTensor(L->biases, L->dBiases, LEARNING_RATE);
  }
