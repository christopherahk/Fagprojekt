#include "Activation_Softmax_CategoricalCrossEntropy.h"
#include "Activations.h"
#include "Tensor.h"
#include <Arduino.h>
#include <math.h>

const int INPUT_ROWS = 56;
const int INPUT_COLS = 100;
const int HIDDEN_NEURONS = 16;
const int OUTPUT_CLASSES = 3;
const float LEARNING_RATE = 0.045f;
const int TRAIN_EPOCHS = 100;
const float GRAD_CLIP = 100.0f;

float clampValue(float x,
                 float limit) { // Makes sure x is between -limit and limit to
                                // prevent exploding gradients.
  if (x > limit) {
    return limit;
  }
  if (x < -limit) {
    return -limit;
  }
  return x;
}

void ensureTensorShape(Tensor &t, int rows,
                       int cols) { // Ensures the tensor has the specified
                                   // shape, reallocating if necessary.
  if (t.rows != rows || t.cols != cols) { // if the shape is different
    t = Tensor(rows, cols);               // reallocate to the new shape
  }
}

struct DenseLayer { // i am dense
  Tensor weights;
  Tensor biases;
  const Tensor *inputs;
  Tensor output;
  Tensor dWeights;
  Tensor dBiases;
  Tensor dInputs;

  DenseLayer(int inputCount, int neuronCount) //
      : weights(inputCount, neuronCount), biases(1, neuronCount),
        inputs(nullptr), output(), dWeights(), dBiases(), dInputs() {
    // Random small initialization to break symmetry.
    for (int r = 0; r < inputCount; r++) {
      for (int c = 0; c < neuronCount; c++) {
        long rnd = random(-100, 101);
        weights(r, c) = static_cast<float>(rnd) / 1000.0f;
      }
    }

    for (int c = 0; c < neuronCount; c++) {
      biases(0, c) = 0.0f;
    }
  }

  void forward(const Tensor &x) {
    inputs = &x;
    ensureTensorShape(output, x.rows, weights.cols);

    for (int r = 0; r < x.rows; r++) {
      for (int c = 0; c < weights.cols; c++) {
        float sum = 0.0f;
        for (int k = 0; k < x.cols; k++) {
          sum += x(r, k) * weights(k, c);
        }
        output(r, c) = sum + biases(0, c);
      }
    }
  }

  void backward(const Tensor &dValues, bool computeDInputs = true) {
    if (inputs == nullptr) {
      // If backward is called before forward,
      // we can't compute gradients, so we just return.
      return;
    }

    ensureTensorShape(dWeights, inputs->cols,
                      dValues.cols); // Ensure dWeights has the correct shape
                                     // (input features x neurons).
    ensureTensorShape(
        dBiases, 1,
        dValues.cols); // Ensure dBiases has the correct shape (1 x neurons).

    for (int r = 0; r < inputs->cols; r++) {
      // Compute dWeights by multiplying inputs^T with dValues.
      for (int c = 0; c < dValues.cols; c++) {
        float sum = 0.0f;
        for (int k = 0; k < inputs->rows; k++) {
          sum += (*inputs)(k, r) * dValues(k, c);
        }
        dWeights(r, c) = sum;
      }
    }

    if (computeDInputs) { // Compute dInputs by multiplying dValues with
                          // weights^T.
      ensureTensorShape(dInputs, dValues.rows, weights.rows);

      for (int r = 0; r < dValues.rows; r++) {
        for (int c = 0; c < weights.rows; c++) {
          float sum = 0.0f;
          for (int k = 0; k < dValues.cols; k++) {
            sum += dValues(r, k) * weights(c, k);
          }
          dInputs(r, c) = sum;
        }
      }
    }

    for (int c = 0; c < dValues.cols;
         c++) { // Compute dBiases by summing dValues across the batch for each
                // neuron.
      float sum = 0.0f;
      for (int r = 0; r < dValues.rows; r++) {
        sum += dValues(r, c);
      }
      dBiases(0, c) = sum;
    }
  }

  void update(float learningRate) { // Update weights and biases using the
                                    // computed gradients, applying gradient
                                    // clipping to prevent exploding gradients.
    for (int r = 0; r < weights.rows; r++) {
      for (int c = 0; c < weights.cols; c++) {
        float grad = clampValue(dWeights(r, c), GRAD_CLIP);
        weights(r, c) -= learningRate * grad;
      }
    }

    for (int c = 0; c < biases.cols;
         c++) { // Update biases with gradient clipping.
      float grad = clampValue(dBiases(0, c), GRAD_CLIP);
      biases(0, c) -= learningRate * grad;
    }
  }
};

Tensor makeInputTensor() {
  Tensor x(INPUT_ROWS, INPUT_COLS);

  // Replace with your real loaded tensor values.
  for (int r = 0; r < x.rows; r++) {
    for (int c = 0; c < x.cols; c++) {
      x(r, c) = ((r + c) % 17) / 17.0f;
    }
  }

  return x;
}

Tensor makeDummyOneHotLabels(const Tensor &x) {
  Tensor y(INPUT_ROWS, OUTPUT_CLASSES);

  // Generate labels from input patterns so this sanity-check task is learnable.
  for (int r = 0; r < x.rows; r++) {
    float score0 = 1.5f * x(r, 0) + 0.9f * x(r, 7) - 0.4f * x(r, 14);
    float score1 = 1.3f * x(r, 1) + 1.0f * x(r, 8) - 0.5f * x(r, 15);
    float score2 = 1.1f * x(r, 2) + 0.8f * x(r, 9) - 0.6f * x(r, 16);

    int classIdx = 0;
    float best = score0;
    if (score1 > best) {
      best = score1;
      classIdx = 1;
    }
    if (score2 > best) {
      classIdx = 2;
    }

    for (int c = 0; c < y.cols; c++) {
      y(r, c) = (c == classIdx) ? 1.0f : 0.0f;
    }
  }

  return y;
}

float accuracyFromOneHot(const Tensor &probabilities,
                         const Tensor &yTrueOneHot) {
  int correct = 0;
  for (int r = 0; r < probabilities.rows; r++) {
    int pred = probabilities.argmaxRow(r);
    int target = yTrueOneHot.argmaxRow(r);
    if (pred == target) {
      correct++;
    }
  }

  return static_cast<float>(correct) / static_cast<float>(probabilities.rows);
}

void runTraining() {
  randomSeed(42);

  Tensor X = makeInputTensor();
  Tensor yTrue = makeDummyOneHotLabels(X);

  DenseLayer dense1(INPUT_COLS, HIDDEN_NEURONS);
  LeakyReLU activation1;
  DenseLayer dense2(HIDDEN_NEURONS, OUTPUT_CLASSES);
  Activation_Softmax_CategoricalCrossEntropy lossActivation;

  Serial.println("Starting NN training...");

  for (int epoch = 0; epoch < TRAIN_EPOCHS; epoch++) {
    yield(); // Allow background tasks to run, preventing watchdog resets during
             // long training. - according to copilot

    dense1.forward(X); // Forward pass through the first dense layer.
    activation1.forward(dense1.output,
                        0.01f); // Forward pass through LeakyReLU activation.
    dense2.forward(
        activation1.output); // Forward pass through the second dense layer.

    float loss = lossActivation.forward(
        dense2.output, yTrue); // Compute loss and softmax probabilities.
    float accuracy = accuracyFromOneHot(
        lossActivation.activation.output,
        yTrue); // Compute accuracy from the softmax output and true labels.

    if (isnan(loss) || isinf(loss)) {
      // checks if the loss is NaN or Inf,
      // which can happen due to numerical instability.
      // If so, it prints an error message and breaks the training loop to
      // prevent further issues.
      Serial.println("Numerical error: loss is NaN/Inf. Stopping training.");
      break;
    }

    // below has to be the opposite order of forward pass.
    lossActivation.backward(
        yTrue); // Backward pass through the combined softmax and loss layer to
                // compute gradients.
    dense2.backward(lossActivation.dInputs,
                    true); // Backward pass through the second dense layer to
                           // compute gradients for weights, biases, and inputs.
    activation1.backward(
        dense2.dInputs); // Backward pass through the LeakyReLU activation to
                         // compute gradients for its inputs.
    dense1.backward(activation1.dInputs,
                    false); // Backward pass through the first dense layer to
                            // compute gradients for weights and biases (no need
                            // to compute dInputs for the first layer).

    dense2.update(LEARNING_RATE);
    dense1.update(LEARNING_RATE);

    Serial.print("epoch ");
    Serial.print(epoch);
    Serial.print(" loss=");
    Serial.print(loss, 6);
    Serial.print(" acc=");
    Serial.println(accuracy, 4);

    yield();
  }

  Serial.println("Basic NN training run completed.");
}

void setup() { Serial.begin(115200); }

void loop() {
  static bool hasRun = false;

  if (!hasRun && Serial) {
    delay(200);
    runTraining();
    hasRun = true;
  }
}
