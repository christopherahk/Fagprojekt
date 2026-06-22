## Code Description

This part is intended for documenting the code that was utilized for the project.

* **C++ Part:** Responsible for the computation on the Arduino itself.
* **Python Files:** Responsible for data preprocessing and streaming.

---

## C++ Code

### Tensors

We started out by constructing a **tensor struct** that serves as one of the most basic building blocks for the whole network. This is a basic 2D tensor featuring the necessary constructors, destructors, operators, and essential matrix operations.

To use a tensor, you must specify its dimensions before assigning data to it. The supported matrix operations include:

* Matrix multiplication and transposing.
* Summing of rows and columns.
* Computing the mean of each tensor (the sum of all indices divided by the total matrix size).
* True class selection (identifying which indices in a matrix are equal to `1.0`).
* Argmax for a row.
* Element-wise application of a function.

The tensor data is stored as a **1D float array** to optimize memory. It utilizes a custom operator responsible for mapping row and column indices to imitate 2D array functionality without the associated memory overhead.

### Convolutional Layers

This struct handles the forward pass, backward pass, and the updating of weights and biases.

The **forward pass** performs a standard 2D convolution. For each output channel and spatial position, it accumulates the dot product of the kernel weights against the corresponding input patch, plus a bias. Zero-padding is handled implicitly; any input index falling outside the bounds is skipped (contributing zero).

The **backward pass** computes three distinct gradients:

* **dBiases:** The sum of upstream gradients over all spatial positions for each output channel.
* **dWeights:** For each kernel weight, it accumulates $\text{upstream grad} \times \text{input value}$ at the corresponding input position.
* **dInputs:** Scatters the upstream gradient back to input positions, weighted by the kernel value. This represents the transposed convolution.

**Architecture Quirk:** Since the struct performs 2D convolution, one would expect an input channel dimension. It lacks one, meaning it only supports single-channel inputs. This remains from an earlier design phase before switching to a 1D Convolutional Neural Network (CNN). In the current implementation, we simulate 1D convolution using this 2D layer by encoding channels in the spatial height dimension rather than as actual input channels. This quirk only persists if there is an intention to use multiple channels.

Both the convolutional and dense layers utilize the **Adam optimizer** via an independent `update` method using the following logic:

* Gradient clipping to ±100 before updating moments, alongside a NaN/Inf guard.
* First and second moment estimates with $\beta_1 = 0.9$ and $\beta_2 = 0.999$.
* Bias correction via `bc1` and `bc2`.

### Dense Layers

In this struct, the **forward pass** is a standard affine transformation:

$$\text{output} = \text{input} \times \text{weights} + \text{bias}$$

The weight matrix has the shape `(inputCount, neuronCount)`, meaning inputs of shape `(batch, inputCount)` produce outputs of shape `(batch, neuronCount)`.

The **backward pass** computes:

* **dWeights:** $\text{input}^T \times \text{dValues}$
* **dBiases:** Column-wise sum of upstream gradients across the batch.
* **dInputs:** $\text{dValues} \times \text{weights}^T$

Weight and bias updates follow the same Adam optimizer logic detailed above.

### Activation Functions

While ReLU, Softmax, Sigmoid, and LeakyReLU are all implemented, only ReLU and Softmax from the `Activations.cpp` file are actively used:

* **ReLU:** Contains forward and backward methods. The forward pass applies ReLU element-wise to the inputs, while the backward pass computes the derivative (returning either 0 or 1 depending on whether the neuron fired).
* **Softmax:** Only handles the forward pass in this file, implementing the standard formula:

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum^K_{j=1}e^{z_j}}$$



The corresponding backward pass is handled externally by `GetLoss.cpp` (computed as the softmax output minus the true value, divided by the number of rows).

### Loss

The system uses **Categorical Cross-Entropy Loss** with a softmax application prior to calculation.

1. Values are clipped between $10^{-7}$ and $1 - 10^{-7}$ to avoid numerical instability.
2. The true class is selected from predictions based on a one-hot encoded label tensor.
3. The final loss is calculated by taking the negative logarithm of this selected value.

### Arduino File

The core network is housed within the Arduino (`.ino`) file. This file defines hyperparameters, global variables for tracking batches, the current learning rate, running loss, and placeholder tensors. It also initializes the layers, activations, and loss mechanics.

#### Data Streaming

Data is received from a Python script (`streamer.py`) in chunks of 256 bytes—determined by testing to be the maximum stable chunk size for this setup. The Arduino sends an acknowledgment back to Python after receiving each chunk to prompt the next. Once all chunks arrive, they are parsed into a float array for window processing.

#### Window Processing & Backpropagation

* **Forward Pass:** The processing starts by assigning the data to the input tensor and creating a one-hot encoded tensor for the true label. The forward pass runs through two convolutional layers with ReLU activations, followed by a 1D Global Average Pooling step.
* **Pooling:** For each filter, pooling averages values across the time dimension to collapse the shape to `(C2_FILTERS,)`.
* **Dense & Backpropagation:** This flattened tensor is fed into the dense layers. If the model is not in a testing phase, backpropagation is triggered (including the backward pass for the global average pooling).
* **Batching:** To simulate batching, the learning rate is scaled by the batch size before updating weights. The resulting probabilities are streamed back to Python.

#### State Triggers

The Arduino script changes behavior based on single-character command codes sent from Python:

* **`'E'` (Export):** Signals the model to export. All tensors are printed and written into a separate header file. While primitive, advanced weight/bias compression fell outside the scope of this project.
* **`'V'` (Validation/Testing):** Flags that testing should start. Window processing resumes, but backpropagation is completely bypassed to calculate validation loss and accuracy.
* **`'D'` (Done):** Finalizes the run. The average loss and final accuracy are transmitted back to Python for storage, and all relevant testing variables are reset.

*Note: If none of these control characters are received, the system continues processing data normally. The script also includes functionality to load pre-saved weights and biases directly.*
