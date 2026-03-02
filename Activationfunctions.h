#pragma once

// Activation functions
void sigmoid(float *arr,
             int size); // Applies sigmoid activation function to array
void relu(float *arr,
          int size); // Applies ReLU (Rectified Linear Unit) activation to array
void leaky_relu(float *arr, int size); // Applies Leaky ReLU activation to array
void softmax(float *arr,
             int size); // Applies softmax activation to normalize probabilities

// Derivatives
void sigmoid_dev(float *arr,
                 int size);          // Computes derivative of sigmoid function
void relu_dev(float *arr, int size); // Computes derivative of ReLU function
void leaky_relu_dev(float *arr,
                    int size); // Computes derivative of Leaky ReLU function
