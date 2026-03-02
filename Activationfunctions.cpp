#include <cmath>

//Sigmoid activationfunction and derivative

void sigmoid(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        float x = arr[i];
        
        if (x >= 0.0f) {
            arr[i] = 1.0f / (1.0f + std::exp(-x));
        } 
        else {
            float z = std::exp(x);
            arr[i] = z / (1.0f + z);
        }
    }
}

void sigmoid_dev(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = arr[i] * (1.0f - arr[i]);
    }
}

//Relu activationfunction and derivative
void relu(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] < 0.0f) {
            arr[i] = 0.0f;
        }
    }
}

void relu_dev(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] < 0.0f) {
            arr[i] = 0.0f;
        }
        else {arr[i] = 1.0f;
        }
    }
}

//Leaky relu activationfunction and derivative
void leaky_relu(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] < 0.0f) {
            arr[i] = 0.01f * arr[i];
        }
    }
}

void leaky_relu_dev(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] < 0.0f) {
            arr[i] = 0.01f;
        } else {
            arr[i] = 1.0f;
        }
    }
}

//Softmax activationfunction
void softmax(float* arr, int size) {
    float sum = 0.0f;
    
    for (int i = 0; i < size; i++) {
        arr[i] = std::exp(arr[i]);
        sum += arr[i];
    }
    
    for (int i = 0; i < size; i++) {
        arr[i] = arr[i] / sum;
    }
}