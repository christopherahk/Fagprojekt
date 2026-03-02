#pragma once
#include "Model.h"
#include "SGD.h"
#include "Tensor.h"

void oneHot(Tensor &batchY, int row, int label, int nClasses);

void copyBatch(const float *train_data, const float *train_labels,
               int startIndex, Tensor &batchX, Tensor &batchY, int nInputs,
               int nClasses, int batchSize);

float *set_labels(const float *data, float *labels, const int rows,
                  const int cols);

void training_loop(int epochs, int trainingSamples, int batchSize,
                   float *train_data, float *train_labels, Tensor &batchX,
                   Tensor &batchY, int nInputs, int nClasses, Model &model,
                   SGD &optimizer1, SGD &optimizer2);
