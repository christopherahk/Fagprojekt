#include "Tensor.h"

void setup() {
  Serial.begin(9600);
  while(!Serial);

  float array1[6] = {0, 1, 2, 3, 4, 5};
  float array2[6] = {5, 4, 3, 2, 1, 0};
  
  Tensor A(array1, 2, 3);
  Tensor B(array2, 3, 2);

  Tensor C = A.matmul(B);
  Tensor C_T = C.transpose();

  C_T.printTensor();
}

void loop() {
}
