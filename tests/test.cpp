#include "Tensor.h"
#include <Arduino.h>
#include <unity.h>

float &testTensorInput(Tensor t) { return t(0, 0); }

void testTensor() {
  float arr[4] = {0.0f, 1.0f, 2.0f, 3.0f};
  Tensor t(arr, 2, 2);

  float &value = testTensorInput(t);

  TEST_ASSERT_EQUAL_FLOAT(0.0f, value);
}
