#include "streaming.h"
#include <Arduino.h>

void setup() {
  Serial.begin(115200);
  initNetwork();
}

void loop() {
  get_input();

  if (input_is_ready()) {
    forwardpass();
    output();

    if (labels_available()) {
      update_weights();
    }

    set_input_empty();
  }
}
