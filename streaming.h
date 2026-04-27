#pragma once

void initNetwork();
void get_input();
bool input_is_ready();
void forwardpass();
void output();
bool labels_available();
void update_weights();
void set_input_empty();
void report_fold_results();
