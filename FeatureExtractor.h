#pragma once
#include <math.h>
// to simplify the trees, we extract 4 stats from fetures pr channel: mean std RMS and max
// input: float[N_channels * window]
// output: float[N_channels * 4] 224 features for 56 channels

const int N_Features_Pr_Channel = 4;

void extractFeatures(const float* signal, int nChannels, int window, float* out){
    for (int ch = 0; ch < nChannels; ch++){

        const float* seg = signal + ch * window;

        float mean = 0, rms = 0, maxVal = seg[0];

        for (int i = 0; i < window; i++){

            mean+= seg[i];
            rms+= seg[i] * seg[i];

            if (seg[i] >maxVal) maxVal = seg[i];
        }
        mean /= window;
        rms = sqrtf(rms/window);

        float var = 0;
        for (int i = 0; i < window; i++){
            float d = seg[i] - mean;
            var += d*d;
        }
        float std = sqrtf(var / window);

        int base = ch * N_Features_Pr_Channel;
        out[base + 0] = mean;
        out[base + 1] = std;
        out[base + 2] = rms;
        out[base + 3] = maxVal;
    }
}