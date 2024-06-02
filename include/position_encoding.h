#pragma once
#include <iostream>
#include <vector>
#include <math.h>

std::vector<std::vector<float>> position_encoding(int em_size, int seq_len, int n){
    std::vector<std::vector<float>> encoding(em_size, std::vector<float>(seq_len, 0.0f));
    int i, j;
    float upper, lower;
    for(i=0; i<em_size; i++){
        for(j=0; j<seq_len; j+=2){
            upper = (float)i;
            lower = (float)std::pow(n, (float)(2*j)/(float)seq_len);
            encoding[i][j] = std::sin(upper/lower);
            encoding[i][j+1] = std::cos(upper/lower);
        }
    }

    return encoding;
}