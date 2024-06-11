#pragma once
#include <iostream>
#include <vector>
#include <math.h>

std::vector<std::vector<float>> position_encoding(int seq_len, int em_size, int n){
    std::vector<std::vector<float>> encoding(seq_len, std::vector<float>(em_size, 0.0f));
    int i, j;
    float upper, lower;
    for(i=0; i<seq_len; i++){
        for(j=0; j<em_size; j+=2){
            upper = (float)i;
            lower = (float)std::pow(n, (float)(2*j)/(float)seq_len);
            encoding[i][j] = std::sin(upper/lower);
            encoding[i][j+1] = std::cos(upper/lower);
        }
    }

    return encoding;
}