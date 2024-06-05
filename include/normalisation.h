#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <climits>
#include <linalg_ops.h>


class LayerNormalization{
    int rows, cols;
    std::vector<std::vector<float>> gamma, beta;
    LayerNormalization(int height, int width){
        rows = height;
        cols = width;
        int i, j;
        for(i=0; i<height; i++){
            gamma.push_back({1.0f});
            beta.push_back({0.0f});
            for(j=0; j<width-1; j++){
                gamma.back().push_back(1.0f);
                beta.back().push_back(0.0f);
            }
        }
    }

    std::vector<std::vector<float>> forward(std::vector<std::vector<float>> x){
        float standard_deviation, mean_value, epsilon, lower_bound;
        epsilon = 0.00001f;
        standard_deviation = std(x);
        mean_value = mean(x);
        lower_bound = std::sqrt(standard_deviation*standard_deviation+epsilon);
        int i, j;
        for(i=0; i<x.size(); i++){
            for(j=0; j<x[i].size(); j++){
                x[i][j] = (x[i][j] - mean_value)/lower_bound*gamma[i][j];
            }
        }

        x = add(x, beta);
        return x;
    }
};