#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <random>
#include <climits>
#include <numeric>

std::vector<std::vector<float>> matmul(std::vector<std::vector<float>> one, std::vector<std::vector<float>> two){
    int i, j, k;
    std::vector<float> temp;
    std::vector<std::vector<float>> res;
    float _sum = 0.0f;
    for(i=0; i<one.size(); i++){
        for(j=0; j<two[0].size(); j++){
            _sum = 0.0f;
            for(k=0; k<one[0].size(); k++){
                _sum += one[i][k]*two[k][j];
            }
            temp.push_back(_sum);
        }
        res.push_back(temp);
        std::vector<float>().swap(temp);
    }
    return res;
}

std::vector<std::vector<float>> softmax(std::vector<std::vector<float>> &mat){
    float _sum;
    int i, j;
    for(i=0; i<mat.size(); i++){
        _sum = 0.0f;
        for(j=0; j<mat[i].size(); j++){
            _sum += std::exp(mat[i][j]);
        }
        for(j=0; j<mat[i].size(); j++){
            mat[i][j] = mat[i][j]/_sum;
        }
    }

    return mat;
}

std::vector<std::vector<float>> transpose(std::vector<std::vector<float>> mat){
    int i, j;
    std::vector<std::vector<float>> res(mat[0].size(), std::vector<float>(mat.size(), 0.0f));
    for(j=0; j<mat[0].size(); j++){
        for(i=0; i<mat.size(); i++){
            res[j][i] = mat[i][j];
        }
    }

    return res;
}

float std(std::vector<std::vector<float>> x){
    float mean_val = mean(x);
    float diff_sum = 0.0f;
    int i, j, count;
    count = 0;
    for(i=0; i<x.size(); i++){
        for(j=0; j<x[i].size(); j++)
            diff_sum += x[i][j] - mean_val;
            count += 1;
    }
    diff_sum = diff_sum/(float)count;
    return std::sqrt(diff_sum);
}

float mean(std::vector<std::vector<float>> x){
    float total_val;
    total_val = 0.0f;
    int i, j, count;
    count = 0;
    for(i=0; i<x.size(); i++){
        for(j=0; j<x[i].size(); j++){
            total_val += x[i][j];
        }
    }
    return total_val/(float)count;
}

float random_value(){
    std::random_device seeder;
    std::mt19937 rng(seeder());
    std::uniform_int_distribution<long> gen(INT_MIN, INT_MAX);
    return (float)gen(rng)/(float)RAND_MAX;
}

std::vector<std::vector<float>> add(std::vector<std::vector<float>> x, std::vector<std::vector<float>> y){
    int i, j;
    for(i=0; i<x.size(); i++){
        for(j=0; j<x[i].size(); j++){
            x[i][j] += y[i][j];
        }
    }
    return x;
}