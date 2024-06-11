#pragma once
#include <iostream>
#include <math.h>
#include <vector>
#include "linalg_ops.h"

class Embedding{
    public:

        int vocab_size, em_size;

        std::vector<std::vector<float>> weight;

        Embedding(int num_embedding=10, int embedding_dim=8){
            int i, j;
            vocab_size = num_embedding;
            em_size = embedding_dim;
            weight = std::vector<std::vector<float>>(num_embedding, std::vector<float>(embedding_dim, 0.0f));
            for(i=0; i<num_embedding; i++){
                for(j=0; j<embedding_dim; j++){
                    weight[i][j] = random_value();
                }
            }
        }

        Embedding operator=(Embedding embed){
            vocab_size = embed.vocab_size;
            em_size = embed.em_size;
            weight = embed.weight;
            return *this;
        }

        std::vector<std::vector<float>> forward(std::vector<int> x){
            std::vector<std::vector<float>> one_hot_encoding(x.size(), std::vector<float>(vocab_size, 0.0f));
            int i;
            for(i=0; i<x.size(); i++){
                one_hot_encoding[i][x[i]] = 1.0f;
            }
            return matmul(one_hot_encoding, weight);
        }
};