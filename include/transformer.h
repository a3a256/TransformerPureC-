#pragma once
#include <iostream>
#include <vector>
#include "mha.h"
#include "normalisation.h"

class FeedForwardNetwork{

    public:


        std::vector<Linear> layers;


        FeedForwardNetwork(int in_channels=64, int out_channels=128){
            Linear layer1(in_channels, out_channels);
            Linear layer2(out_channels, in_channels);
            layers.push_back(layer1);
            layers.push_back(layer2);
        }

        FeedForwardNetwork operator=(FeedForwardNetwork ffn){
            layers = ffn.layers;

            return *this;
        }

        std::vector<std::vector<float>> forward(std::vector<std::vector<float>> x){
            x = layers[0].forward(x);
            x = relu(x);
            x = layers[1].forward(x);
            return x;
        }
};

class EncoderLayer{
    std::vector<std::vector<float>> mat;
    int embedding, heads;
    MultiHeadAttention mha;
    LayerNormalization norm1, norm2;
    FeedForwardNetwork ffn;
    EncoderLayer(int sequence_len, int em_size, int num_heads, int hidden_neurons){
        embedding = em_size;
        heads = num_heads;
        mha = MultiHeadAttention(em_size, num_heads);
        norm1 = LayerNormalization(sequence_len, em_size);
        norm2 = LayerNormalization(sequence_len, em_size);
        ffn = FeedForwardNetwork(em_size, hidden_neurons);
    }

    std::vector<std::vector<float>> forward(std::vector<std::vector<float>> x){
        x = add(x, mha.forward(x, x, x));
        x = norm1.forward(x);
        x = add(x, ffn.forward(x));
        x = norm2.forward(x);
    }
};