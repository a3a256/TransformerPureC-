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

        return x;
    }
};

class DecoderLayer{
    MultiHeadAttention mha1, mha2;
    LayerNormalization norm1, norm2, norm3;
    FeedForwardNetwork ffn;
    DecoderLayer(int sequence_len, int em_size, int num_heads, int hidden_neurons){
        mha1 = MultiHeadAttention(em_size, num_heads);
        mha2 = MultiHeadAttention(em_size, num_heads);
        norm1 = LayerNormalization(sequence_len, em_size);
        norm2 = LayerNormalization(sequence_len, em_size);
        norm3 = LayerNormalization(sequence_len, em_size);
        ffn = FeedForwardNetwork(em_size, hidden_neurons);
    }

    std::vector<std::vector<float>> forward(std::vector<std::vector<float>> x, std::vector<std::vector<float>> encoded){
        x = add(x, mha1.forward(x, x, x));
        x = norm1.forward(x);
        x = add(x, mha2.forward(x, x, encoded));
        x = norm2.forward(x);
        x = add(x, ffn.forward(x));
        x = norm3.forward(x);
        return x;
    }
};