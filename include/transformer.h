#pragma once
#include <iostream>
#include <vector>
#include "mha.h"

class FeedForwardNetwork{
    std::vector<Linear> layers;
    FeedForwardNetwork(int in_channels, int out_channels){
        Linear layer1(in_channels, out_channels);
        Linear layer2(out_channels, in_channels);
        layers.push_back(layer1);
        layers.push_back(layer2);
    }

    std::vector<std::vector<float>> forward(std::vector<std::vector<float>> x){
        x = layers[0].forward(x);
        x = relu(x);
        x = layers[1].forward(x);
        return x;
    }
};

class EncoderLayer{};