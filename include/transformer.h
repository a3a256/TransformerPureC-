#pragma once
#include <iostream>
#include <vector>
#include "mha.h"
#include "normalisation.h"
#include "position_encoding.h"
#include "embedding.h"

class EmbeddingEncoding{
    public:
        int sequence_len, embedding_dim, embedding_size;
        Embedding embed;
        std::vector<std::vector<float>> position_encoder;
        EmbeddingEncoding(int seq_len=512, int em_size=64, int vocab_size=1000, int n=10000){
            embedding_dim = em_size;
            embedding_size = vocab_size;
            sequence_len = seq_len;
            position_encoder = position_encoding(embedding_dim, sequence_len, n);
            embed = Embedding(embedding_size, embedding_dim);
        }

        EmbeddingEncoding operator=(EmbeddingEncoding encode){
            sequence_len = encode.sequence_len;
            embedding_dim = encode.embedding_dim;
            embedding_size = encode.embedding_size;
            embed = encode.embed;
            position_encoder = encode.position_encoder;
        }

        std::vector<std::vector<float>> forward(std::vector<int> x){
            std::vector<std::vector<float>> y;
            y = embed.forward(x);
            return add(y, position_encoder);
        }
};

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

    public:
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

    public:

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
            x = add(x, mha2.forward(x, encoded, encoded));
            x = norm2.forward(x);
            x = add(x, ffn.forward(x));
            x = norm3.forward(x);
            return x;
        }
};

class Encoder{

    public:

        std::vector<EncoderLayer> enc_layers;

        EmbeddingEncoding pos_encoder;
        
        Encoder(int sequence_len=512, int em_size=64, int vocab_size=1000, int num_heads=8, int hidden_neurons=2048, int num_layers=6, int n=10000){
            int i;
            for(i=0; i<num_layers; i++){
                EncoderLayer enc_layer(sequence_len, em_size, num_heads, hidden_neurons);
                enc_layers.push_back(enc_layer);
            }

            pos_encoder = EmbeddingEncoding(sequence_len, em_size, vocab_size, n);
        }

        Encoder operator=(Encoder encode){
            enc_layers = encode.enc_layers;
            pos_encoder = encode.pos_encoder;

            return *this;
        }

        std::vector<std::vector<float>> forward(std::vector<int> x){
            int i;
            std::vector<std::vector<float>> y;
            y = pos_encoder.forward(x);
            for(i=0; i<enc_layers.size(); i++){
                y = enc_layers[i].forward(y);
            }
            return y;
        }
};

class Decoder{

    public:

        std::vector<DecoderLayer> decoder_layers;

        EmbeddingEncoding pos_encoder;

        Decoder(int sequence_len=512, int em_size=64, int vocab_size=1000, int num_heads=8, int hidden_neurons=2048, int num_layers=6, int n=10000){
            int i;
            for(i=0; i<num_layers; i++){
                DecoderLayer decode(sequence_len, em_size, num_heads, hidden_neurons);
                decoder_layers.push_back(decode);
            }

            pos_encoder = EmbeddingEncoding(sequence_len, em_size, vocab_size, n);
        }

        Decoder operator=(Decoder decode){
            decoder_layers = decode.decoder_layers;
            pos_encoder = decode.pos_encoder;

            return *this;
        }

        std::vector<std::vector<float>> forward(std::vector<int> x, std::vector<std::vector<float>> context){
            int i;

            std::vector<std::vector<float>> y;

            y = pos_encoder.forward(x);


            for(i=0; i<decoder_layers.size(); i++){
                y = decoder_layers[i].forward(y, context);
            }

            return y;
        }

};



class Transformer{
    public:

        Encoder encode;
        Decoder decode;

        Linear fc;

        Transformer(int sequence_len, int em_size, int num_heads, int hidden_neurons, int num_layers, int n, int tokens){

            encode = Encoder(sequence_len, em_size, num_heads, hidden_neurons, num_layers, n);

            decode = Decoder(sequence_len, em_size, num_heads, hidden_neurons, num_layers, n);

            fc = Linear(em_size, tokens);
        }

        std::vector<std::vector<float>> forward(std::vector<int> x, std::vector<int> y){
            std::vector<std::vector<float>> context;
            std::vector<std::vector<float>> output;
            context = encode.forward(x);

            output = decode.forward(y, context);

            output = fc.forward(output);

            output = softmax(output);

            return output;
        }
};