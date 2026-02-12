#include <iostream>
#include <vector>
#include <cmath>
#include "math/activation_functions.hpp"
#include <memory>

class NeuralNetwork
{
    private:
        int in_features_;
        int out_features_;
        int num_hidden_layers_;
        int batch_size_;
        std::vector<int> hidden_layer_sizes_;

        std::vector<Matrix> input_cache_;
        std::vector<Matrix> biases_;
        std::vector<Matrix> weights_;

    private:
        void update_init_weights_ReLU(std::vector<Matrix>& weights);
        Matrix calculate_gradient(Matrix pred, Matrix actual);
        void update_gradient_using_filter(Matrix& gradient, Matrix &output);
    
    public:
        NeuralNetwork(int input, int output, int hiddenL, int batchS);

        NeuralNetwork(int input, int output, std::vector<int> hidden_sz, int batchS);
        
        Matrix forward_pass(Matrix &inputs);

        Matrix loss_fn(Matrix& pred, Matrix& actual);

        void backward_pass(std::vector<Matrix>& inputs, std::vector<Matrix>& weights, Matrix& filtered_gradient, int batch_size);
};

