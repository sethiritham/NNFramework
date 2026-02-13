#include <iostream>
#include <vector>
#include <cmath>
#include "math/activation_functions.hpp"
#include <memory>

class NeuralNetwork
{
    public:
        std::size_t in_features_;
        std::size_t out_features_;
        std::size_t num_hidden_layers_;
        std::size_t batch_size_;

        std::vector<int> layer_sizes_;
        std::vector<Matrix> input_cache_;
        std::vector<Matrix> biases_;
        std::vector<Matrix> weights_;

        const double learning_rate;
        Matrix actual_prediction;

    private:
        /**
         *  @brief Initial weights for ReLU 
         * √(2/in_features)
         *  @param weights Weights for each layer, ordered input to output 
         */
        void update_init_weights_ReLU(std::vector<Matrix>& weights);

        /**
         * @brief Gradient is d(Loss), activation for ReLU : gradient[i] = 0 if pred[i] = 0
         * @param pred The prediction matrix
         * @param actual The true prediction 
         */
        Matrix calculate_and_filter_gradient(Matrix& pred, Matrix& loss);

        double convert_loss_to_gradient(double element) 
        { 
            return std::pow(element, 0.5) * 2.0; 
        }    

    public:
        /**
         * @brief Initiates the Neural Network
         * weights, biases, inputs are used by the backprop and hence are cached at each layer
         * @param input in_features
         * @param output out_features
         * @param hiddenL num_hidden_layers
         * @param batchS batch_size
         */
        NeuralNetwork(int input, int output, int hiddenL, int batchS, double lr);

        /**
         * @brief Initiates the Neural Network
         * weights, biases, inputs are used by the backprop and hence are cached at each layer
         * @param input in_features
         * @param output out_features
         * @param hidden_sz size of each hidden layer
         * @param batchS batch_size
         */
        NeuralNetwork(int input, int output, std::vector<int> hidden_sz, int batchS, double lr);
        
        /**
         * @brief executes the forward pass and returns the prediction
         */
        Matrix forward_pass(Matrix &inputs);

        /**
         * @brief ReLU loss function
         */
        Matrix loss_fn(Matrix& pred, Matrix& actual);

        /**
         * @brief exectues backward pass, weights and biases updated
         */
        void backward_pass();
};

