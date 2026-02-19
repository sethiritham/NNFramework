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
         * @brief Gradient is d(Loss), activation for ReLU : gradient[i][j] = 0 if pred[i][j] = 0
         * @param grad_output gradient output after forward pass
         * @param pred prediction matrix  
         */
        Matrix calculate_and_filter_gradient_ReLU(Matrix& grad_output, Matrix& pred);

        /**
         * @brief Gradient is d(Loss), activation for sigmoid : gradient[i][j] = sigmoid(gradient[i][j])*(1 - gradient[i][j])
         * @param grad_output gradient output after forward pass
         */
        Matrix calculate_and_filter_gradient_sigmoid(Matrix& grad_output);

        /**
         * @brief Softmax + CrossEntropy loss gradient = pred - actual
         * @param grad_output gradient output after forward pass
         */
        Matrix calculate_and_filter_gradient_softmax(Matrix& pred);

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
        NeuralNetwork(std::size_t input, std::size_t output, std::size_t hiddenL, std::size_t batchS, double lr);

        /**
         * @brief Initiates the Neural Network
         * weights, biases, inputs are used by the backprop and hence are cached at each layer
         * @param input in_features
         * @param output out_features
         * @param hidden_sz size of each hidden layer
         * @param batchS batch_size
         */
        NeuralNetwork(std::size_t input, std::size_t output, std::vector<int> hidden_sz, std::size_t batchS, double lr);
        
        /**
         * @brief executes the forward pass and returns the prediction
         */
        Matrix forward_pass(Matrix &inputs);

        /**
         * @brief ReLU loss function
         */
        double loss_fn(const Matrix& pred,const Matrix& actual);

        /**
         * @brief Cross entropy loss function
         */
        double cross_entropy_loss(Matrix& log_pred, Matrix& actual);

        /**
         * @brief exectues backward pass, weights and biases updated
         */
        void backward_pass();
};