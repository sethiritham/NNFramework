#include <iostream>
#include <vector>
#include <cmath>
#include "math/activation_functions.hpp"
#include <memory>
#define LOG(x) std::cout << x << std::endl

class NeuralNetwork
{
    private:
        int input_size_;
        int output_size_;
        int hidden_layers_;
        int batch_size_;
        std::vector<Matrix> input_cache_;
        std::vector<int> hidden_size_;
        Matrix biases;
        std::vector<Matrix> weights_;

    private:
        std::vector<Matrix> generate_init_weights(std::vector<Matrix>& weights)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dist(input_size_, output_size_);

            for(int i = 0; i < weights.size(); i++)
            {
                for(int row = 0; row < weights[i].num_rows; row++)
                {
                    for(int col = 0; col < weights[i].num_cols; col++)
                    {
                        weights[i][row][col] = dist(gen) * std::sqrt(2.0 / input_size_);
                    }
                }
            }

            return weights;
        }

        Matrix calculate_gradient(Matrix pred, Matrix actual)
        {
            return pred - actual;
        }

        void update_gradient_using_filter(Matrix& gradient, Matrix &output)
        {
            for(int row = 0; row < output.num_rows; row++)
            {
                for(int col = 0; col < output.num_cols; col++)
                {
                    if(output[row][col] == 0.0)
                    {
                        gradient[row][col] == 0.0;
                    }
                }
            }
        }
    
    public:
        NeuralNetwork(int input, int output, int hiddenL, int batchS) 

        : input_size_(input), output_size_(output), hidden_layers_(hiddenL), 
        batch_size_(batchS), biases(batch_size_, output_size_), hidden_size_(hiddenL)

        {
            std::cout<<"Started initializer"<<std::endl;
            Matrix biases = Matrix::fill_matrix(0.0, batch_size_, output_size_);

            weights_.reserve(hidden_layers_ + 1); 
            input_cache_.reserve(hidden_layers_ + 1);

            int x  = (int)((input - output)/(hiddenL + 1));

            int prev = input;

            for (int i = 0; i < hiddenL;  i++)
            {
                hidden_size_[i] = prev - x;
                prev = hidden_size_[i];
            }

            weights_.emplace_back(input, output);
            input_cache_.emplace_back(batch_size_, input);
            weights_ = generate_init_weights(weights_);
        }

        NeuralNetwork(int input, int output, std::vector<int> hidden_sz) 
        : input_size_(input), output_size_(output), hidden_size_(hidden_sz), biases(batch_size_, output_size_)
        {
            weights_.reserve(hidden_layers_ + 1); 
            input_cache_.reserve(hidden_layers_ + 1);

            weights_.emplace_back(input, output);  
            input_cache_.emplace_back(batch_size_, input);
            Matrix biases = Matrix::fill_matrix(0.0, batch_size_, output_size_);
            weights_ = generate_init_weights(weights_);

            hidden_layers_ = hidden_size_.size();
        }

        
        
        Matrix forward_pass(Matrix &inputs)
        {
            LOG("STARTED FORWARD PASS");
            input_cache_.clear(); //cleaing the input cache at the start of the forward pass 

            Matrix prev_pred = inputs;

            for(int i = 0; i < hidden_layers_ + 1; i++)
            {
                LOG("TRYING WEIGHTS");
                weights_[i].display_matrix();

                (prev_pred*weights_[i]).display_matrix();
                prev_pred = prev_pred*weights_[i] + biases;
                update_using_ReLU(prev_pred);
                input_cache_.emplace_back(prev_pred);
                std::cout<<"LOOP "<<i<<" COMPLETED"<<std::endl;
            }

            prev_pred.display_matrix();
            return prev_pred;
        }



        Matrix loss_fn(Matrix& pred, Matrix& actual)
        {
            Matrix loss_matrix(pred.num_rows, pred.num_cols);
            for(int i  = 0; i < pred.num_rows; i++)
            {
                for(int j = 0; j < pred.num_cols; j++)
                {
                    loss_matrix[i][j] = (std::pow((pred[i][j] - actual[i][j]), 2.0))/2.0;
                }
            }
            return loss_matrix;
        }

        void backward_pass(std::vector<Matrix>& inputs, std::vector<Matrix>& weights, Matrix& filtered_gradient, int batch_size)
        {
            
        }
};

