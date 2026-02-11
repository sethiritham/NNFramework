#include <iostream>
#include <vector>
#include <cmath>
#include "math/activation_functions.hpp"
#include <memory>

class NeuralNetwork
{
    private:
        int input_size_;
        int output_size_;
        int hidden_layers_;
        int batch_size_;
        std::vector<Matrix> input_cache;
        std::vector<int> hidden_size_;
        Matrix biases;
        std::vector<Matrix> weights;

    private:
        void initialize_weights(std::vector<Matrix>& m, int input, int output)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dist(input, output);

            for(int i = 0; i < m.size(); i++)
            {
                for(int row = 0; row < m[i].num_rows; row++)
                {
                    for(int col = 0; col < m[i].num_cols; col++)
                    {
                        m[i][row][col] = dist(gen) * std::sqrt(2.0 / input);
                    }
                }
            }
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

            std::cout<<"Made bias"<<std::endl;

            initialize_weights(weights, input_size_, output_size_);

            std::cout<<"weights initialized"<<std::endl;

            int x  = (int)((input - output)/(hiddenL + 1));

            int prev = input;

            std::cout<<"Before initializing hidden_size_"<<std::endl;
            for (int i = 0; i < hiddenL;  i++)
            {
                std::cout<<"error"<<std::endl;
                std::cout<<prev-x<<std::endl;
                hidden_size_[i] = prev - x;
                std::cerr<<"PROBLEM"<<std::endl;
                prev = hidden_size_[i];
            }
            std::cout<<"after initializing hidden_size_"<<std::endl;
        }

        NeuralNetwork(int input, int output, std::vector<int> hidden_sz) 
        : input_size_(input), output_size_(output), hidden_size_(hidden_sz), biases(batch_size_, output_size_)
        {
            Matrix biases = Matrix::fill_matrix(0.0, batch_size_, output_size_);
            initialize_weights(weights, input_size_, output_size_);

            hidden_layers_ = hidden_size_.size();
        }

        
        
        Matrix forward_pass(Matrix &inputs)
        {
            input_cache = std::vector<Matrix> (); //cleaing the input cache at the start of the forward pass 

            Matrix prev_pred = inputs;

            for(int i = 0; i < hidden_layers_ + 1; i++)
            {
                prev_pred = weights[i]*prev_pred + biases;
                update_using_ReLU(prev_pred);
                input_cache[i] = prev_pred;
            };

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

