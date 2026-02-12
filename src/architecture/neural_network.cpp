#include "neural_network.hpp"


//PUBLIC FUNCTIONS
NeuralNetwork::NeuralNetwork(int input, int output, int hiddenL, int batchS) 
:   in_features_(input), out_features_(output), batch_size_(batchS), num_hidden_layers_(hiddenL), 
    hidden_layer_sizes_(num_hidden_layers_)
    {
        std::cout<<"Started initializer"<<std::endl;

        weights_.reserve(num_hidden_layers_ + 1); 
        biases_.reserve(num_hidden_layers_ + 1);
        input_cache_.reserve(num_hidden_layers_ + 1);

        int x  = (int)((input - output)/(hiddenL + 1));

        int prev = input;

        for (int i = 0; i < hiddenL;  i++)
        {
            hidden_layer_sizes_[i] = prev - x;
            prev = hidden_layer_sizes_[i];
        }

        for(int i = 0; i < num_hidden_layers_ + 1; i++)
        {
            weights_.emplace_back(input, output);
            input_cache_.emplace_back(batch_size_, input);
            biases_.emplace_back(batch_size_, out_features_);
        }

        for(int i = 0; i < biases_.size(); i++)
        {
            Matrix::fill_matrix_double(0.0, biases_[i]);
        }

        update_init_weights_ReLU(weights_);
    }

NeuralNetwork::NeuralNetwork(int input, int output, std::vector<int> hidden_sz, int batchS) 
:   in_features_(input), out_features_(output), batch_size_(batchS), num_hidden_layers_(hidden_sz.size()), 
    hidden_layer_sizes_(hidden_sz.size())
    {
        std::cout<<"Started initializer"<<std::endl;

        weights_.reserve(num_hidden_layers_ + 1);
        biases_.reserve(num_hidden_layers_ + 1);
        input_cache_.reserve(num_hidden_layers_ + 1);

        int x  = (int)((input - output)/(num_hidden_layers_ + 1));

        int prev = input;

        for (int i = 0; i < num_hidden_layers_;  i++)
        {
            hidden_layer_sizes_[i] = prev - x;
            prev = hidden_layer_sizes_[i];
        }

        for(int i = 0; i < num_hidden_layers_ + 1; i++)
        {
            weights_.emplace_back(input, output);
            input_cache_.emplace_back(batch_size_, input);
            biases_.emplace_back(batch_size_, out_features_);
        }

        update_init_weights_ReLU(weights_);
    }

Matrix NeuralNetwork::forward_pass(Matrix &inputs)
    {
        LOG("STARTED FORWARD PASS");
        input_cache_.clear(); //cleaing the input cache at the start of the forward pass 

        Matrix prev_pred = inputs;

        for(int i = 0; i < num_hidden_layers_ + 1; i++)
        {
            weights_[i].display_matrix();

            (prev_pred*weights_[i]).display_matrix();
            prev_pred = prev_pred*weights_[i] + biases_[i];
            update_using_ReLU(prev_pred);
            input_cache_.emplace_back(prev_pred);
        }

        LOG("COMPLETED FORWARD PASS");
        return prev_pred;
    }

Matrix NeuralNetwork::loss_fn(Matrix& pred, Matrix& actual)
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


//PRIVATE FUNCTIONS
void NeuralNetwork::update_init_weights_ReLU(std::vector<Matrix>& weights)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(in_features_, out_features_);

    for(int i = 0; i < weights.size(); i++)
    {
        for(int row = 0; row < weights[i].num_rows; row++)
        {
            for(int col = 0; col < weights[i].num_cols; col++)
            {
                weights[i][row][col] = dist(gen) * std::sqrt(2.0 / in_features_);
            }
        }
    }
}

Matrix NeuralNetwork::calculate_gradient(Matrix pred, Matrix actual)
{
    return pred - actual; // differntial of loss function ((y' - y)^2)/2
}

void NeuralNetwork::update_gradient_using_filter(Matrix& gradient, Matrix &output)
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
    
