#include "neural_network.hpp"


//PUBLIC FUNCTIONS
NeuralNetwork::NeuralNetwork(int input, int output, int hiddenL, int batchS) 
:   in_features_(input), out_features_(output), num_hidden_layers_(hiddenL), batch_size_(batchS),
    layer_sizes_(num_hidden_layers_ + 2)
    {
        std::cout<<"Started initializer"<<std::endl;

        weights_.reserve(num_hidden_layers_ + 1); 
        biases_.reserve(num_hidden_layers_ + 1);
        input_cache_.reserve(num_hidden_layers_ + 1);

        int x  = (int)((input - output)/(num_hidden_layers_ + 1));

        for (int i = 0; i < num_hidden_layers_ + 2;  i++)
        {
            layer_sizes_[i] = input - x*i;
            input_cache_.emplace_back(batch_size_, layer_sizes_[i]);
        }

        for(int j = 0; j < num_hidden_layers_ + 1; j++)
        {
            weights_.emplace_back(layer_sizes_[j], layer_sizes_[j + 1]);
            biases_.emplace_back(1, layer_sizes_[j + 1]);
        }

        LOG("WEIGHTS AND BIASES CHECK\nBIASES");

        LOG(biases_[1].num_elements());

        for(std::size_t i = 0; i < biases_.size(); i++)
        {
            Matrix::fill_matrix_double(0.0, biases_[i]);
        }

        update_init_weights_ReLU(weights_);
    }

NeuralNetwork::NeuralNetwork(int input, int output, std::vector<int> hidden_sz, int batchS) 
:   in_features_(input), out_features_(output), num_hidden_layers_(hidden_sz.size()), batch_size_(batchS), 
    layer_sizes_(hidden_sz.size() + 2)
    {
        std::cout<<"Started initializer"<<std::endl;

        weights_.reserve(num_hidden_layers_ + 1);
        biases_.reserve(num_hidden_layers_ + 1);
        input_cache_.reserve(num_hidden_layers_ + 1);

        int x  = (int)((input - output)/(num_hidden_layers_ + 1));

        int prev = input;

        for (int i = 0; i < num_hidden_layers_;  i++)
        {
            layer_sizes_[i] = prev - x;
            prev = layer_sizes_[i];
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

        input_cache_.emplace_back(inputs);

        Matrix prev_pred = input_cache_[0];

        for(int i = 0; i < num_hidden_layers_ + 1; i++)
        {
            prev_pred = Matrix::row_wise_sum(prev_pred*weights_[i], biases_[i]);
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

    for(std::size_t i = 0; i < weights.size(); i++)
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

Matrix NeuralNetwork::calculate_and_filter_gradient(Matrix& loss)
{
    Matrix pred = forward_pass(input_cache_[1 + num_hidden_layers_]);

    Matrix filtered_gradient = map_matrix<double>(filtered_gradient, [](double element) { 
                return std::pow(element, 0.5) * 2.0;    
    });
    
    for(int row = 0; row < filtered_gradient.num_rows; row++)
    {
        for(int col = 0; col < filtered_gradient.num_cols; col++)
        {
            if(pred[row][col] == 0.0)
            {
                filtered_gradient[row][col] = 0.0;
            }
        }
    } 

    return filtered_gradient;
}

void NeuralNetwork::backward_pass(Matrix& loss, const double& learning_rate)
{
    Matrix filtered_gradient = calculate_and_filter_gradient(loss);


    for(int i = num_hidden_layers_ + 2; i > 0; i--)
    {

        Matrix d_weight(layer_sizes_[i - 2], layer_sizes_[i - 1]); //dW
        Matrix d_bias(batch_size_, layer_sizes_[i - 1]); // db
        Matrix grad_output(batch_size_, layer_sizes_[i - 2]); // input for next layer

        grad_output = input_cache_[i - 1]; 

        d_weight = ~grad_output*filtered_gradient;
        d_bias = d_bias.collapse_horizontal(d_bias);

        grad_output = filtered_gradient*(~weights_[i - 1]);

        weights_[i - 1] = weights_[i - 1] - d_weight*learning_rate;

        biases_[i - 1] = biases_[i - 1] - d_bias * learning_rate ;

    }
}



    
