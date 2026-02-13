#include "neural_network.hpp"


//PUBLIC FUNCTIONS
NeuralNetwork::NeuralNetwork(std::size_t input, std::size_t output, std::size_t hiddenL, std::size_t batchS, double lr) 
:   in_features_(input), out_features_(output), num_hidden_layers_(hiddenL), batch_size_(batchS),
    layer_sizes_(num_hidden_layers_ + 2), learning_rate(lr)
    {
        std::cout<<"Started initializer"<<std::endl;

        weights_.reserve(num_hidden_layers_ + 1); 
        biases_.reserve(num_hidden_layers_ + 1);
        input_cache_.reserve(num_hidden_layers_ + 1);

        int x  = (int)((in_features_ - out_features_)/(num_hidden_layers_ + 1));

        for (std::size_t i = 0; i < num_hidden_layers_ + 2;  i++)
        {
            layer_sizes_[i] = input - x*i;
            input_cache_.emplace_back(batch_size_, layer_sizes_[i]);
        }

        layer_sizes_[num_hidden_layers_ + 1] = out_features_;

        for(std::size_t j = 0; j < num_hidden_layers_ + 1; j++)
        {
            weights_.emplace_back(layer_sizes_[j], layer_sizes_[j + 1]);
            biases_.emplace_back(1, layer_sizes_[j + 1]);
        }

        for(std::size_t i = 0; i < biases_.size(); i++)
        {
            Matrix::fill_matrix_double(0.0, biases_[i]);
        }

        update_init_weights_ReLU(weights_);
    }

NeuralNetwork::NeuralNetwork(std::size_t input, std::size_t output, std::vector<int> hidden_sz, std::size_t batchS, double lr) 
:   in_features_(input), out_features_(output), num_hidden_layers_(hidden_sz.size()), batch_size_(batchS), 
    layer_sizes_(hidden_sz.size() + 2), learning_rate(lr)
    {
        std::cout<<"Started initializer"<<std::endl;

        weights_.reserve(num_hidden_layers_ + 1);
        biases_.reserve(num_hidden_layers_ + 1);
        input_cache_.reserve(num_hidden_layers_ + 1);

        int x  = (int)((input - output)/(num_hidden_layers_ + 1));

        int prev = input;

        for (std::size_t i = 0; i < num_hidden_layers_;  i++)
        {
            layer_sizes_[i] = prev - x;
            prev = layer_sizes_[i];
        }

        for(std::size_t i = 0; i < num_hidden_layers_ + 1; i++)
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

        for(std::size_t i = 0; i < num_hidden_layers_ + 1; i++)
        {
            prev_pred = Matrix::row_wise_sum(prev_pred*weights_[i], biases_[i]);
            update_using_ReLU(prev_pred);
            input_cache_.emplace_back(prev_pred);
        }

        prev_pred.display_matrix();

        return prev_pred;
    }

Matrix NeuralNetwork::loss_fn(Matrix& pred, Matrix& actual)
    {
        actual_prediction = actual;

        Matrix loss_matrix(pred.num_rows, pred.num_cols);

        loss_matrix = pred - actual;
        
        loss_matrix = map_matrix<double>(loss_matrix, 
            [](double element)
            {
                return ((std::pow(element, 2.0))/2.0); 
            });
        
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
        for(std::uint32_t row = 0; row < weights[i].num_rows; row++)
        {
            for(std::uint32_t col = 0; col < weights[i].num_cols; col++)
            {
                weights[i][row][col] = dist(gen) * std::sqrt(2.0 / in_features_);
            }
        }
    }
}

Matrix NeuralNetwork::calculate_and_filter_gradient(Matrix& grad_output, Matrix& pred)
{
    Matrix filtered_gradient = grad_output;
    
    for(std::uint32_t row = 0; row < pred.num_rows; row++)
    {
        for(std::uint32_t col = 0; col < pred.num_cols; col++)
        {
            if(pred[row][col] == 0.0)
            {
                filtered_gradient[row][col] = 0.0;
            }
        }
    } 

    return filtered_gradient;
}

void NeuralNetwork::backward_pass()
{
    Matrix filtered_gradient = calculate_and_filter_gradient(actual_prediction, input_cache_[1 + num_hidden_layers_]);


    for(std::size_t i = num_hidden_layers_ + 1; i > 0; i--)
    {

        Matrix d_weight(layer_sizes_[i - 1], layer_sizes_[i]); //dW
        Matrix d_bias(1, layer_sizes_[i]); // db
        Matrix grad_output(batch_size_, layer_sizes_[i - 1]); // input for next layer

        grad_output = input_cache_[i - 1]; 

        d_weight = ~grad_output*filtered_gradient;

        grad_output = filtered_gradient*(~weights_[i - 1]);

        weights_[i - 1] = weights_[i - 1] - d_weight*learning_rate;

        biases_[i - 1] = biases_[i - 1] - d_bias * learning_rate;

        filtered_gradient = calculate_and_filter_gradient(grad_output, input_cache_[i - 1]);
    }
}



    
