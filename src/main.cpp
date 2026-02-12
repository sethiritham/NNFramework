#include "architecture/neural_network.hpp"
#include <random>

int main()
{
    std::cout<<"STARTED"<<std::endl;

    Matrix accurate_pred(7, 1);

    std::vector<double> accurate_pred_array;

    LOG(accurate_pred_array.size());
    for(int i = 0; i < 7; i++)
    {
        accurate_pred_array.push_back(10.0);
    }

    LOG(accurate_pred_array.size());

    NeuralNetwork nn(1, 1, 3, 7);

    std::cout<<"CREATED NEURAL NETWORK"<<std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-7, 7);

    Matrix input(7, 1);

    for(int i = 0; i < 7; i++)
    {
        input[i][0] = dist(gen);
    }

    LOG("INPUT MATRIX\n");
    input.display_matrix();

    Matrix pred = nn.forward_pass(input);

    pred.display_matrix();

    
    Matrix::fill_matrix_array(accurate_pred_array, accurate_pred);

    Matrix loss_matrix = nn.loss_fn(pred, accurate_pred);

    loss_matrix.display_matrix();

}