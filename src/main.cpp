#include "architecture/neural_network.hpp"
#include <random>

int main()
{
    std::cout<<"STARTED"<<std::endl;

    Matrix accurate_pred(7, 1);

    accurate_pred.fill_matrix_double(0.0, accurate_pred);
    accurate_pred[1][0] = 1.0;

    NeuralNetwork nn(7, 1, 3, 7, 0.01);

    std::cout<<"CREATED NEURAL NETWORK"<<std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-7, 7);

    Matrix input(7, 7);

    for(int i = 0; i < 7; i++)
    {
        for(int j = 0; j < 7; j++)
        {
            input[i][j] = dist(gen);
        }
    }

    LOG("INPUT MATRIX\n");
    input.display_matrix();

    Matrix pred = nn.forward_pass(input);

    LOG("PREDICTION MATRIX\n");
    pred.display_matrix();

    LOG("ACTUAL MATRIX\n");
    accurate_pred.display_matrix();

    Matrix loss_matrix = nn.loss_fn(pred, accurate_pred);

    LOG("LOSS MATRIX");
    loss_matrix.display_matrix();


    LOG("ONE OF THE WEIGHTS BEFORE BACK PROP");
    nn.weights_[2].display_matrix();

    nn.backward_pass();

    LOG("BACKWARD PASS COMPLETE ONE OF THE WEIGHT AFTER");
    nn.weights_[2].display_matrix();

    LOG("SECOND FORWARD PASS NOW");

    pred = nn.forward_pass(input);

    LOG("PREDICTION MATRIX\n");
    pred.display_matrix();

    loss_matrix = nn.loss_fn(pred, accurate_pred);

    LOG("LOSS NOW\n");
    loss_matrix.display_matrix();


    LOG("ONE OF THE WEIGHTS BEFORE BACK PROP");
    nn.weights_[2].display_matrix();

    nn.backward_pass();

    LOG("BACKWARD PASS COMPLETE ONE OF THE WEIGHT AFTER");
    nn.weights_[2].display_matrix();

}