#include "architecture/neural_network.hpp"
#include <random>


int main()
{
    std::cout<<"STARTED"<<std::endl;

    NeuralNetwork nn(1, 1, 3, 7);

    std::cout<<"CREATED NEURAL NETWORK"<<std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-7, 7);

    Matrix input(7, 1);

    for(int i = 0; i < 7; i++)
    {
        input[i][1] = dist(gen);
    }

    Matrix pred = nn.forward_pass(input);

    pred.display_matrix();

    std::cout<<"PRINTED MATRIX! "<<input[0][0]<<std::endl;

}