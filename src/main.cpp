#include "architecture/neural_network.hpp"
#include <random>
#define LOG(x) std::cout << x << std::endl

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
        input[i][0] = dist(gen);
    }

    LOG("INPUT MATRIX\n");
    input.display_matrix();

    Matrix pred = nn.forward_pass(input);

    LOG("IF THIS IS VISIBLE FORWARD PASS IS NOT THE PROB!");

    pred.display_matrix();

    std::cout<<"PRINTED MATRIX! "<<input[0][0]<<std::endl;

}