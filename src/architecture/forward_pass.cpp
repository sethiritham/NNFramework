#include <iostream>
#include <vector>
#include <cmath>
#include "../math/matrix.hpp"
#include <memory>

class NeuralNetwork
{
    private:
        int m_input_size;
        int m_output_size;
        int m_hidden_layers;
        std::vector<int> m_hidden_size;

    public:
        NeuralNetwork(int input, int output, int hiddenL) : m_input_size(input), m_output_size(output), m_hidden_layers(hiddenL) 
        {
            int x  = (int)((input - output)/(hiddenL + 1));

            int prev = input;
            for (int i = 0; i < hiddenL;  i++)
            {
                m_hidden_size[i] = prev - x;
                prev = m_hidden_size[i];
            }
        }
};





