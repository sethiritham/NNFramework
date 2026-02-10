#include <iostream>
#include <vector>
#include <cmath>
#include "../math/matrix.hpp"
#include "../math/activation_functions.hpp"
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

        NeuralNetwork(int input, int output, std::vector<int> hidden_sz) : m_input_size(input), m_output_size(output), m_hidden_size(hidden_sz) 
        {
            m_hidden_layers = m_hidden_size.size();
        }

        
        double forward_pass(Matrix &x, Matrix &w, Matrix &b)
        {
            static std::vector<Matrix> weights(1 + m_hidden_layers);
            Matrix output(m_output_size, 1);
            Matrix prev_pred = x;

            for(int i = 0; i < m_hidden_layers; i++)
            {
                prev_pred = w*prev_pred + b;
                ReLU(prev_pred);
            }
        }
};






