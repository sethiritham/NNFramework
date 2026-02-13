#include "activation_functions.hpp"


void update_using_ReLU(Matrix &m)
{
    for(int row = 0; row < m.num_rows; row++)
    {
        for(int col = 0; col < m.num_cols; col++)
        {
            m[row][col] = (m[row][col] < 0) ? 0 : m[row][col];
        }
    }
}

void update_using_softmax(Matrix &m)
{

}

void update_using_sigmoid(Matrix &m)
{
    for(int row = 0; row < m.num_rows; row++)
    {
        for(int col = 0; col < m.num_cols; col++)
        {
            double& val = m[row][col];
            m[row][col] = std::pow((1 + std::exp(-val)), -1);
        }
    }
}
