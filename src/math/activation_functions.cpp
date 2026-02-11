#include "activation_functions.hpp"
#include "matrix.hpp"

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

void sigmoid(double &val)
{
    val = std::pow((1 + std::exp(-val)), -1);
}
