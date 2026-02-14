#include "activation_functions.hpp"


void update_using_ReLU(Matrix &m)
{
    for(std::uint32_t row = 0; row < m.num_rows; row++)
    {
        for(std::uint32_t col = 0; col < m.num_cols; col++)
        {
            m[row][col] = (m[row][col] < 0) ? 0 : m[row][col];
        }
    }
}

void update_using_sigmoid(Matrix& m)
{
    for(std::uint32_t row = 0; row < m.num_rows; row++)
    {
        for(std::uint32_t col = 0; col < m.num_cols; col++)
        {
            double& val = m[row][col];
            m[row][col] = std::pow((1 + std::exp(-val)), -1);
        }
    }
}

void update_using_softmax(Matrix& m)
{
    double sum_of_all = 0.0;
    for(std::uint32_t row = 0; row < m.num_rows; row++)
    {
        for(std::uint32_t col = 0; col < m.num_cols; col++)
        {
            sum_of_all += std::exp(m[row][col]);
        }
    }

    for(std::uint32_t row = 0; row < m.num_rows; row++)
    {
        for(std::uint32_t col = 0; col < m.num_cols; col++)
        {
            m[row][col] = std::exp(m[row][col])/sum_of_all;
        }
    }
}


void update_using_LogSoftmax(Matrix& m)
{       
    for(std::uint32_t row = 0; row < m.num_rows; row++)
    {
        double max_val = -1.0 * std::numeric_limits<double>::infinity();
        double sum = 0.0;
        
        for(std::uint32_t col = 0; col < m.num_cols; col++) 
        {
            if(m[row][col] > max_val) max_val = m[row][col];
        }

        for(std::uint32_t col = 0; col < m.num_cols; col++)
        {
            sum += std::exp(m[row][col] - max_val);
        }

        for(std::uint32_t col = 0; col < m.num_cols; col++)
        {
            m[row][col] = m[row][col] - (std::log(sum) + max_val);
        }

    }
}
