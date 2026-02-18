#include "architecture/neural_network.hpp"
#include <random>

int main()
{
    NeuralNetwork nn(5, 1, 2, 3, 0.01);

    Matrix input_matrix(3, 5);


    for(int k = 0; k < 4; k++)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for(int i = 0; i < input_matrix.num_rows; i ++)
        {
            for(int j = 0; j < input_matrix.num_cols; j++)
            {
                input_matrix[i][j] = dis(gen);
            }
        }

        Matrix actual_prediction_matrix(3, 1);

        actual_prediction_matrix.fill_matrix_double(0.0, actual_prediction_matrix);
        actual_prediction_matrix[2][0] = 1.0;

        Matrix pred(3, 1);

        pred = nn.forward_pass(input_matrix);
        double loss = 0.0;

        loss = nn.loss_fn(pred, actual_prediction_matrix);

        nn.backward_pass();

        LOG("LOSS IS: "<<std::endl<<loss);
    }
    
}