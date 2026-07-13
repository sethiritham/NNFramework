#include "ScopeTimer.hpp"
#include "neural_network.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/**
 *@brief processes the MNIST dataset
 *@param filename Entire filepath with the filename
 *@param X the input matrix for the NN
 *@param Y the actual(ideal) prediction output
 *@param num_samples Batch size
 */
void load_mnist_csv(const std::string &filename, Matrix &X, Matrix &Y,
                    int num_samples) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file\n";
    return;
  }
  std::string line;

  int row = 0;

  std::getline(file, line);

  Y.fill_matrix_double(0.0, Y);
  X.fill_matrix_double(0.0, X);

  while (std::getline(file, line) && row < num_samples) {
    std::stringstream ss(line);
    std::string val;

    std::getline(ss, val, ',');

    LOG("VAL IS " + val);
    int label = std::stoi(val);

    LOG("LABEL: " << label);

    Y[row][label] = 1.0;

    int col = 0;
    while (std::getline(ss, val, ',')) {
      X[row][col] = std::stod(val) / 255.0;
      col++;
    }
    row++;
  }
}

int main() {
  int num_samples = 128;

  NeuralNetwork nn(784, 10, {256, 128}, num_samples, 1e-4);

  Matrix input_matrix(num_samples, 784);
  Matrix actual_prediction_matrix(num_samples, 10);

  actual_prediction_matrix.fill_matrix_double(0, actual_prediction_matrix);
  input_matrix.fill_matrix_double(0, input_matrix);

  load_mnist_csv("/Users/rizzam/Codes/C++/NNFramework/data/mnist_train.csv",
                 input_matrix, actual_prediction_matrix, num_samples);

  actual_prediction_matrix.display_matrix();

  double total_time = 0.0;
  int loop_cycles = 20000;
  for (int k = 0; k < loop_cycles; k++) {
    double loss = 0.0;

    Matrix pred(num_samples, 10);

    ScopeTimer timer("FORWARD PASS");
    pred = nn.forward_pass(input_matrix);

    timer.time_display();

    total_time += timer.get_time_ms();

    loss = nn.cross_entropy_loss(pred, actual_prediction_matrix);

    nn.backward_pass();

    LOG("LOSS IS: " << std::endl << loss);
  }

  LOG("AVERAGE TIME DURATION AFTER CACHE TILING: " << total_time / loop_cycles
                                                   << "ms");

  nn.save_model_int8("/Users/rizzam/Codes/"
                     "C++/NNFramework/model.bin");
}
