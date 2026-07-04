#include "neural_network.hpp"
#include <fstream>
#include <sstream>

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
  int num_samples = 20;
  Matrix input_matrix(num_samples, 784);
  Matrix true_matrix(num_samples, 10);

  load_mnist_csv("/mnt/5b2b3cbe-5bcb-4a72-a855-bc9cc94ccf4f/RITHAM/CODES/C++/"
                 "NNFramework/data/mnist_test.csv",
                 input_matrix, true_matrix, 20);

  NeuralNetwork nn(784, 10, {256, 128}, num_samples, 1e-4);

  nn.load_model_bin("/mnt/5b2b3cbe-5bcb-4a72-a855-bc9cc94ccf4f/RITHAM/CODES/"
                    "C++/NNFramework/model.bin");

  double acc = 0.0;

  Matrix output_pred(num_samples, 10);

  output_pred.fill_matrix_double(0.0, output_pred);

  output_pred = nn.forward_pass(input_matrix);

  for (int sample = 0; sample < num_samples; sample++) {

    double max_prob = 0.0;

    int pred_num = 0;

    int actual_num = 0;

    for (int i = 0; i < 10; i++) {
      if (true_matrix[sample][i] == 1.0) {
        actual_num = i;
      }
    }

    for (int i = 0; i < 10; i++) {
      if (max_prob < output_pred[sample][i]) {
        max_prob = output_pred[sample][i];
        pred_num = i;
      }
    }

    if (pred_num == actual_num)
      acc += 1.0;
  }

  acc = acc / 20.0;

  LOG("ACCURACY: " << acc);
}
