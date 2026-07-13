#include "quantize8.hpp"
#include <cmath>
#include <cstdint>
#include <vector>

std::vector<QWeight32> quantize_weights(std::vector<Matrix> &weights) {
  std::vector<QWeight32> q_weights;
  q_weights.reserve(weights.size());

  for (size_t i = 0; i < weights.size(); i++) {
    QWeight32 wt_blk;
    wt_blk.cols = weights[i].num_cols;
    wt_blk.rows = weights[i].num_rows;
    std::vector<QBlock32> blks;

    std::vector<double> w_data = weights[i].get_data();

    int num_full_blks = weights[i].num_elements() / 32;
    int num_left = weights[i].num_elements() - 32 * num_full_blks;

    for (int j = 0; j < num_full_blks * 32; j += 32) {
      QBlock32 blk;
      double max = 0.0;
      for (int k = 0; k < 32; k++) {
        if (max < std::abs(w_data[j + k])) {
          max = std::abs(w_data[j + k]);
        }
      }

      max = (max == 0.0) ? 1.0 : max;
      float scale = (float)max / (float)127;

      blk.scale = scale;

      for (int k = 0; k < 32; k++) {
        blk.weights[k] = std::round(w_data[j + k] / scale);
      }

      blks.push_back(blk);
    }

    if (num_left == 0) {
      wt_blk.qblocks = blks;
      q_weights.push_back(wt_blk);
      continue;
    }

    QBlock32 blk;
    double max = 0.0;

    for (int j = num_full_blks * 32; j < num_full_blks * 32 + num_left; j++) {
      if (max < std::abs(w_data[j])) {
        max = std::abs(w_data[j]);
      }
    }

    max = (max == 0.0) ? 1.0 : max;
    float scale = (float)max / (float)127;

    for (int j = num_full_blks * 32; j < num_full_blks * 32 + num_left; j++) {
      blk.weights[j - num_full_blks * 32] = std::round(w_data[j] / scale);
    }

    blk.scale = scale;

    blks.push_back(blk);

    wt_blk.qblocks = blks;
    q_weights.push_back(wt_blk);
  }

  return q_weights;
}

Matrix dequantize_weight(QWeight32 q_weight) {
  Matrix wt(q_weight.rows, q_weight.cols);

  int num_elements = wt.num_rows * wt.num_cols;

  int num_full_blks = num_elements / 32;
  int num_left = num_elements - 32 * num_full_blks;

  std::vector<double> elements;
  elements.reserve(num_elements);

  for (int i = 0; i < num_full_blks; i++) {
    QBlock32 blk = q_weight.qblocks[i];
    float scale = blk.scale;
    for (int j = 0; j < 32; j++) {
      int8_t val = blk.weights[j];
      double element = val * scale;
      elements.push_back(element);
    }
  }

  {
    QBlock32 blk = q_weight.qblocks[num_full_blks];
    float scale = blk.scale;

    for (int i = 0; i < num_left; i++) {
      int8_t val = blk.weights[i];
      double element = val * scale;
      elements.push_back(element);
    }
  }

  for (uint16_t i = 0; i < wt.num_rows; i++) {
    for (uint16_t j = 0; j < wt.num_cols; j++) {
      double val = elements[i * wt.num_cols + j];
      wt[i][j] = val;
    }
  }

  return wt;
}

Matrix multiply_quantized(QWeight32 &weights, Matrix &inputs) {
  Matrix output(weights.rows, inputs.num_cols);

  for (uint32_t i = 0; i < inputs.num_rows; i++) {
    for (uint32_t k = 0; k < inputs.num_cols; k++) {
      if (inputs[i][k] == 0.0)
        continue;
      double input_val = inputs[i][k];

      for (uint32_t j = 0; j < weights.cols; j++) {
        int block_index = (k * weights.cols + j) / 32;
        int val_index = (k * weights.cols + j) % 32;

        int qwt = weights.qblocks[block_index].weights[val_index];
        float scale = weights.qblocks[block_index].scale;

        output[i][j] = (double)qwt * (double)scale * input_val;
      }
    }
  }

  return output;
}
