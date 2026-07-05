#include "quantize8.hpp"

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
