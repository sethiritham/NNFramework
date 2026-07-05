#pragma once
#include "matrix.hpp"
#include <cmath>
#include <cstdint>
#include <vector>

struct QBlock32 {
  float scale;
  int8_t weights[32];
};

struct QWeight32 {
  uint16_t rows;
  uint16_t cols;
  std::vector<QBlock32> qblocks;
};

std::vector<QWeight32> quantize_weights(std::vector<Matrix> &weights);
