#pragma once
#include "matrix.hpp"
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

Matrix dequantize_weight(QWeight32 q_weight);

Matrix multiply_quantized(QWeight32 &weights, Matrix &inputs);

void muliply_quantized_chunked(QWeight32 &weights, Matrix &inputs,
                               Matrix &output, uint32_t start_row,
                               uint32_t end_row);

Matrix multiply_quantized_multithreaded(QWeight32 &weights, Matrix &inputs);
