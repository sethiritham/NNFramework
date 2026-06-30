#pragma once
#include "matrix.hpp"
#include <cmath>

void update_using_ReLU(Matrix &m);

void update_using_softmax(Matrix &m);

void update_using_sigmoid(Matrix &m);

void update_using_LogSoftmax(Matrix &m);
