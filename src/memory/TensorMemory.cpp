#include "TensorMemory.hpp"

TensorStorage::TensorStorage(size_t size) {
  char *ptr = allocator.allocate(size);
  data_ptr = (float *)ptr;
}
