#include "MemoryAllocator.hpp"
#include <cstddef>

extern MemoryAllocator allocator;

class TensorStorage {
private:
  float *data_ptr;
  std::size_t size;

public:
  TensorStorage(size_t size);

  TensorStorage(const TensorStorage &other);

  TensorStorage &operator=(const TensorStorage &other);

  TensorStorage(TensorStorage &&other) noexcept;

  TensorStorage &operator=(TensorStorage &&other);

  ~TensorStorage();
};
