#include "MemoryAllocator.hpp"
#include <cstdlib>

MemoryAllocator::MemoryAllocator() {
  memoryPointer =
      reinterpret_cast<char *>(std::aligned_alloc(128, 1024 * 1024 * 1024));
}
