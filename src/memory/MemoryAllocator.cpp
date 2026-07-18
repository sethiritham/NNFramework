#include "MemoryAllocator.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#define MB 1024 * 1024
#define GB 1024 * 1024 * 1024

MemoryAllocator::MemoryAllocator() {
  memoryPointer =
      reinterpret_cast<char *>(std::aligned_alloc(128, 1024 * 1024 * 1024));

  for (auto &free_list : free_lists) {
    free_list = nullptr;
  }

  for (int i = 0; i < 512; i++) {
    sections[511 - i] = (i * 2 * MB + memoryPointer);
    ptrs_per_section[i] = 0;
  }
}

size_t bit_ceil(size_t n) {
  if (n <= 1)
    return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

int countr_zero(uint32_t x) {
  if (x == 0)
    return 32;
  int count = 0;
  if ((x & 0x0000FFFF) == 0) {
    count += 16;
    x >>= 16;
  }
  if ((x & 0x000000FF) == 0) {
    count += 8;
    x >>= 8;
  }
  if ((x & 0x0000000F) == 0) {
    count += 4;
    x >>= 4;
  }
  if ((x & 0x00000003) == 0) {
    count += 2;
    x >>= 2;
  }
  if ((x & 0x00000001) == 0) {
    count += 1;
  }
  return count;
}

uint8_t get_index(size_t size) {
  size_t closet_pow2 = bit_ceil(size);
  uint8_t index = countr_zero(closet_pow2) - 7;
  return index;
}

char *MemoryAllocator::allocate(std::size_t size) {

  size = (size < 128) ? 128 : size;

  if (size > 2 * MB) {
    return reinterpret_cast<char *>(std::aligned_alloc(128, size));
  }

  uint8_t index = get_index(size);

  if (free_lists[index] == nullptr) {

    if (top_index < 0) {
      return reinterpret_cast<char *>(std::aligned_alloc(128, size));
    }

    char *ptr = sections[top_index];
    top_index--;

    if (index == 14)
      return ptr;

    if (!ptr)
      return reinterpret_cast<char *>(std::aligned_alloc(128, size));

    int stride = 128 << index;

    Node *current = (Node *)(ptr + stride);
    free_lists[index] = (Node *)(ptr + stride);

    for (int i = 2 * stride; i < 2 * MB; i += stride) {
      current->next = (Node *)((char *)current + stride);
      current = (Node *)((char *)current + stride);
    }

    current->next = nullptr;

    uint8_t blk_index = (ptr - memoryPointer) / (2 * MB);

    ptrs_per_section[blk_index]++;

    return ptr;
  } else {
    char *ptr = (char *)(free_lists[index]);
    free_lists[index] = free_lists[index]->next;

    uint8_t blk_index = (ptr - memoryPointer) / (2 * MB);
    ptrs_per_section[blk_index]++;

    return ptr;
  }
}

void MemoryAllocator::deallocate(char *ptr, size_t size) {

  size = (size < 128) ? 128 : size;

  if (size > 2 * MB) {
    std::free((void *)ptr);
  }

  uint8_t index = get_index(size);

  Node *freed_block = (Node *)ptr;

  freed_block->next = free_lists[index];

  free_lists[index] = freed_block;

  uint8_t blk_index = (ptr - memoryPointer) / (2 * MB);

  ptrs_per_section[blk_index]--;
}
