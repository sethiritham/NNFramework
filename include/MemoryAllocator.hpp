#include <array>
#include <cstddef>
#include <cstdint>

struct Node {
  Node *next;
};

struct Slab {
  int next_block = -1;
  uint16_t num_allocs = 0;
  Node *free_list_head = nullptr;
};

class MemoryAllocator {
private:
  int top_index = 511;
  static const int NUM_BUCKETS = 15;
  char *memoryPointer;
  std::array<int, NUM_BUCKETS> head_slab_indices;
  std::array<Slab, 512> slabs;
  std::array<int16_t, 512> free_slab_indices;

public:
  MemoryAllocator();

  char *allocate(std::size_t size);

  void deallocate(char *ptr, size_t size);

  ~MemoryAllocator();
};
