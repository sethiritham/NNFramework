#include <array>
#include <cstddef>

struct Node {
  Node *next;
};

struct SuperBlock {
  Node *chunk;
};

class MemoryAllocator {
private:
  int top_index = 511;
  static const int NUM_BUCKETS = 15;
  char *memoryPointer;
  std::array<Node *, NUM_BUCKETS> free_lists;
  std::array<char *, 512> sections;
  std::array<int, 512> ptrs_per_section;

public:
  MemoryAllocator();

  char *allocate(std::size_t size);

  void deallocate(char *ptr, size_t size);

  ~MemoryAllocator();
};
