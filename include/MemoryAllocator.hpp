#include <array>
#include <cstddef>

struct Node {
  Node *next;
};

class MemoryAllocator {
private:
  int top_index = 511;
  static const int NUM_BUCKETS = 15;
  char *memoryPointer;
  std::array<Node *, NUM_BUCKETS> free_lists;
  std::array<char *, 512> sections;

public:
  MemoryAllocator();

  char *allocate(std::size_t size);

  ~MemoryAllocator();
};
