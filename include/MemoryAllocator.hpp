#include <cstddef>

struct Node {
  Node *next;
};

class MemoryAllocator {
private:
  char *memoryPointer;

public:
  MemoryAllocator();

  char *allocate(std::size_t size);

  ~MemoryAllocator();
};
