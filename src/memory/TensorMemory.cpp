#include "TensorMemory.hpp"
#include <cstring>

TensorStorage::TensorStorage(size_t size) {
  char *ptr = allocator.allocate(size);
  data_ptr = (float *)ptr;
  this->size = size;
}

TensorStorage::TensorStorage(const TensorStorage &other) {

  char *ptr = allocator.allocate(other.size);
  data_ptr = (float *)ptr;

  std::memcpy(ptr, other.data_ptr, other.size);
  this->size = other.size;
}

TensorStorage &TensorStorage::operator=(const TensorStorage &other) {

  if (this == &other)
    return *this;

  allocator.deallocate(data_ptr, size);
  char *ptr = allocator.allocate(other.size);

  data_ptr = (float *)ptr;
  std::memcpy(ptr, other.data_ptr, other.size);

  this->size = other.size;

  return *this;
}

TensorStorage::TensorStorage(TensorStorage &&other) noexcept {
  this->data_ptr = other.data_ptr;
  this->size = other.size;

  other.data_ptr = nullptr;
  other.size = 0;
}

TensorStorage &TensorStorage::operator=(TensorStorage &&other) {
  if (this == &other)
    return *this;

  allocator.deallocate(this->data_ptr, this->size);

  this->data_ptr = other.data_ptr;
  this->size = other.size;

  other.data_ptr = nullptr;
  other.size = 0;

  return *this;
}

TensorStorage::~TensorStorage() {
  if (this->data_ptr != nullptr)
    allocator.deallocate(this->data_ptr, this->size);
}
