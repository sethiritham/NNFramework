#include "TensorMemory.hpp"
#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>

struct SmallVector {
  int dimensions;
  std::array<size_t, 5> shape;
  std::array<size_t, 5> stride;
};

class Tensor {
private:
  std::shared_ptr<TensorStorage> storage;
  SmallVector shape_stride;
  size_t offset_storage;

public:
  Tensor(std::initializer_list<size_t> shape);

  Tensor(const Tensor &other);

  Tensor &operator=(const Tensor &other);

  Tensor(const Tensor &&other) noexcept;

  Tensor &operator=(const Tensor &&other);

  ~Tensor();

public:
  Tensor &operator+(const Tensor &other);

  Tensor &operator*(const Tensor &other);

  const double &operator[](int &row) const;

  double &operator[](int &row);

  Tensor &operator~();

  Tensor &operator*(const double &scalar);

  bool &operator==(const Tensor &other);

  Tensor &operator-(const Tensor &other);

public:
  Tensor hadamaard(const Tensor &other);
};
