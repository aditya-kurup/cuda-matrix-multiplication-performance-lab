#include "reference.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  using namespace matmul;
  const Shape shape{2, 3, 2};
  const std::vector<float> a{1, 2, 3, 4, 5, 6};
  const std::vector<float> b{7, 8, 9, 10, 11, 12};
  const std::vector<float> expected{58, 64, 139, 154};
  const auto actual = cpu_reference(a, b, shape);
  assert(actual == expected);
  assert(compare(actual, expected).passed);
  auto wrong = actual;
  wrong[0] += 0.01f;
  assert(!compare(wrong, expected).passed);
  wrong[0] = std::nanf("");
  assert(!compare(wrong, expected).passed);
  assert(deterministic_matrix(20, 42) == deterministic_matrix(20, 42));
  std::cout << "CPU reference checks passed\n";
}
