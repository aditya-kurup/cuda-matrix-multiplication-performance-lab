#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace matmul {

struct Shape {
  int m;
  int k;
  int n;
};

inline std::vector<float> deterministic_matrix(std::size_t count, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> distribution(-8, 8);
  std::vector<float> values(count);
  for (float& value : values) {
    value = static_cast<float>(distribution(generator)) / 9.0f;
  }
  return values;
}

// Row-major: A[i,k] is A[i*K+k], B[k,j] is B[k*N+j], C[i,j] is C[i*N+j].
inline std::vector<float> cpu_reference(const std::vector<float>& a,
                                        const std::vector<float>& b, Shape shape) {
  if (shape.m <= 0 || shape.k <= 0 || shape.n <= 0 ||
      a.size() != static_cast<std::size_t>(shape.m) * shape.k ||
      b.size() != static_cast<std::size_t>(shape.k) * shape.n) {
    throw std::invalid_argument("invalid matrix dimensions or input sizes");
  }
  std::vector<float> c(static_cast<std::size_t>(shape.m) * shape.n);
  for (int row = 0; row < shape.m; ++row) {
    for (int col = 0; col < shape.n; ++col) {
      double sum = 0.0;
      for (int inner = 0; inner < shape.k; ++inner) {
        sum += static_cast<double>(a[static_cast<std::size_t>(row) * shape.k + inner]) *
               static_cast<double>(b[static_cast<std::size_t>(inner) * shape.n + col]);
      }
      c[static_cast<std::size_t>(row) * shape.n + col] = static_cast<float>(sum);
    }
  }
  return c;
}

struct ErrorStats {
  bool passed = true;
  float max_absolute = 0.0f;
  float max_relative = 0.0f;
};

inline ErrorStats compare(const std::vector<float>& actual,
                          const std::vector<float>& expected,
                          float absolute_tolerance = 1e-3f,
                          float relative_tolerance = 1e-4f) {
  if (actual.size() != expected.size()) {
    throw std::invalid_argument("output sizes differ");
  }
  ErrorStats stats;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float absolute = std::abs(actual[i] - expected[i]);
    const float relative = absolute / std::max(std::abs(expected[i]), 1e-6f);
    stats.max_absolute = std::max(stats.max_absolute, absolute);
    stats.max_relative = std::max(stats.max_relative, relative);
    if (!std::isfinite(actual[i]) ||
        absolute > absolute_tolerance + relative_tolerance * std::abs(expected[i])) {
      stats.passed = false;
    }
  }
  return stats;
}

}  // namespace matmul
