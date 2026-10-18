#include "reference.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kTile = 16;
constexpr int kWarmups = 3;
constexpr int kDefaultTrials = 9;

void check_cuda(cudaError_t result, const char* expression, const char* file, int line) {
  if (result != cudaSuccess) {
    std::ostringstream message;
    message << file << ':' << line << " " << expression << ": " << cudaGetErrorString(result);
    throw std::runtime_error(message.str());
  }
}
#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)

__global__ void naive_kernel(const float* a, const float* b, float* c, int m, int k, int n) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;
  float sum = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    sum += a[row * k + inner] * b[inner * n + col];
  }
  c[row * n + col] = sum;
}

__global__ void tiled_kernel(const float* a, const float* b, float* c, int m, int k, int n) {
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  float sum = 0.0f;

  for (int base = 0; base < k; base += kTile) {
    const int a_col = base + threadIdx.x;
    const int b_row = base + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();
    for (int inner = 0; inner < kTile; ++inner) {
      sum += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = sum;
}

struct DeviceBuffer {
  float* data = nullptr;
  explicit DeviceBuffer(std::size_t count) { CUDA_CHECK(cudaMalloc(&data, count * sizeof(float))); }
  ~DeviceBuffer() { if (data) cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct PinnedBuffer {
  float* data = nullptr;
  explicit PinnedBuffer(std::size_t count) { CUDA_CHECK(cudaMallocHost(&data, count * sizeof(float))); }
  ~PinnedBuffer() { if (data) cudaFreeHost(data); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

struct Event {
  cudaEvent_t value{};
  Event() { CUDA_CHECK(cudaEventCreate(&value)); }
  ~Event() { if (value) cudaEventDestroy(value); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};

struct Timing {
  std::vector<float> trials;
  float median = 0;
  float min = 0;
  float max = 0;
};

Timing summarize(std::vector<float> samples) {
  if (samples.empty()) throw std::invalid_argument("no timing samples");
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t middle = sorted.size() / 2;
  return {std::move(samples), sorted.size() % 2 ? sorted[middle] :
                                      (sorted[middle - 1] + sorted[middle]) / 2.0f,
          sorted.front(), sorted.back()};
}

template <typename Action>
Timing time_cuda(Action action, int trials) {
  Event start, stop;
  for (int i = 0; i < kWarmups; ++i) {
    action();
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> samples;
  samples.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    CUDA_CHECK(cudaEventRecord(start.value));
    action();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop.value));
    CUDA_CHECK(cudaEventSynchronize(stop.value));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.value, stop.value));
    samples.push_back(milliseconds);
  }
  return summarize(std::move(samples));
}

void launch(bool tiled, const DeviceBuffer& a, const DeviceBuffer& b, DeviceBuffer& c,
            matmul::Shape shape) {
  const dim3 block(kTile, kTile);
  const dim3 grid((shape.n + kTile - 1) / kTile, (shape.m + kTile - 1) / kTile);
  if (tiled) {
    tiled_kernel<<<grid, block>>>(a.data, b.data, c.data, shape.m, shape.k, shape.n);
  } else {
    naive_kernel<<<grid, block>>>(a.data, b.data, c.data, shape.m, shape.k, shape.n);
  }
  CUDA_CHECK(cudaGetLastError());
}

struct KernelResult {
  std::string name;
  matmul::ErrorStats correctness;
  Timing timing;
  double gflops = 0;
  double speedup = 0;
};

struct ShapeResult {
  matmul::Shape shape;
  Timing transfer;
  std::vector<KernelResult> kernels;
};

std::vector<float> make_input(std::size_t count, std::uint32_t seed) {
  return matmul::deterministic_matrix(count, seed);
}

ShapeResult run_shape(matmul::Shape shape, int trials, bool benchmark) {
  const std::size_t a_count = static_cast<std::size_t>(shape.m) * shape.k;
  const std::size_t b_count = static_cast<std::size_t>(shape.k) * shape.n;
  const std::size_t c_count = static_cast<std::size_t>(shape.m) * shape.n;
  const auto a = make_input(a_count, 2026u + static_cast<unsigned>(shape.m));
  const auto b = make_input(b_count, 4061u + static_cast<unsigned>(shape.n));
  const auto expected = matmul::cpu_reference(a, b, shape);
  PinnedBuffer host_a(a_count), host_b(b_count);
  std::copy(a.begin(), a.end(), host_a.data);
  std::copy(b.begin(), b.end(), host_b.data);
  DeviceBuffer device_a(a_count), device_b(b_count), device_c(c_count);
  auto copy_inputs = [&] {
    CUDA_CHECK(cudaMemcpyAsync(device_a.data, host_a.data, a_count * sizeof(float),
                               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(device_b.data, host_b.data, b_count * sizeof(float),
                               cudaMemcpyHostToDevice));
  };
  copy_inputs();
  CUDA_CHECK(cudaDeviceSynchronize());
  ShapeResult result{shape, {}, {}};
  if (benchmark) result.transfer = time_cuda(copy_inputs, trials);

  for (bool tiled : {false, true}) {
    launch(tiled, device_a, device_b, device_c, shape);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> actual(c_count);
    CUDA_CHECK(cudaMemcpy(actual.data(), device_c.data, c_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    auto errors = matmul::compare(actual, expected);
    const std::string name = tiled ? "tiled" : "naive";
    std::cout << shape.m << 'x' << shape.k << " * " << shape.k << 'x' << shape.n << ' '
              << name << ": " << (errors.passed ? "PASS" : "FAIL")
              << " max_abs=" << errors.max_absolute
              << " max_rel=" << errors.max_relative << '\n';
    if (!errors.passed) throw std::runtime_error("GPU correctness failure: " + name);
    KernelResult kernel{name, errors, {}, 0, 0};
    if (benchmark) {
      kernel.timing = time_cuda([&] { launch(tiled, device_a, device_b, device_c, shape); }, trials);
      kernel.gflops = 2.0 * shape.m * shape.k * shape.n /
                      (static_cast<double>(kernel.timing.median) * 1e6);
      std::cout << "  median=" << kernel.timing.median << " ms"
                << " range=[" << kernel.timing.min << ", " << kernel.timing.max << "] ms"
                << " GFLOP/s=" << kernel.gflops << '\n';
    }
    result.kernels.push_back(std::move(kernel));
  }
  if (benchmark) {
    const double naive_ms = result.kernels[0].timing.median;
    for (auto& kernel : result.kernels) {
      kernel.speedup = naive_ms / kernel.timing.median;
    }
  }
  return result;
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec << std::setfill(' ');
        } else {
          out << c;
        }
    }
  }
  return out.str();
}

void write_timing(std::ostream& out, const Timing& timing) {
  out << "{\"median_ms\":" << timing.median << ",\"min_ms\":" << timing.min
      << ",\"max_ms\":" << timing.max << ",\"trials_ms\":[";
  for (std::size_t i = 0; i < timing.trials.size(); ++i) {
    if (i) out << ',';
    out << timing.trials[i];
  }
  out << "]}";
}

void write_json(const std::string& path, const cudaDeviceProp& gpu,
                int runtime_version, int driver_version, int trials,
                bool expanded, const std::vector<ShapeResult>& results) {
  const auto now = std::chrono::system_clock::now();
  const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &timestamp);
#else
  gmtime_r(&timestamp, &utc);
#endif
  const std::filesystem::path destination(path);
  if (destination.has_parent_path()) std::filesystem::create_directories(destination.parent_path());
  std::ofstream out(destination);
  if (!out) throw std::runtime_error("cannot open results file: " + path);
  out << std::setprecision(9);
  out << "{\n  \"schema_version\": 1,\n  \"generated_at_utc\": \""
      << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ") << "\",\n"
      << "  \"environment\": {\"gpu_name\": \"" << json_escape(gpu.name)
      << "\", \"compute_capability\": \"" << gpu.major << '.' << gpu.minor
      << "\", \"cuda_runtime_version\": " << runtime_version
      << ", \"cuda_driver_version\": " << driver_version
      << ", \"cuda_header_version\": " << CUDART_VERSION
      << ", \"cuda_compiler_version\": \"" << __CUDACC_VER_MAJOR__ << '.'
      << __CUDACC_VER_MINOR__ << "\"},\n"
      << "  \"methodology\": {\"warmups\": " << kWarmups
      << ", \"trials\": " << trials
      << ", \"sweep\": \"" << (expanded ? "expanded" : "short")
      << "\", \"timing\": \"CUDA events in default stream; kernel-only and pinned-host H2D measured separately; median, minimum and maximum of repeated trials\"},\n"
      << "  \"results\": [\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    out << "    {\"m\": " << result.shape.m << ", \"k\": " << result.shape.k
        << ", \"n\": " << result.shape.n << ", \"h2d\": ";
    write_timing(out, result.transfer);
    out << ", \"kernels\": [\n";
    for (std::size_t j = 0; j < result.kernels.size(); ++j) {
      const auto& kernel = result.kernels[j];
      out << "      {\"name\": \"" << kernel.name
          << "\", \"block\": [16, 16], \"tile_size\": "
          << (kernel.name == "tiled" ? kTile : 0)
          << ", \"correctness\": {\"passed\": true, \"max_absolute_error\": "
          << kernel.correctness.max_absolute << ", \"max_relative_error\": "
          << kernel.correctness.max_relative << "}, \"timing\": ";
      write_timing(out, kernel.timing);
      out << ", \"gflops\": " << kernel.gflops
          << ", \"speedup_vs_naive\": " << kernel.speedup << '}';
      if (j + 1 < result.kernels.size()) out << ',';
      out << '\n';
    }
    out << "    ]}";
    if (i + 1 < results.size()) out << ',';
    out << '\n';
  }
  out << "  ]\n}\n";
  if (!out) throw std::runtime_error("failed while writing results file: " + path);
}

void usage(const char* program) {
  std::cout << "Usage: " << program
            << " [--test] [--benchmark] [--expanded] [--trials N] [--output FILE]\n"
            << "No mode flags: run tests and the short benchmark. --expanded adds larger shapes.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool test = false, benchmark = false, expanded = false;
    int trials = kDefaultTrials;
    std::string output = "docs/results/benchmark.json";
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--test") test = true;
      else if (argument == "--benchmark") benchmark = true;
      else if (argument == "--expanded") expanded = true;
      else if (argument == "--trials" && i + 1 < argc) {
        trials = std::stoi(argv[++i]);
      } else if (argument == "--output" && i + 1 < argc) {
        output = argv[++i];
      } else if (argument == "--help" || argument == "-h") {
        usage(argv[0]);
        return 0;
      } else {
        usage(argv[0]);
        throw std::invalid_argument("unknown or incomplete argument: " + argument);
      }
    }
    if (trials < 1 || trials > 1000) throw std::invalid_argument("trials must be 1..1000");
    if (!test && !benchmark) test = benchmark = true;
    if (expanded && !benchmark) throw std::invalid_argument("--expanded requires --benchmark");
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) throw std::runtime_error("No NVIDIA CUDA GPU found. Select a GPU runtime in Colab.");
    cudaDeviceProp gpu{};
    CUDA_CHECK(cudaGetDeviceProperties(&gpu, 0));
    CUDA_CHECK(cudaSetDevice(0));
    int runtime_version = 0, driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    std::cout << "GPU: " << gpu.name << " (compute capability " << gpu.major << '.' << gpu.minor
              << ")\nCUDA runtime " << runtime_version << ", driver " << driver_version << '\n';

    if (test) {
      const std::vector<matmul::Shape> cases{{1, 1, 1}, {2, 3, 4}, {7, 11, 5},
                                             {16, 16, 16}, {17, 19, 23}, {31, 33, 15},
                                             {37, 5, 41}};
      std::cout << "Correctness suite (absolute tolerance 1e-3 + relative tolerance 1e-4 * |reference|):\n";
      for (auto shape : cases) run_shape(shape, trials, false);
    }
    if (benchmark) {
      std::vector<matmul::Shape> shapes{{64, 64, 64}, {128, 128, 128},
                                        {256, 256, 256}, {127, 65, 191},
                                        {256, 128, 384}};
      if (expanded) {
        shapes.insert(shapes.end(), {{512, 512, 512}, {768, 512, 256}, {511, 257, 769}});
      }
      std::vector<ShapeResult> results;
      for (auto shape : shapes) results.push_back(run_shape(shape, trials, true));
      write_json(output, gpu, runtime_version, driver_version, trials, expanded, results);
      std::cout << "Wrote measured results to " << output << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
