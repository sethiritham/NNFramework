#include <chrono>
#include <iostream>
#include <string>

struct ScopeTimer {
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::string scope_name;

  ScopeTimer(std::string name) : scope_name(name) {
    start = std::chrono::high_resolution_clock::now();
  }

  ~ScopeTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    // Output format optimized for terminal readability or piping into a CSV
    std::cout << "[BENCHMARK] " << scope_name << " | Latency: " << duration
              << " us (" << duration / 1000.0 << " ms)\n";
  }
};
