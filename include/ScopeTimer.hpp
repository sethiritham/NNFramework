#include <chrono>
#include <iostream>
#include <string>

struct ScopeTimer {
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::string scope_name;

  double time_in_ms = 0;

  ScopeTimer(std::string name) : scope_name(name) {
    start = std::chrono::high_resolution_clock::now();
  }

  void time_display() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    time_in_ms = duration / 1000.0;

    // Output format optimized for terminal readability or piping into a CSV
    std::cout << "[BENCHMARK] " << scope_name << " | Latency: " << duration
              << " us (" << duration / 1000.0 << " ms)\n";
  }

  double get_time_ms() { return time_in_ms; }

  ~ScopeTimer() {}
};
