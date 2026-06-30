# Compiler settings
CXX = g++

# CXXFLAGS Explained:
# -Iinclude     : Tells the compiler to look in the include/ directory for .hpp files
# -O3           : Maximize speed (enables loop unrolling, vectorization, etc.)
# -march=native : Instructs the compiler to use hardware-specific instructions for YOUR CPU
# -std=c++17    : Modern C++ standard
# -Wall -Wextra : Enable all standard warnings to catch bugs early
CXXFLAGS = -Wall -Wextra -std=c++17 -O3 -march=native -Iinclude

# Name of your final executable
TARGET = nn_engine

# Automatically find all .cpp files in src/ and its subdirectories
SRCS = $(wildcard src/*.cpp) $(wildcard src/*/*.cpp)

# Map the .cpp files to .o (object) files
OBJS = $(SRCS:.cpp=.o)

# Default target runs when you just type 'make'
all: $(TARGET)

# Linking step
$(TARGET): $(OBJS)
	@echo "Linking..."
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo "Build complete: ./$(TARGET)"

# Compilation step for individual files
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Quick command to build and run the executable
run: $(TARGET)
	./$(TARGET)

# Clean up binaries
clean:
	@echo "Cleaning build files..."
	rm -f $(OBJS) $(TARGET)

# Phony targets prevent conflicts
.PHONY: all clean run
