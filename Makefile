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
TARGET_TRAIN = train_engine
TARGET_PREDICT = inference_engine

SHARED_SRCS = $(wildcard src/math/*.cpp) $(wildcard src/architecture/*.cpp)
SHARED_OBJS = $(SHARED_SRCS:*.cpp=.o)

# Default target runs when you just type 'make'
all: $(TARGET_TRAIN) $(TARGET_PREDICT)

# Linking step for training
$(TARGET_TRAIN): $(SHARED_OBJS) src/train.o
	@echo "Linking Training Engine...."
	$(CXX) $(CXXFLAGS) -o $@ $^

# Linking step for inference
$(TARGET_PREDICT): $(SHARED_OBJS) src/Inference.o
	@echo "Linking Inference Engine...."
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compilation step for individual files
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Quick command to build and run the executable
run_train: $(TARGET_TRAIN)
	./$(TARGET_TRAIN)

run_inference: $(TARGET_PREDICT)
	./$(TARGET_PREDICT)

# Clean up binaries
clean:
	@echo "Cleaning build files..."
	rm -f $(SHARED_OBJS) $(TARGET_PREDICT) $(TARGET_TRAIN)

# Phony targets prevent conflicts
.PHONY: all clean run_train run_inference
