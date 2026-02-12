# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++23 -Wall -Wextra -Wpedantic -O2
DEBUGFLAGS := -g -O0 -DDEBUG
INCLUDES := -Iinclude -Isrc

# Directories
SRC_DIR := src
BUILD_DIR := build
INC_DIR := include

# Target executable name
TARGET := $(BUILD_DIR)/program

# Find all .cpp files recursively in src/
SOURCES := $(shell find $(SRC_DIR) -name '*.cpp')

# Generate object file paths in build/ mirroring src/ structure
OBJECTS := $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Dependency files
DEPS := $(OBJECTS:.o=.d)

# Default target
.PHONY: all
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@
	@echo "Build complete: $(TARGET)"

# Compile source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# Include dependency files
-include $(DEPS)

# Debug build
.PHONY: debug
debug: CXXFLAGS := -std=c++23 -Wall -Wextra -Wpedantic $(DEBUGFLAGS)
debug: clean $(TARGET)

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	@echo "Build directory cleaned"

# Run the program
.PHONY: run
run: $(TARGET)
	./$(TARGET)

# Rebuild from scratch
.PHONY: rebuild
rebuild: clean all

# Print variables for debugging the Makefile
.PHONY: info
info:
	@echo "CXX:      $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "SOURCES:  $(SOURCES)"
	@echo "OBJECTS:  $(OBJECTS)"
	@echo "TARGET:   $(TARGET)"