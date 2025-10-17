CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3 -march=native
INCLUDES = -I./include
LDFLAGS = -lm

# Main tools
MTX2SPASM_SRCS = src/mtx2spasm.cpp mmio.c
MTX2SPASM_OBJS = $(MTX2SPASM_SRCS:.cpp=.o)
MTX2SPASM_OBJS := $(MTX2SPASM_OBJS:.c=.o)

SPASM_INFO_SRCS = src/spasm_info.cpp
SPASM_INFO_OBJS = $(SPASM_INFO_SRCS:.cpp=.o)

# Target executables
MTX2SPASM = mtx2spasm
SPASM_INFO = spasm_info

# Default target - build main tools
all: $(MTX2SPASM) $(SPASM_INFO)

# Build mtx2spasm converter
$(MTX2SPASM): $(MTX2SPASM_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build spasm_info tool
$(SPASM_INFO): $(SPASM_INFO_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile C++ files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile C files
%.o: %.c
	gcc -O3 -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(MTX2SPASM) $(SPASM_INFO) $(MTX2SPASM_OBJS) $(SPASM_INFO_OBJS) src/*.o *.o

# Install (optional)
install: all
	@echo "Binaries built successfully:"
	@echo "  - mtx2spasm: Converts MTX format to SPASM format"
	@echo "  - spasm_info: Displays information about SPASM files"

.PHONY: all clean install