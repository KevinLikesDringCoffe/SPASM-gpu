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

SPMV_COMPARE_SRCS = src/spmv_compare.cpp mmio.c
SPMV_COMPARE_OBJS = $(SPMV_COMPARE_SRCS:.cpp=.o)
SPMV_COMPARE_OBJS := $(SPMV_COMPARE_OBJS:.c=.o)

# Target executables
MTX2SPASM = mtx2spasm
SPASM_INFO = spasm_info
SPMV_COMPARE = spmv_compare

# Default target - build main tools
all: $(MTX2SPASM) $(SPASM_INFO) $(SPMV_COMPARE)

# Build mtx2spasm converter
$(MTX2SPASM): $(MTX2SPASM_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build spasm_info tool
$(SPASM_INFO): $(SPASM_INFO_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build spmv_compare tool
$(SPMV_COMPARE): $(SPMV_COMPARE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile C++ files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile C files
%.o: %.c
	gcc -O3 -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(MTX2SPASM) $(SPASM_INFO) $(SPMV_COMPARE) $(MTX2SPASM_OBJS) $(SPASM_INFO_OBJS) $(SPMV_COMPARE_OBJS) src/*.o *.o

# Install (optional)
install: all
	@echo "Binaries built successfully:"
	@echo "  - mtx2spasm: Converts MTX format to SPASM format"
	@echo "  - spasm_info: Displays information about SPASM files"

.PHONY: all clean install