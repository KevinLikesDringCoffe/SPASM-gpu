NVCC := nvcc
CXX := g++

CUDA_ARCH := -arch=sm_80

NVCC_FLAGS := -O3 $(CUDA_ARCH) -Xcompiler -fopenmp
CXX_FLAGS := -O3 -std=c++17 -fopenmp

INCLUDE_DIRS := -I. -I../spasm_converter/include -I../spasm_converter

SPASM_CONVERTER_DIR := ../spasm_converter
MMIO_OBJ := $(SPASM_CONVERTER_DIR)/mmio.o

SRC_DIR := src
KERNEL_DIR := kernels
BUILD_DIR := build

TARGET := spasm_spmv_benchmark

CUDA_SRCS := $(KERNEL_DIR)/spasm_spmv_kernel.cu $(SRC_DIR)/spasm_spmv.cu main.cu
CPP_SRCS := $(SRC_DIR)/cpu_reference.cpp

CUDA_OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(notdir $(CUDA_SRCS)))
CPP_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(notdir $(CPP_SRCS)))

all: $(BUILD_DIR) $(MMIO_OBJ) $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(MMIO_OBJ):
	@echo "Building MMIO library..."
	$(MAKE) -C $(SPASM_CONVERTER_DIR) mmio.o

$(BUILD_DIR)/%.o: $(KERNEL_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

$(TARGET): $(CUDA_OBJS) $(CPP_OBJS) $(MMIO_OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
