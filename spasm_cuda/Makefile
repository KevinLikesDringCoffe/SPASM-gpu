NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++14

TARGET = spasm_spmv_benchmark
TARGET_INT = test_int

OBJS = main.o \
       spasm_io.o \
       spasm_spmv_cpu.o \
       spasm_spmv_gpu.o \
       spasm_spmv_kernel.o

OBJS_INT = test_int.o \
           spasm_io.o \
           spasm_spmv_int.o \
           spasm_spmv_gpu_int.o \
           spasm_spmv_kernel_int.o

all: $(TARGET) $(TARGET_INT)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(TARGET_INT): $(OBJS_INT)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

main.o: main.cu include/spasm_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c main.cu -o $@

test_int.o: test_int.cu include/spasm_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c test_int.cu -o $@

spasm_io.o: src/spasm_io.cu include/spasm_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c src/spasm_io.cu -o $@

spasm_spmv_cpu.o: src/spasm_spmv_cpu.cu include/spasm_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c src/spasm_spmv_cpu.cu -o $@

spasm_spmv_gpu.o: src/spasm_spmv_gpu.cu include/spasm_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c src/spasm_spmv_gpu.cu -o $@

spasm_spmv_kernel.o: kernels/spasm_spmv_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -c kernels/spasm_spmv_kernel.cu -o $@

spasm_spmv_int.o: src/spasm_spmv_int.cu include/spasm_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c src/spasm_spmv_int.cu -o $@

spasm_spmv_gpu_int.o: src/spasm_spmv_gpu_int.cu include/spasm_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c src/spasm_spmv_gpu_int.cu -o $@

spasm_spmv_kernel_int.o: kernels/spasm_spmv_kernel_int.cu
	$(NVCC) $(NVCC_FLAGS) -c kernels/spasm_spmv_kernel_int.cu -o $@

clean:
	rm -f $(OBJS) $(OBJS_INT) $(TARGET) $(TARGET_INT)

.PHONY: all clean
