NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++14

TARGET = spasm_spmv_benchmark

OBJS = main.o \
       spasm_io.o \
       spasm_spmv_cpu.o \
       spasm_spmv_gpu.o \
       spasm_spmv_kernel.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

main.o: main.cu include/spasm_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c main.cu -o $@

spasm_io.o: src/spasm_io.cu include/spasm_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c src/spasm_io.cu -o $@

spasm_spmv_cpu.o: src/spasm_spmv_cpu.cu include/spasm_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c src/spasm_spmv_cpu.cu -o $@

spasm_spmv_gpu.o: src/spasm_spmv_gpu.cu include/spasm_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c src/spasm_spmv_gpu.cu -o $@

spasm_spmv_kernel.o: kernels/spasm_spmv_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -c kernels/spasm_spmv_kernel.cu -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
