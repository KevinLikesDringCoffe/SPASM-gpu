NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70 --ptx
CUBIN_FLAGS = -O3 -arch=sm_70 -cubin

KERNELS = simple_spmv

all: ptx cubin

ptx: $(KERNELS:=.ptx)

cubin: $(KERNELS:=.cubin)

%.ptx: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

%.cubin: %.cu
	$(NVCC) $(CUBIN_FLAGS) $< -o $@

clean:
	rm -f *.ptx *.cubin

.PHONY: all ptx cubin clean
