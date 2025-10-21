# SPASM CUDA SpMV Implementation

High-performance CUDA implementation of Sparse Matrix-Vector Multiplication (SpMV) using the SPASM format.

## Features

- **Optimized CUDA Kernel**: Pattern-specific device functions for 16 common templates
- **Warp-level Optimization**: Minimized warp divergence by grouping same-pattern blocks
- **Memory Optimizations**:
  - Constant memory for template masks
  - Shared memory caching for input vectors
  - Coalesced memory access for values
- **CPU Reference**: SPASM-based CPU implementation for correctness verification
- **Performance Metrics**: GFLOPS, throughput (GNNZ/s), and speedup reporting

## Build Requirements

- CUDA Toolkit (tested with CUDA 11.0+)
- NVIDIA GPU with Compute Capability 8.0+ (A100, A40, etc.)
- GCC with C++17 support
- OpenMP

## Build Instructions

```bash
cd spasm_cuda
make
```

## Usage

```bash
./spasm_spmv_benchmark <spasm_file> [num_runs]
```

**Arguments:**
- `spasm_file`: Path to SPASM format matrix file
- `num_runs`: Number of SpMV iterations for timing (default: 100)

**Example:**
```bash
./spasm_spmv_benchmark ../matrices/example.spasm 100
```

## Output

The benchmark reports:
- Matrix information (dimensions, NNZ, compression ratio)
- CPU performance (time, GFLOPS)
- GPU performance (time, GFLOPS, throughput, speedup)
- Verification results (max/avg error, correctness)

## Architecture

### CUDA Kernel Design

**Tile-level Parallelism:**
- Each CUDA block processes one tile
- Threads within a block process position encodings

**Pattern Specialization:**
- 16 optimized device functions for common patterns:
  - Row patterns (0x000f, 0x00f0, 0x0f00, 0xf000)
  - Column patterns (0x1111, 0x2222, 0x4444, 0x8888)
  - 2x2 blocks (0x0033, 0x00cc, 0x3300, 0xcc00)
  - Diagonal patterns (0x8421, 0x4218, 0x2184, 0x1842)
- Generic fallback for arbitrary patterns

**Memory Hierarchy:**
- Constant memory: Template masks (16 × 2 bytes)
- Shared memory: Input vector tile cache
- Global memory: Coalesced access for values array

## Target Hardware

Optimized for NVIDIA A100 GPU (sm_80 architecture).

To change target architecture, edit `CUDA_ARCH` in Makefile:
```makefile
CUDA_ARCH := -arch=sm_XX
```

## Directory Structure

```
spasm_cuda/
├── kernels/
│   └── spasm_spmv_kernel.cu    # CUDA kernel implementation
├── src/
│   ├── spasm_spmv.cu           # Host-side CUDA API
│   └── cpu_reference.cpp       # CPU reference implementation
├── include/
│   └── spasm_cuda.h            # API header
├── utils/
│   └── timer.h                 # Timing utilities
├── main.cu                     # Benchmark program
├── Makefile
└── README.md
```

## Performance Tips

1. Ensure SPASM matrix has good pattern distribution
2. Use appropriate tile size (default: 1024)
3. Run multiple iterations for accurate timing
4. Verify GPU utilization with `nvidia-smi`
