# SPASM CUDA SpMV Usage Guide

## Quick Start

### 1. Build the Code

```bash
cd spasm_cuda
make
```

This creates the executable `spasm_spmv_benchmark`.

### 2. Prepare Input Data

You need a SPASM format file. If you have an MTX (Matrix Market) file, convert it first:

```bash
cd ../spasm_converter
make
./mtx2spasm your_matrix.mtx output.spasm
```

### 3. Run the Benchmark

```bash
cd ../spasm_cuda
./spasm_spmv_benchmark output.spasm -n 100
```

Or use the test script:

```bash
./test_example.sh output.spasm 100
```

## Command Line Options

```
./spasm_spmv_benchmark <spasm_file> [options]

Required:
  <spasm_file>     Path to SPASM format matrix file

Options:
  -n <iterations>  Number of GPU iterations (default: 100)
  -h, --help       Show help message
```

## What the Program Does

1. **Loads** the SPASM matrix from the file
2. **Runs CPU SpMV** once (for correctness verification)
3. **Runs GPU SpMV** multiple times (for performance measurement)
4. **Compares** CPU and GPU results (tolerance: 1e-4)
5. **Reports** performance metrics:
   - CPU time and GFLOP/s
   - GPU time and GFLOP/s
   - Verification status
   - Speedup

## Expected Output

```
==========================================
SPASM SpMV Benchmark
==========================================
File: matrix.spasm
GPU iterations: 100

Loading SPASM matrix...
Loaded SPASM matrix: 10000x10000, nnz=100000, tiles=50, positions=2500

Running CPU SpMV...
CPU SpMV Performance:
  Time: 5.234 ms
  Performance: 38.2 GFLOP/s

Running GPU SpMV...
GPU SpMV Performance:
  Total time: 12.456 ms (100 iterations)
  Average time: 0.125 ms per iteration
  Performance: 1600.0 GFLOP/s

Verifying results...
Verification PASSED! Max error: 0.000001

==========================================
SUCCESS: GPU results match CPU results!
==========================================
Speedup: 41.9x
```

## Implementation Details

### Kernel Configuration
- **Grid size**: Number of tiles in the matrix
- **Block size**: Maximum number of 4x4 blocks per tile (capped at 1024)
- **Thread mapping**: Each thread processes one 4x4 sparse block

### Memory Usage
The program allocates GPU memory for:
- Tile positions (COO format)
- Tile block ranges
- Position encodings (packed 32-bit)
- Matrix values (float)
- Template patterns (16-bit masks)
- Input vector x
- Output vector y

### Performance Tips

1. **Use more iterations** (-n flag) for more accurate GPU timing
2. **Larger matrices** will show better GPU speedup
3. **Check tile distribution**: Imbalanced tiles may cause load imbalance

## Testing on A100

Since this machine doesn't have a GPU, you'll need to transfer to an A100 server:

```bash
# On this machine (create tarball)
cd /home/zywu/paperwork/SPASM-gpu
tar czf spasm_cuda.tar.gz spasm_cuda/

# On A100 server
scp <user>@<this-machine>:/home/zywu/paperwork/SPASM-gpu/spasm_cuda.tar.gz .
tar xzf spasm_cuda.tar.gz
cd spasm_cuda
make
./spasm_spmv_benchmark <your_matrix.spasm> -n 1000
```

## Troubleshooting

### Compilation Issues

**Problem**: `nvcc: command not found`
- **Solution**: Load CUDA module or set PATH to include CUDA toolkit

**Problem**: `unsupported GPU architecture 'compute_80'`
- **Solution**: Edit Makefile and change `-arch=sm_80` to match your GPU:
  - V100: `-arch=sm_70`
  - A100: `-arch=sm_80`
  - H100: `-arch=sm_90`

### Runtime Issues

**Problem**: `CUDA error ... invalid device`
- **Solution**: This machine has no GPU. Transfer to a GPU-enabled server.

**Problem**: `Verification FAILED`
- **Solution**: Check the matrix file is not corrupted. Try reducing tolerance or check for numerical issues.

**Problem**: `out of memory`
- **Solution**: Matrix is too large. Try a smaller matrix or use a GPU with more memory.

## Performance Analysis

The program reports:
- **CPU performance**: Single-threaded performance
- **GPU performance**: Parallel performance across all SMs
- **Speedup**: Ratio of CPU time to GPU time

Typical results on A100:
- Small matrices (< 1M nnz): 5-20x speedup
- Medium matrices (1-10M nnz): 20-50x speedup
- Large matrices (> 10M nnz): 50-100x speedup

## Code Structure

```
spasm_cuda/
├── include/spasm_cuda.h           # API definitions
├── src/
│   ├── spasm_io.cu                # File I/O
│   ├── spasm_spmv_cpu.cu          # CPU reference
│   └── spasm_spmv_gpu.cu          # GPU memory management
├── kernels/spasm_spmv_kernel.cu   # CUDA kernel
├── main.cu                        # Main program
├── Makefile                       # Build configuration
└── README.md                      # Technical documentation
```

## Next Steps

1. **Test with your matrices**: Convert your MTX files and benchmark
2. **Tune parameters**: Try different tile sizes when converting
3. **Analyze performance**: Compare with other SpMV implementations
4. **Optimize**: Consider shared memory, better load balancing, etc.
