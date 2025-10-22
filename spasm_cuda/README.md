# SPASM CUDA SpMV Implementation

This directory contains a CUDA implementation of Sparse Matrix-Vector Multiplication (SpMV) using the SPASM format.

## Architecture

### Kernel Design
- **1 tile per CUDA block**: Each CUDA block processes one SPASM tile
- **1 4x4 block per thread**: Each thread processes one 4x4 sparse block
- **Template-based optimization**: Uses specialized functions for 16 common sparsity patterns

### File Structure
```
spasm_cuda/
├── include/
│   └── spasm_cuda.h          # Main header file
├── src/
│   ├── spasm_io.cu           # SPASM file loading
│   ├── spasm_spmv_cpu.cu     # CPU reference implementation
│   └── spasm_spmv_gpu.cu     # GPU memory management
├── kernels/
│   └── spasm_spmv_kernel.cu  # CUDA kernel implementation
├── main.cu                   # Benchmark program
└── Makefile                  # Build configuration
```

## Building

The code requires NVIDIA CUDA Toolkit with compute capability 8.0 (for A100 GPU).

```bash
cd spasm_cuda
make
```

This will produce the executable `spasm_spmv_benchmark`.

## Usage

```bash
./spasm_spmv_benchmark <spasm_file> [options]

Options:
  -n <iterations>  Number of iterations for GPU benchmark (default: 100)
  -h, --help       Print help message
```

### Example

```bash
# Run with default 100 iterations
./spasm_spmv_benchmark matrix.spasm

# Run with 1000 iterations for more accurate timing
./spasm_spmv_benchmark matrix.spasm -n 1000
```

## Output

The program will:
1. Load the SPASM matrix from file
2. Run CPU SpMV (single iteration for verification)
3. Run GPU SpMV (multiple iterations for performance measurement)
4. Compare CPU and GPU results
5. Report performance metrics (time, GFLOP/s, speedup)

### Example Output
```
========================================
SPASM SpMV Benchmark
========================================
File: matrix.spasm
GPU iterations: 100

Loading SPASM matrix...
Loaded SPASM matrix: 10000x10000, nnz=100000, tiles=100, positions=5000

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

========================================
SUCCESS: GPU results match CPU results!
========================================
Speedup: 41.9x
```

## Implementation Details

### CUDA Kernel Strategy

The kernel uses a tile-based approach:
- Each tile is assigned to one CUDA block
- Within each tile, threads process 4x4 blocks in parallel
- Uses atomic operations to accumulate results to avoid race conditions

### Template Patterns

The implementation supports 16 predefined sparsity patterns:
- Patterns 0-3: Full rows (0x000f, 0x00f0, 0x0f00, 0xf000)
- Patterns 4-7: Full columns (0x1111, 0x2222, 0x4444, 0x8888)
- Patterns 8-11: 2x2 sub-blocks (0x0033, 0x00cc, 0x3300, 0xcc00)
- Patterns 12-15: Diagonals (0x8421, 0x4218, 0x2184, 0x1842)

### Memory Layout

The SPASM format stores:
- **Tile positions**: COO format for non-empty tiles
- **Tile block ranges**: Start/end indices for each tile's blocks
- **Position encodings**: Packed 32-bit encoding (column index, row index, template ID, flags)
- **Values**: 4 float values per position encoding
- **Template patterns**: 16-bit masks for sparsity patterns

## Performance Considerations

- **Thread utilization**: Each block processes all 4x4 blocks within a tile in parallel
- **Memory coalescing**: Position encodings and values are accessed sequentially
- **Atomic operations**: Used for result accumulation (may be a bottleneck for dense tiles)
- **Occupancy**: Limited by register usage and shared memory (not used in this basic version)

## Testing on A100

When you transfer this to the A100 server:

1. Copy the entire `spasm_cuda` directory
2. Ensure you have a SPASM format file (use `mtx2spasm` from `spasm_converter` to convert)
3. Run `make` to compile
4. Execute the benchmark

If you need to adjust the compute capability, edit the Makefile and change `-arch=sm_80` to match your GPU.
