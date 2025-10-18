# SpMV Implementation for COO and SPASM Formats

This document describes the Sparse Matrix-Vector multiplication (SpMV) implementations for both COO (Coordinate) and SPASM formats.

## Overview

We have implemented and verified two SpMV implementations:
1. **COO SpMV**: Traditional coordinate format implementation
2. **SPASM SpMV**: Template-based block SpMV implementation using the SPASM compressed format

## Implementation Details

### COO SpMV (`spmv_coo.h`)

The COO implementation is straightforward:
```cpp
for each non-zero entry (row, col, value):
    y[row] += value * x[col]
```

Time Complexity: O(nnz) where nnz is the number of non-zero elements.

### SPASM SpMV (`spmv_spasm.h`)

The SPASM implementation uses template-based block operations:

1. **Template-Specific Functions**: For each of the 16 template patterns, we have a specialized function that performs 4x4 block SpMV efficiently.

2. **Block Processing**:
   - Build a block-to-tile mapping table for O(1) tile lookup
   - Process all blocks sequentially
   - For each block:
     - Extract template ID, row/col indices from position encoding
     - Look up which tile the block belongs to
     - Calculate global position using tile offset
     - Execute the corresponding template function
   - Accumulate results to the output vector

3. **Key Features**:
   - **Function Lookup Table**: Uses `std::array<BlockSpMVFunc, 16>` for O(1) template function lookup
   - **Efficient Tile Mapping**: Pre-computes block-to-tile mapping using `tileBlockRanges`
   - **Bounds Checking**: Handles matrices that aren't multiples of 4
   - **Tile-Aware**: Correctly handles tile offsets for global position calculation

### Template Pattern Examples

#### Template 0 (0x000f): Horizontal Row Pattern
```
* * * *
. . . .
. . . .
. . . .
```
Implementation:
```cpp
y[0] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
```

#### Template 4 (0x1111): Vertical Column Pattern
```
* . . .
* . . .
* . . .
* . . .
```
Implementation:
```cpp
float x0 = x[0];
y[0] += values[0] * x0;
y[1] += values[1] * x0;
y[2] += values[2] * x0;
y[3] += values[3] * x0;
```

#### Template 12 (0x8421): Diagonal Pattern
```
* . . .
. * . .
. . * .
. . . *
```
Implementation:
```cpp
y[0] += values[0] * x[0];
y[1] += values[1] * x[1];
y[2] += values[2] * x[2];
y[3] += values[3] * x[3];
```

## Verification Program

The `spmv_compare` program verifies that both implementations produce identical results:

### Usage
```bash
./spmv_compare <matrix.mtx> <matrix.spasm>
```

### Example Output
```
========================================
SpMV Comparison Test
========================================

Matrix: 1138 x 1138
Non-zeros: 2596

Step 5: Performing COO SpMV...
  COO SpMV completed in 5 μs

Step 6: Performing SPASM SpMV...
  SPASM SpMV completed in 45 μs

Step 7: Comparing results...
----------------------------------------
COO vs SPASM:
  Max absolute error: 9.765625e-04
  Avg absolute error: 9.266979e-06
  Relative error: 3.533840e-08
  Differences > tolerance: 0
  Results match: YES

Performance Summary:
----------------------------------------
  COO SpMV:                  5 μs
  SPASM SpMV:               45 μs
  Speedup (COO/SPASM): 0.11x

Overall Result: SUCCESS - All implementations match!
```

## Test Results

All test matrices produce matching results within numerical tolerance:

| Matrix | Size | NNZ | COO Time | SPASM Time | Speedup | Match |
|--------|------|-----|----------|------------|---------|-------|
| test_mixed | 16x16 | 44 | <1 μs | 1 μs | 0.00x | ✓ YES |
| 1138_bus | 1138x1138 | 2,596 | 5 μs | 45 μs | 0.11x | ✓ YES |
| Chebyshev4 | 68121x68121 | 5,377,761 | 18.1 ms | 23.9 ms | 0.76x | ✓ YES |

### Numerical Accuracy

- **Relative Error**: < 1.03e-07 for all test cases
- **Max Absolute Error**: < 64 for large matrices (Chebyshev4)
- **Tolerance**: 1% relative error for large values, 1% absolute error for small values

The small numerical differences are expected due to:
1. Different order of floating-point operations
2. Accumulation patterns in block processing
3. Compiler optimizations

### Performance Notes

The current SPASM implementation builds a block-to-tile mapping table at runtime, which adds initialization overhead. For small matrices, this overhead dominates. For large matrices (like Chebyshev4), the SPASM format achieves competitive performance (0.76x speedup compared to COO).

## Implementation Notes

### Random Vector Generation

Input vectors are generated using:
```cpp
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
```

### Block SpMV Design

Each template pattern has a hand-optimized implementation that:
1. Minimizes memory accesses
2. Reuses loaded values when possible
3. Performs operations in an order that maximizes ILP (Instruction-Level Parallelism)

For example, column patterns reuse the input vector element:
```cpp
float x0 = x[0];  // Load once
y[0] += values[0] * x0;  // Reuse
y[1] += values[1] * x0;  // Reuse
y[2] += values[2] * x0;  // Reuse
y[3] += values[3] * x0;  // Reuse
```

### Block-to-Tile Mapping

The SPASM implementation uses an efficient mapping strategy:

```cpp
// Build mapping table once (O(numBlocks))
std::vector<int> blockToTile(A.getNumPositions(), -1);
for (size_t tileIdx = 0; tileIdx < A.tilePositions.size(); tileIdx++) {
    const auto& range = A.tileBlockRanges[tileIdx];
    for (uint32_t blockIdx = range.blockStart; blockIdx < range.blockEnd; blockIdx++) {
        blockToTile[blockIdx] = tileIdx;
    }
}

// Use mapping during computation (O(1) lookup)
int tileIdx = blockToTile[posIdx];
if (tileIdx >= 0) {
    tileRow = A.tilePositions[tileIdx].tileRowIdx;
    tileCol = A.tilePositions[tileIdx].tileColIdx;
}
```

This approach:
- Pre-computes the mapping once
- Enables O(1) tile lookup per block
- Handles blocks not belonging to any tile gracefully

### Generic Fallback

For patterns not in the standard 16 templates, a generic implementation is provided:
```cpp
void blockSpMV_generic(PatternMask pattern, const float* values,
                       const float* x, float* y) {
    int val_idx = 0;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            if (pattern & (1 << (row * 4 + col))) {
                y[row] += values[val_idx++] * x[col];
            }
        }
    }
}
```

## Future Optimizations

Potential improvements for the SPASM SpMV implementation:

1. **SIMD Vectorization**: Use SSE/AVX instructions for block operations
2. **Parallel Processing**: Process tiles in parallel using OpenMP or threads
3. **Persistent Mapping**: Store block-to-tile mapping with the SPASM file to avoid rebuild
4. **Tile-Based Processing**: Process by tiles instead of blocks for better cache locality
5. **Prefetching**: Add explicit prefetch instructions for next block data
6. **GPU Implementation**: Port to CUDA/HIP for GPU acceleration

Note: The current implementation removed the `spmvDirect()` method and only uses the optimized `spmv()` method with efficient block-to-tile mapping.

## Building

Add to your Makefile:
```makefile
SPMV_COMPARE = spmv_compare
$(SPMV_COMPARE): src/spmv_compare.o mmio.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
```

Then build:
```bash
make spmv_compare
```

## Conclusion

Both implementations produce numerically equivalent results, confirming the correctness of:
1. The SPASM format conversion
2. The template-based block decomposition
3. The SpMV computation using templates

The SPASM format successfully compresses the sparse matrix while enabling efficient block-based computation through template specialization.