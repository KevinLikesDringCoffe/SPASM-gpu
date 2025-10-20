# SpMV Throughput Measurement

## What is Throughput?

Throughput measures the computational performance of SpMV (Sparse Matrix-Vector multiplication) in GFLOPS (Giga Floating Point Operations Per Second).

## Calculation

For SpMV operation `y = A * x` where A is a sparse matrix:

- **FLOPs**: Each non-zero element requires 2 operations (1 multiply + 1 add)
  ```
  Total FLOPs = 2 × nnz
  ```

- **GFLOPS**: Throughput in billions of FLOPs per second
  ```
  GFLOPS = (2 × nnz) / (execution_time_seconds) / 10^9
  ```

## Output Format

The service now returns additional performance metrics:

```json
{
  "y": [...],
  "execution_time_ms": 0.059,
  "transfer_h2d_ms": 0.123,
  "transfer_d2h_ms": 0.045,
  "transfer_time_ms": 0.168,
  "total_time_ms": 0.227,
  "num_rows": 10000,
  "num_cols": 10000,
  "nnz": 1000000,
  "gflops": 33.90
}
```

### Metrics Explained

- **execution_time_ms**: Pure GPU kernel execution time (warmup excluded)
- **transfer_h2d_ms**: Host to Device data transfer time
- **transfer_d2h_ms**: Device to Host result transfer time
- **total_time_ms**: Total time including all transfers and overhead
- **gflops**: Computational throughput (based on execution_time_ms)

## Performance Analysis

### Typical Performance Ranges

| GPU Type | Expected GFLOPS (CSR SpMV) |
|----------|---------------------------|
| V100     | 20-50 GFLOPS             |
| A100     | 40-80 GFLOPS             |
| RTX 3090 | 25-60 GFLOPS             |
| T4       | 15-35 GFLOPS             |

*Note: Actual performance depends on matrix structure, sparsity pattern, and problem size.*

### Factors Affecting Throughput

1. **Matrix Size**: Larger matrices typically achieve higher throughput
2. **Sparsity Pattern**: Regular patterns perform better than irregular ones
3. **nnz per row**: More uniform distribution improves performance
4. **Memory Bandwidth**: Often the limiting factor for SpMV

### Bandwidth vs Compute

SpMV is typically **memory-bound** rather than compute-bound:

```
Arithmetic Intensity = FLOPs / Bytes Transferred
                     = 2 / (sizeof(float) + sizeof(int) + sizeof(float))
                     = 2 / 12 = 0.17 FLOPs/Byte
```

This low arithmetic intensity means performance is limited by memory bandwidth rather than compute power.

## Example Output

```
6. Results:
   Execution time:    0.059 ms
   Transfer H2D time: 0.123 ms
   Transfer D2H time: 0.045 ms
   Total time:        0.227 ms

   Throughput:        33.90 GFLOPS
```

## Server Logs

The GPU service now logs throughput information:

```
INFO:__main__:Executing SpMV: rows=10000, cols=10000, nnz=1000000
INFO:__main__:Execution completed: 0.059 ms, 33.90 GFLOPS
```

## Benchmarking Tips

1. **Warm-up**: First execution includes JIT compilation overhead
2. **Multiple runs**: Average over multiple runs for stable measurements
3. **Problem size**: Use realistic problem sizes for meaningful results
4. **Comparison**: Compare with optimized libraries (cuSPARSE, etc.)

## Using Throughput Data

```python
result = client.execute_spmv_csr(cubin, matrix_data)

print(f"Performance: {result['gflops']:.2f} GFLOPS")
print(f"Efficiency: {result['gflops'] / theoretical_peak * 100:.1f}%")
```
