# Remote GPU Client

Simple command-line tools for executing SpMV on remote GPU server.

## Quick Start

### 1. Configure Connection

Edit `.env` file:
```bash
GPU_SERVER_URL=http://120.25.194.125:2110
GPU_API_KEY=your-secret-api-key
```

### 2. Run SpMV with MTX File

```bash
python spmv_mtx.py your_matrix.mtx
```

That's it! The script will:
- Upload MTX file (if not already on server)
- Compile CUDA kernel
- Execute SpMV
- Display results

## Generate Test Matrix

```bash
# Generate 1000x1000 matrix with 1% density
python generate_mtx.py test.mtx --size 1000 --density 0.01

# Run SpMV
python spmv_mtx.py test.mtx
```

## Command-Line Options

### spmv_mtx.py

```bash
python spmv_mtx.py <mtx_file> [options]

Options:
  --kernel <path>     CUDA kernel file (default: ../kernels/simple_spmv.cu)
  --server <url>      GPU server URL (default: from .env)
  --api-key <key>     API key (default: from .env)
```

Example:
```bash
python spmv_mtx.py matrix.mtx --kernel custom_kernel.cu
```

### generate_mtx.py

```bash
python generate_mtx.py <output> [options]

Options:
  --size <n>          Matrix size (default: 1000)
  --density <d>       Sparsity density (default: 0.01)
```

Example:
```bash
python generate_mtx.py big_matrix.mtx --size 10000 --density 0.001
```

## Output

```
Connecting to http://120.25.194.125:2110...
Processing MTX file: matrix.mtx
MTX file already exists on server: matrix.mtx
Auto-detected server GPU architecture: sm_75
Compiling kernels/simple_spmv.cu -> kernels/simple_spmv.cubin (arch=sm_75)
Compilation successful: kernels/simple_spmv.cubin
Executing SpMV with MTX file: matrix.mtx

Results:
  Matrix:      1000x1000, nnz=10000
  Kernel time: 0.045 ms
  Throughput:  0.44 GFLOPS
```

## Advanced Examples

### Use Your Own Kernel

```bash
python spmv_mtx.py matrix.mtx --kernel my_optimized_kernel.cu
```

### Override Server Settings

```bash
python spmv_mtx.py matrix.mtx --server http://another-server:5000 --api-key different-key
```

## Other Scripts

The following example scripts are also available but not needed for basic usage:

- `example_spmv.py` - Full example with generated data
- `example_mtx.py` - Full example with MTX workflow
- `test_connection.py` - Test server connection
- `test_multiple_runs.py` - Test multiple executions

For simple usage, just use **`spmv_mtx.py`**.
