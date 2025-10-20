#!/usr/bin/env python3
"""
Example script demonstrating remote GPU execution for SpMV
"""

import numpy as np
from remote_client import RemoteGPUClient
import os


def generate_test_matrix_csr(num_rows, num_cols, density=0.01):
    """Generate a random sparse matrix in CSR format"""
    nnz = int(num_rows * num_cols * density)

    # Generate random non-zeros
    rows = np.random.randint(0, num_rows, nnz)
    cols = np.random.randint(0, num_cols, nnz)
    values = np.random.randn(nnz).astype(np.float32)

    # Sort by row then column
    indices = np.lexsort((cols, rows))
    rows = rows[indices]
    cols = cols[indices]
    values = values[indices]

    # Build CSR format
    row_ptr = np.zeros(num_rows + 1, dtype=np.int32)
    for i in range(nnz):
        row_ptr[rows[i] + 1] += 1
    row_ptr = np.cumsum(row_ptr)

    return {
        'num_rows': num_rows,
        'num_cols': num_cols,
        'nnz': nnz,
        'values': values,
        'col_indices': cols.astype(np.int32),
        'row_ptr': row_ptr,
    }


def main():
    # Configuration
    SERVER_URL = os.environ.get('GPU_SERVER_URL', 'http://localhost:5000')
    API_KEY = os.environ.get('GPU_API_KEY', 'your-secret-api-key-change-this')

    print("=" * 60)
    print("Remote GPU SpMV Execution Example")
    print("=" * 60)

    # Initialize client
    print(f"\n1. Connecting to GPU server: {SERVER_URL}")
    client = RemoteGPUClient(SERVER_URL, API_KEY)

    # Health check
    if not client.health_check():
        print("ERROR: GPU server is not available")
        return
    print("   Server is healthy")

    # Get GPU info
    gpu_info = client.get_gpu_info()
    if gpu_info:
        print(f"\n2. GPU Information:")
        print(f"   Device: {gpu_info['device_name']}")
        print(f"   Compute Capability: {gpu_info['compute_capability']}")
        print(f"   Total Memory: {gpu_info['total_memory_mb']} MB")
        print(f"   Multiprocessors: {gpu_info['multiprocessor_count']}")

    # Generate test matrix
    print(f"\n3. Generating test matrix...")
    num_rows = 10000
    num_cols = 10000
    density = 0.01

    matrix_data = generate_test_matrix_csr(num_rows, num_cols, density)
    print(f"   Matrix: {num_rows} x {num_cols}")
    print(f"   Non-zeros: {matrix_data['nnz']}")
    print(f"   Density: {density * 100:.2f}%")

    # Generate input vector
    x = np.random.randn(num_cols).astype(np.float32)
    matrix_data['x'] = x

    # Compile and execute
    print(f"\n4. Compiling CUDA kernel...")
    kernel_path = os.path.join(os.path.dirname(__file__), '..', 'kernels', 'simple_spmv.cu')
    kernel_path = os.path.abspath(kernel_path)

    print(f"\n5. Executing SpMV on remote GPU...")
    result = client.compile_and_execute_spmv(
        kernel_path,
        matrix_data,
        arch='sm_70'  # Change to match your GPU architecture
    )

    # Display results
    print(f"\n6. Results:")
    print(f"   Execution time:    {result['execution_time_ms']:.3f} ms")
    print(f"   Transfer H2D time: {result['transfer_h2d_ms']:.3f} ms")
    print(f"   Transfer D2H time: {result['transfer_d2h_ms']:.3f} ms")
    print(f"   Total time:        {result['total_time_ms']:.3f} ms")

    # Verify result
    y_gpu = np.array(result['y'], dtype=np.float32)
    print(f"\n   Output vector shape: {y_gpu.shape}")
    print(f"   Output vector norm: {np.linalg.norm(y_gpu):.6f}")

    # Compute reference on CPU for verification
    print(f"\n7. Verifying result (computing reference on CPU)...")
    y_cpu = np.zeros(num_rows, dtype=np.float32)
    for i in range(num_rows):
        for j in range(matrix_data['row_ptr'][i], matrix_data['row_ptr'][i + 1]):
            col = matrix_data['col_indices'][j]
            y_cpu[i] += matrix_data['values'][j] * x[col]

    # Compare results
    diff = np.linalg.norm(y_gpu - y_cpu)
    rel_error = diff / (np.linalg.norm(y_cpu) + 1e-10)

    print(f"   Relative error: {rel_error:.2e}")
    if rel_error < 1e-5:
        print("   ✓ Results match!")
    else:
        print("   ✗ Results do not match!")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
