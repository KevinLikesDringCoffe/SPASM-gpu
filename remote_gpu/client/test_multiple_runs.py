#!/usr/bin/env python3
"""
Test multiple consecutive runs to verify no context cleanup errors
"""

import numpy as np
from remote_client import RemoteGPUClient
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)


def generate_small_matrix(size=100):
    """Generate a small test matrix"""
    nnz = size * 10
    rows = np.random.randint(0, size, nnz)
    cols = np.random.randint(0, size, nnz)
    values = np.random.randn(nnz).astype(np.float32)

    indices = np.lexsort((cols, rows))
    rows = rows[indices]
    cols = cols[indices]
    values = values[indices]

    row_ptr = np.zeros(size + 1, dtype=np.int32)
    for i in range(nnz):
        row_ptr[rows[i] + 1] += 1
    row_ptr = np.cumsum(row_ptr)

    return {
        'num_rows': size,
        'num_cols': size,
        'nnz': nnz,
        'values': values,
        'col_indices': cols.astype(np.int32),
        'row_ptr': row_ptr,
        'x': np.random.randn(size).astype(np.float32)
    }


def main():
    SERVER_URL = os.environ.get('GPU_SERVER_URL', 'http://120.25.194.125:2110')
    API_KEY = os.environ.get('GPU_API_KEY', 'your-secret-api-key-change-this')

    print("=" * 60)
    print("Multiple Runs Test (Context Cleanup Verification)")
    print("=" * 60)

    client = RemoteGPUClient(SERVER_URL, API_KEY)

    if not client.health_check():
        print("ERROR: Server not available")
        return

    print("\nRunning 5 consecutive SpMV executions...")

    kernel_path = os.path.join(os.path.dirname(__file__), '..', 'kernels', 'simple_spmv.cu')
    kernel_path = os.path.abspath(kernel_path)

    # Compile once
    print("\nCompiling kernel...")
    cubin = client.compile_cuda_kernel(kernel_path)

    for i in range(5):
        print(f"\n--- Run {i+1}/5 ---")
        matrix_data = generate_small_matrix(100)

        try:
            result = client.execute_spmv_csr(cubin, matrix_data)
            print(f"✓ Success: {result['execution_time_ms']:.3f} ms, {result['gflops']:.2f} GFLOPS")
        except Exception as e:
            print(f"✗ Failed: {e}")
            return

    print("\n" + "=" * 60)
    print("All runs completed successfully!")
    print("No context cleanup errors detected.")
    print("=" * 60)


if __name__ == '__main__':
    main()
