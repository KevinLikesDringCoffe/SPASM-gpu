#!/usr/bin/env python3
"""
Simple command-line client for SpMV execution with MTX files

Usage:
    python spmv_mtx.py <mtx_file>
    python spmv_mtx.py <mtx_file> --kernel <kernel.cu>
"""

import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from remote_client import RemoteGPUClient

# Load .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)


def main():
    parser = argparse.ArgumentParser(description='Execute SpMV on remote GPU using MTX file')
    parser.add_argument('mtx_file', help='Path to MTX file')
    parser.add_argument('--kernel', default='../kernels/simple_spmv.cu',
                       help='Path to CUDA kernel (default: ../kernels/simple_spmv.cu)')
    parser.add_argument('--method', default='custom', choices=['custom', 'cusparse'],
                       help='Execution method: custom kernel or cuSPARSE baseline (default: custom)')
    parser.add_argument('--runs', type=int, default=100,
                       help='Number of runs to average (default: 100)')
    parser.add_argument('--server', default=None,
                       help='GPU server URL (default: from .env)')
    parser.add_argument('--api-key', default=None,
                       help='API key (default: from .env)')

    args = parser.parse_args()

    # Check MTX file exists
    if not os.path.exists(args.mtx_file):
        print(f"Error: MTX file not found: {args.mtx_file}")
        sys.exit(1)

    # Get server configuration
    server_url = args.server or os.environ.get('GPU_SERVER_URL', 'http://120.25.194.125:2110')
    api_key = args.api_key or os.environ.get('GPU_API_KEY', 'your-secret-api-key-change-this')

    # Get kernel path (only needed for custom method)
    kernel_path = None
    if args.method == 'custom':
        kernel_path = os.path.abspath(args.kernel)
        if not os.path.exists(kernel_path):
            print(f"Error: Kernel file not found: {kernel_path}")
            sys.exit(1)

    # Initialize client
    print(f"Connecting to {server_url}...")
    client = RemoteGPUClient(server_url, api_key)

    # Check server health
    if not client.health_check():
        print("Error: Server not responding")
        sys.exit(1)

    # Execute SpMV
    print(f"Processing MTX file: {args.mtx_file}")
    print(f"Method: {args.method}, Runs: {args.runs}")
    try:
        result = client.upload_and_execute_spmv(
            cu_file=kernel_path,
            mtx_file=args.mtx_file,
            method=args.method,
            num_runs=args.runs
        )

        # Display results
        print("\nResults:")
        print(f"  Method:          {result.get('method', args.method)}")
        print(f"  Matrix:          {result['num_rows']}x{result['num_cols']}, nnz={result['nnz']}")
        print(f"  Runs:            {result['num_runs']}")
        print(f"  Avg kernel time: {result['execution_time_ms']:.3f} ms")
        if result['num_runs'] > 1:
            print(f"  Std deviation:   {result['execution_time_std_ms']:.3f} ms")
        print(f"  Throughput:      {result['gflops']:.2f} GFLOPS")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
