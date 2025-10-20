import requests
import base64
import subprocess
import os
from typing import Dict, Any, Optional
import numpy as np


class RemoteGPUClient:
    """Client for remote GPU execution service"""

    def __init__(self, server_url: str, api_key: Optional[str] = None):
        """
        Initialize remote GPU client

        Args:
            server_url: URL of GPU server (e.g., "http://gpu-server:5000")
            api_key: API key for authentication (optional)
        """
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update({'X-API-Key': self.api_key})

    def health_check(self) -> bool:
        """Check if server is healthy"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU device information"""
        try:
            response = self.session.get(f"{self.server_url}/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('info')
            return None
        except Exception as e:
            print(f"Failed to get GPU info: {e}")
            return None

    def get_server_arch(self) -> Optional[str]:
        """
        Get server GPU architecture from compute capability

        Returns:
            Architecture string like "sm_75" or None if failed
        """
        info = self.get_gpu_info()
        if info and 'compute_capability' in info:
            cc = info['compute_capability']
            # Convert [7, 5] to "sm_75"
            if isinstance(cc, list) and len(cc) == 2:
                return f"sm_{cc[0]}{cc[1]}"
        return None

    def compile_cuda_kernel(self, cu_file: str, output_cubin: Optional[str] = None,
                           arch: Optional[str] = None) -> str:
        """
        Compile CUDA kernel to cubin format

        Args:
            cu_file: Path to .cu source file
            output_cubin: Output cubin file path (optional, default: same name as .cu)
            arch: GPU architecture (optional, auto-detected from server if None)

        Returns:
            Path to compiled cubin file
        """
        if not os.path.exists(cu_file):
            raise FileNotFoundError(f"CUDA source file not found: {cu_file}")

        # Auto-detect architecture from server if not specified
        if arch is None:
            arch = self.get_server_arch()
            if arch:
                print(f"Auto-detected server GPU architecture: {arch}")
            else:
                print("Warning: Could not auto-detect GPU architecture, using sm_70")
                arch = "sm_70"

        if output_cubin is None:
            output_cubin = cu_file.replace('.cu', '.cubin')

        cmd = [
            'nvcc',
            '-cubin',
            '-O3',
            f'-arch={arch}',
            cu_file,
            '-o', output_cubin
        ]

        print(f"Compiling {cu_file} -> {output_cubin} (arch={arch})")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"CUDA compilation failed:\n{result.stderr}")

        print(f"Compilation successful: {output_cubin}")
        return output_cubin

    def execute_spmv_csr(self, cubin_file: str, matrix_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute CSR SpMV on remote GPU

        Args:
            cubin_file: Path to compiled cubin file
            matrix_data: Dictionary containing:
                - num_rows: number of rows
                - num_cols: number of columns
                - nnz: number of non-zeros
                - values: CSR values array (list or numpy array)
                - col_indices: CSR column indices (list or numpy array)
                - row_ptr: CSR row pointer (list or numpy array)
                - x: input vector (list or numpy array)

        Returns:
            Dictionary containing:
                - y: output vector
                - execution_time_ms: kernel execution time
                - transfer_time_ms: data transfer time
                - total_time_ms: total time
                - num_rows, num_cols, nnz: matrix dimensions
        """
        # Read cubin file
        with open(cubin_file, 'rb') as f:
            kernel_binary = f.read()

        # Encode kernel as base64
        kernel_b64 = base64.b64encode(kernel_binary).decode('ascii')

        # Convert numpy arrays to lists if needed
        matrix_data_serializable = {}
        for key, value in matrix_data.items():
            if isinstance(value, np.ndarray):
                matrix_data_serializable[key] = value.tolist()
            elif isinstance(value, (list, int, float)):
                matrix_data_serializable[key] = value
            else:
                matrix_data_serializable[key] = value

        # Prepare request
        request_data = {
            'kernel': kernel_b64,
            'matrix_data': matrix_data_serializable
        }

        # Send request
        print(f"Sending SpMV request to {self.server_url}/execute_spmv")
        print(f"Matrix: {matrix_data['num_rows']}x{matrix_data['num_cols']}, nnz={matrix_data['nnz']}")

        response = self.session.post(
            f"{self.server_url}/execute_spmv",
            json=request_data,
            timeout=300  # 5 minutes timeout
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.text}")

        result = response.json()

        if not result.get('success'):
            raise RuntimeError(f"Execution failed: {result.get('error')}")

        return result['result']

    def compile_and_execute_spmv(self, cu_file: str, matrix_data: Dict[str, Any],
                                 arch: Optional[str] = None) -> Dict[str, Any]:
        """
        Compile CUDA kernel and execute SpMV in one call

        Args:
            cu_file: Path to .cu source file
            matrix_data: Matrix data dictionary (see execute_spmv_csr)
            arch: GPU architecture (optional, auto-detected from server if None)

        Returns:
            Execution result dictionary
        """
        # Compile kernel (auto-detects arch if not specified)
        cubin_file = self.compile_cuda_kernel(cu_file, arch=arch)

        # Execute on remote GPU
        result = self.execute_spmv_csr(cubin_file, matrix_data)

        return result
