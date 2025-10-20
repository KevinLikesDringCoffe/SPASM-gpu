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
            timeout=600  # 10 minutes timeout for large matrices
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

    def check_mtx_exists(self, mtx_filename: str) -> Dict[str, Any]:
        """
        Check if MTX file exists on server

        Args:
            mtx_filename: Name of MTX file

        Returns:
            Dictionary with exists flag and matrix info if exists
        """
        response = self.session.get(
            f"{self.server_url}/check_mtx/{mtx_filename}",
            timeout=10
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.text}")

        result = response.json()
        if not result.get('success'):
            raise RuntimeError(f"Check failed: {result.get('error')}")

        return result

    def upload_mtx_file(self, mtx_file_path: str) -> Dict[str, Any]:
        """
        Upload MTX file to server

        Args:
            mtx_file_path: Path to MTX file

        Returns:
            Dictionary with upload result and matrix info
        """
        if not os.path.exists(mtx_file_path):
            raise FileNotFoundError(f"MTX file not found: {mtx_file_path}")

        filename = os.path.basename(mtx_file_path)

        # Read and encode file
        with open(mtx_file_path, 'rb') as f:
            content = f.read()

        content_b64 = base64.b64encode(content).decode('ascii')

        # Send to server
        print(f"Uploading MTX file: {filename}")
        response = self.session.post(
            f"{self.server_url}/upload_mtx",
            json={
                'filename': filename,
                'content': content_b64
            },
            timeout=600  # 10 minutes for large files (up to 2GB)
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.text}")

        result = response.json()
        if not result.get('success'):
            raise RuntimeError(f"Upload failed: {result.get('error')}")

        print(f"Upload successful: {result['info']['num_rows']}x{result['info']['num_cols']}, "
              f"nnz={result['info']['nnz']}")

        return result

    def list_mtx_files(self) -> list:
        """
        List all cached MTX files on server

        Returns:
            List of matrix information dictionaries
        """
        response = self.session.get(
            f"{self.server_url}/list_mtx",
            timeout=10
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.text}")

        result = response.json()
        if not result.get('success'):
            raise RuntimeError(f"List failed: {result.get('error')}")

        return result.get('matrices', [])

    def execute_spmv_mtx(self, cubin_file: Optional[str], mtx_filename: str,
                        x: np.ndarray, method: str = 'custom') -> Dict[str, Any]:
        """
        Execute SpMV using MTX file on server

        Args:
            cubin_file: Path to compiled cubin file (None for cusparse method)
            mtx_filename: Name of MTX file on server
            x: Input vector (numpy array)
            method: 'custom' (default) or 'cusparse' for baseline

        Returns:
            Execution result dictionary
        """
        # Prepare request data
        request_data = {
            'mtx_filename': mtx_filename,
            'method': method
        }

        # Add kernel for custom method
        if method == 'custom':
            if cubin_file is None:
                raise ValueError("cubin_file required for custom method")
            with open(cubin_file, 'rb') as f:
                kernel_binary = f.read()
            kernel_b64 = base64.b64encode(kernel_binary).decode('ascii')
            request_data['kernel'] = kernel_b64

        # Convert numpy array to list if needed
        if isinstance(x, np.ndarray):
            x_list = x.tolist()
        else:
            x_list = x

        request_data['x'] = x_list

        # Send request
        print(f"Executing SpMV with MTX file: {mtx_filename} (method: {method})")

        response = self.session.post(
            f"{self.server_url}/execute_spmv_mtx",
            json=request_data,
            timeout=600  # 10 minutes for large matrices
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server error: {response.text}")

        result = response.json()

        if not result.get('success'):
            raise RuntimeError(f"Execution failed: {result.get('error')}")

        return result['result']

    def upload_and_execute_spmv(self, cu_file: Optional[str], mtx_file: str,
                                x: Optional[np.ndarray] = None,
                                arch: Optional[str] = None,
                                method: str = 'custom') -> Dict[str, Any]:
        """
        Complete workflow: upload MTX if needed, compile kernel, and execute SpMV

        Args:
            cu_file: Path to CUDA kernel source (None for cusparse method)
            mtx_file: Path to MTX file
            x: Input vector (optional, will generate random if None)
            arch: GPU architecture (optional, auto-detected if None)
            method: 'custom' (default) or 'cusparse' for baseline

        Returns:
            Execution result dictionary
        """
        mtx_filename = os.path.basename(mtx_file)

        # Check if MTX exists on server
        check_result = self.check_mtx_exists(mtx_filename)

        if check_result['exists']:
            print(f"MTX file already exists on server: {mtx_filename}")
            info = check_result['info']
        else:
            # Upload MTX file
            upload_result = self.upload_mtx_file(mtx_file)
            info = upload_result['info']

        # Generate input vector if not provided
        if x is None:
            print(f"Generating random input vector (size={info['num_cols']})")
            x = np.random.randn(info['num_cols']).astype(np.float32)

        # Compile kernel if using custom method
        if method == 'custom':
            if cu_file is None:
                raise ValueError("cu_file required for custom method")
            cubin_file = self.compile_cuda_kernel(cu_file, arch=arch)
        else:
            cubin_file = None

        # Execute SpMV
        result = self.execute_spmv_mtx(cubin_file, mtx_filename, x, method=method)

        return result
