import pycuda.driver as cuda
import numpy as np
import time
from typing import Dict, Any
import threading
import weakref

# Try to import cupy for cuSPARSE support
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusparse
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class CUDAExecutor:
    """Execute CUDA kernels with timing information"""

    _local = threading.local()
    _initialized = False
    _lock = threading.Lock()
    _contexts = []  # Keep track of all contexts for cleanup

    def __init__(self):
        self._ensure_cuda_initialized()

    @classmethod
    def _ensure_cuda_initialized(cls):
        """Ensure CUDA is initialized once globally"""
        with cls._lock:
            if not cls._initialized:
                cuda.init()
                cls._initialized = True

    @classmethod
    def _get_or_create_context(cls):
        """Get or create CUDA context for current thread"""
        if not hasattr(cls._local, 'context') or cls._local.context is None:
            cls._ensure_cuda_initialized()
            device = cuda.Device(0)
            # Create context (this automatically pushes it to the stack)
            ctx = device.make_context()
            # Pop it immediately so we can manage it manually
            ctx.pop()
            # Store context reference for cleanup
            with cls._lock:
                cls._contexts.append(weakref.ref(ctx))
            cls._local.context = ctx
            cls._local.device = device
            cls._local.context_pushed = False
        return cls._local.context

    @classmethod
    def cleanup_all_contexts(cls):
        """Clean up all CUDA contexts - call this on shutdown"""
        with cls._lock:
            for ctx_ref in cls._contexts:
                ctx = ctx_ref()
                if ctx is not None:
                    try:
                        # Pop all contexts from stack
                        while True:
                            try:
                                cuda.Context.pop()
                            except cuda.LogicError:
                                break
                        ctx.detach()
                    except:
                        pass
            cls._contexts.clear()

    def execute_spmv_csr(self, kernel_data: bytes, matrix_data: Dict[str, Any], num_runs: int = 1) -> Dict[str, Any]:
        """
        Execute CSR SpMV kernel

        Args:
            kernel_data: Compiled CUDA kernel (cubin format)
            matrix_data: Dictionary containing:
                - num_rows: number of rows
                - num_cols: number of columns
                - nnz: number of non-zeros
                - values: CSR values array
                - col_indices: CSR column indices
                - row_ptr: CSR row pointer
                - x: input vector
            num_runs: Number of runs to average (default: 1)

        Returns:
            Dictionary containing:
                - y: output vector
                - execution_time_ms: average kernel execution time in milliseconds
                - transfer_time_ms: data transfer time in milliseconds
                - total_time_ms: total time including transfers
                - num_runs: number of runs performed
        """
        # Get or create CUDA context for this thread
        ctx = self._get_or_create_context()

        # Push context to make it active
        ctx.push()

        try:
            start_total = time.perf_counter()

            # Extract matrix data
            num_rows = matrix_data['num_rows']
            num_cols = matrix_data['num_cols']
            nnz = matrix_data['nnz']

            values = np.array(matrix_data['values'], dtype=np.float32)
            col_indices = np.array(matrix_data['col_indices'], dtype=np.int32)
            row_ptr = np.array(matrix_data['row_ptr'], dtype=np.int32)
            x = np.array(matrix_data['x'], dtype=np.float32)

            # Initialize output vector
            y = np.zeros(num_rows, dtype=np.float32)

            # Load kernel module
            module = cuda.module_from_buffer(kernel_data)
            kernel_func = module.get_function("csr_spmv_kernel")

            # Transfer data to GPU
            start_transfer = time.perf_counter()

            values_gpu = cuda.mem_alloc(values.nbytes)
            col_indices_gpu = cuda.mem_alloc(col_indices.nbytes)
            row_ptr_gpu = cuda.mem_alloc(row_ptr.nbytes)
            x_gpu = cuda.mem_alloc(x.nbytes)
            y_gpu = cuda.mem_alloc(y.nbytes)

            cuda.memcpy_htod(values_gpu, values)
            cuda.memcpy_htod(col_indices_gpu, col_indices)
            cuda.memcpy_htod(row_ptr_gpu, row_ptr)
            cuda.memcpy_htod(x_gpu, x)
            cuda.memcpy_htod(y_gpu, y)

            end_transfer = time.perf_counter()
            transfer_time_h2d = (end_transfer - start_transfer) * 1000

            # Execute kernel
            block_size = 256
            grid_size = (num_rows + block_size - 1) // block_size

            # Warm-up run
            kernel_func(
                np.int32(num_rows),
                values_gpu,
                col_indices_gpu,
                row_ptr_gpu,
                x_gpu,
                y_gpu,
                block=(block_size, 1, 1),
                grid=(grid_size, 1)
            )
            ctx.synchronize()

            # Timed execution - multiple runs
            execution_times = []
            for _ in range(num_runs):
                start_kernel = time.perf_counter()

                kernel_func(
                    np.int32(num_rows),
                    values_gpu,
                    col_indices_gpu,
                    row_ptr_gpu,
                    x_gpu,
                    y_gpu,
                    block=(block_size, 1, 1),
                    grid=(grid_size, 1)
                )
                ctx.synchronize()

                end_kernel = time.perf_counter()
                execution_times.append((end_kernel - start_kernel) * 1000)

            # Calculate average execution time
            execution_time = np.mean(execution_times)
            execution_time_std = np.std(execution_times) if num_runs > 1 else 0.0

            # Transfer result back
            start_transfer_back = time.perf_counter()
            cuda.memcpy_dtoh(y, y_gpu)
            end_transfer_back = time.perf_counter()
            transfer_time_d2h = (end_transfer_back - start_transfer_back) * 1000

            # Free GPU memory
            values_gpu.free()
            col_indices_gpu.free()
            row_ptr_gpu.free()
            x_gpu.free()
            y_gpu.free()

            end_total = time.perf_counter()
            total_time = (end_total - start_total) * 1000

            # Calculate throughput
            # SpMV: 2 * nnz FLOPs (one multiply, one add per non-zero)
            flops = 2 * nnz
            gflops = (flops / (execution_time / 1000.0)) / 1e9  # GFLOPS

            return {
                'y': y.tolist(),
                'execution_time_ms': execution_time,
                'execution_time_std_ms': execution_time_std,
                'transfer_h2d_ms': transfer_time_h2d,
                'transfer_d2h_ms': transfer_time_d2h,
                'transfer_time_ms': transfer_time_h2d + transfer_time_d2h,
                'total_time_ms': total_time,
                'num_rows': num_rows,
                'num_cols': num_cols,
                'nnz': nnz,
                'gflops': gflops,
                'num_runs': num_runs
            }
        except Exception:
            raise
        finally:
            # Always pop context after execution
            ctx.pop()

    def execute_spmv_cusparse(self, matrix_data: Dict[str, Any], num_runs: int = 1) -> Dict[str, Any]:
        """
        Execute CSR SpMV using cuSPARSE (baseline)

        Args:
            matrix_data: Dictionary containing CSR format data
            num_runs: Number of runs to average (default: 1)

        Returns:
            Dictionary with execution results
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available. Install with: pip install cupy-cuda11x or cupy-cuda12x")

        start_total = time.perf_counter()

        # Extract matrix data
        num_rows = matrix_data['num_rows']
        num_cols = matrix_data['num_cols']
        nnz = matrix_data['nnz']

        values = np.array(matrix_data['values'], dtype=np.float32)
        col_indices = np.array(matrix_data['col_indices'], dtype=np.int32)
        row_ptr = np.array(matrix_data['row_ptr'], dtype=np.int32)
        x = np.array(matrix_data['x'], dtype=np.float32)

        # Transfer data to GPU
        start_transfer = time.perf_counter()

        values_gpu = cp.array(values)
        col_indices_gpu = cp.array(col_indices)
        row_ptr_gpu = cp.array(row_ptr)
        x_gpu = cp.array(x)

        end_transfer = time.perf_counter()
        transfer_time_h2d = (end_transfer - start_transfer) * 1000

        # Create CSR matrix using cuSPARSE
        csr_matrix = cusparse.csr_matrix(
            (values_gpu, col_indices_gpu, row_ptr_gpu),
            shape=(num_rows, num_cols)
        )

        # Warm-up run
        y_gpu = csr_matrix.dot(x_gpu)
        cp.cuda.Stream.null.synchronize()

        # Timed execution - multiple runs
        execution_times = []
        for _ in range(num_runs):
            start_kernel = time.perf_counter()

            y_gpu = csr_matrix.dot(x_gpu)
            cp.cuda.Stream.null.synchronize()

            end_kernel = time.perf_counter()
            execution_times.append((end_kernel - start_kernel) * 1000)

        # Calculate average execution time
        execution_time = np.mean(execution_times)
        execution_time_std = np.std(execution_times) if num_runs > 1 else 0.0

        # Transfer result back
        start_transfer_back = time.perf_counter()
        y = cp.asnumpy(y_gpu)
        end_transfer_back = time.perf_counter()
        transfer_time_d2h = (end_transfer_back - start_transfer_back) * 1000

        end_total = time.perf_counter()
        total_time = (end_total - start_total) * 1000

        # Calculate throughput
        flops = 2 * nnz
        gflops = (flops / (execution_time / 1000.0)) / 1e9

        return {
            'y': y.tolist(),
            'execution_time_ms': execution_time,
            'execution_time_std_ms': execution_time_std,
            'transfer_h2d_ms': transfer_time_h2d,
            'transfer_d2h_ms': transfer_time_d2h,
            'transfer_time_ms': transfer_time_h2d + transfer_time_d2h,
            'total_time_ms': total_time,
            'num_rows': num_rows,
            'num_cols': num_cols,
            'nnz': nnz,
            'gflops': gflops,
            'method': 'cuSPARSE',
            'num_runs': num_runs
        }
