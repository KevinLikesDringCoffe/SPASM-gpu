# Remote GPU Execution Service

A client-server system for executing CUDA kernels on a remote GPU server. This is designed for scenarios where you have a development machine without GPU and a separate GPU server.

## Features

- ✅ Remote CUDA kernel execution
- ✅ Automatic GPU architecture detection
- ✅ **MTX file support** with server-side caching
- ✅ Performance metrics (GFLOPS, timing)
- ✅ CSR format SpMV acceleration
- ✅ API key authentication

## Architecture

```
Coding Machine (no GPU)          GPU Server
┌─────────────────────┐         ┌──────────────────────┐
│  Client             │         │  Flask Service       │
│  - Compile CUDA     │ ──HTTP──> - Load cubin        │
│  - Prepare data     │         │  - Execute kernel    │
│  - Send request     │ <─────  │  - Return results    │
└─────────────────────┘         │  - Timing info       │
                                └──────────────────────┘
```

## Directory Structure

```
remote_gpu/
├── server/              # GPU server code
│   ├── gpu_service.py   # Flask REST API
│   ├── cuda_executor.py # CUDA kernel executor
│   ├── config.py        # Configuration
│   └── requirements.txt # Python dependencies
├── client/              # Client code
│   ├── remote_client.py # Client library
│   ├── example_spmv.py  # Example usage
│   └── requirements.txt # Client dependencies
└── kernels/             # CUDA kernel sources
    ├── simple_spmv.cu   # CSR SpMV kernel
    └── Makefile         # Compilation rules
```

## Installation

### On Coding Machine (Client)

1. **Install CUDA Toolkit** (for compilation, no GPU required):
   ```bash
   # Ubuntu/Debian
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-3

   # Add to ~/.bashrc
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

   # Verify
   nvcc --version
   ```

2. **Install Python dependencies**:
   ```bash
   cd remote_gpu/client
   pip install -r requirements.txt
   ```

3. **Configure client**:
   ```bash
   # Copy example env file
   cp .env.example .env

   # Edit .env file with your settings
   # GPU_SERVER_URL=http://120.25.194.125:2110
   # GPU_API_KEY=your-secret-api-key-here
   ```

### On GPU Server

1. **Verify NVIDIA driver and CUDA**:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. **Install Python dependencies**:
   ```bash
   cd remote_gpu/server
   pip install -r requirements.txt
   ```

3. **Configure the service**:
   ```bash
   # Copy example env file
   cp .env.example .env

   # Edit .env file with your settings
   # GPU_SERVICE_HOST=0.0.0.0
   # GPU_SERVICE_PORT=5910
   # GPU_SERVICE_API_KEY=your-secret-api-key-here
   ```

4. **Start the service**:
   ```bash
   python gpu_service.py
   ```

   For production, use gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5910 --timeout 300 gpu_service:app
   ```

## Usage

### Quick Start (Command-Line)

The simplest way to use the service:

```bash
cd remote_gpu/client

# Configure .env file first
cp .env.example .env
# Edit .env with your server URL and API key

# Run SpMV with any MTX file
python spmv_mtx.py your_matrix.mtx
```

**That's it!** The script handles everything automatically:
- ✅ Upload MTX file (only if not cached on server)
- ✅ Compile CUDA kernel with correct GPU architecture
- ✅ Execute SpMV and display results

See [client/README.md](client/README.md) for detailed command-line options.

### Generate Test Matrix

```bash
# Generate a random sparse matrix
python generate_mtx.py test.mtx --size 1000 --density 0.01

# Run SpMV
python spmv_mtx.py test.mtx
```

### Example Scripts (Optional)

Full-featured examples for learning:

```bash
# Example 1: Basic SpMV with generated data
python example_spmv.py

# Example 2: MTX workflow demonstration
python example_mtx.py
```

### Using the Client Library

```python
from remote_client import RemoteGPUClient
import numpy as np

# Initialize client
client = RemoteGPUClient(
    server_url="http://gpu-server:5000",
    api_key="your-api-key"
)

# Check server health
if not client.health_check():
    print("Server not available")
    exit(1)

# Get GPU info
gpu_info = client.get_gpu_info()
print(f"GPU: {gpu_info['device_name']}")

# Prepare matrix data (CSR format)
matrix_data = {
    'num_rows': 1000,
    'num_cols': 1000,
    'nnz': 5000,
    'values': [...],       # float32 array
    'col_indices': [...],  # int32 array
    'row_ptr': [...],      # int32 array
    'x': [...]             # float32 input vector
}

# Compile and execute
result = client.compile_and_execute_spmv(
    'kernels/simple_spmv.cu',
    matrix_data,
    arch='sm_70'  # Match your GPU architecture
)

# Get results
y = result['y']
exec_time = result['execution_time_ms']
gflops = result['gflops']
print(f"Execution time: {exec_time:.3f} ms")
print(f"Throughput: {gflops:.2f} GFLOPS")
```

### Using MTX Files

```python
from remote_client import RemoteGPUClient

client = RemoteGPUClient("http://gpu-server:5000", "api-key")

# One-step: upload MTX (if not cached), compile, and execute
result = client.upload_and_execute_spmv(
    cu_file="kernels/simple_spmv.cu",
    mtx_file="path/to/matrix.mtx"
)

print(f"Throughput: {result['gflops']:.2f} GFLOPS")

# Check what matrices are cached on server
matrices = client.list_mtx_files()
for mat in matrices:
    print(f"{mat['filename']}: {mat['num_rows']}x{mat['num_cols']}")
```

See [MTX_USAGE.md](MTX_USAGE.md) for detailed MTX file documentation.


## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "GPU Execution Service",
  "version": "1.0.0"
}
```

### GET /info
Get GPU device information.

**Response:**
```json
{
  "success": true,
  "info": {
    "device_name": "Tesla V100",
    "compute_capability": [7, 0],
    "total_memory_mb": 16160,
    "multiprocessor_count": 80
  }
}
```

### POST /execute_spmv
Execute SpMV kernel on GPU.

**Request:**
```json
{
  "kernel": "base64-encoded cubin file",
  "matrix_data": {
    "num_rows": 1000,
    "num_cols": 1000,
    "nnz": 5000,
    "values": [...],
    "col_indices": [...],
    "row_ptr": [...],
    "x": [...]
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "y": [...],
    "execution_time_ms": 1.234,
    "transfer_h2d_ms": 0.567,
    "transfer_d2h_ms": 0.123,
    "transfer_time_ms": 0.690,
    "total_time_ms": 2.345,
    "num_rows": 1000,
    "num_cols": 1000,
    "nnz": 5000
  }
}
```

## GPU Architecture Selection

Common NVIDIA GPU architectures:
- `sm_70`: V100, Titan V
- `sm_75`: Turing (RTX 20 series, T4)
- `sm_80`: A100
- `sm_86`: RTX 30 series
- `sm_89`: RTX 40 series (Ada)
- `sm_90`: H100

Check your GPU architecture:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

## Security Notes

1. **Change the default API key** in production
2. Use HTTPS in production (add reverse proxy like nginx)
3. Consider firewall rules to restrict access
4. The service has a 500MB request size limit

## Troubleshooting

### Client compilation errors
- Ensure CUDA Toolkit is installed
- Check `nvcc --version` works
- Verify architecture matches GPU server

### Server connection errors
- Check firewall allows port 5000
- Verify server is running: `curl http://server:5000/health`
- Check API key matches

### PyCUDA errors on server
- Ensure NVIDIA drivers are installed
- Check `nvidia-smi` works
- Verify PyCUDA installation: `python -c "import pycuda.autoinit"`

## Performance Tips

1. Minimize data transfer by batching operations
2. Use appropriate block/grid sizes for your kernel
3. Consider using msgpack for large data transfers
4. Profile with `transfer_time_ms` vs `execution_time_ms`

## Next Steps

- Extend to support more kernel types
- Add support for SPASM format SpMV
- Implement kernel caching
- Add batch processing capabilities
