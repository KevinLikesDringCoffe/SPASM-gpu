# Remote GPU Execution Service

A client-server system for executing CUDA kernels on a remote GPU server. This is designed for scenarios where you have a development machine without GPU and a separate GPU server.

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
   # Create .env file or set environment variables
   export GPU_SERVICE_HOST=0.0.0.0
   export GPU_SERVICE_PORT=5000
   export GPU_SERVICE_API_KEY=your-secret-api-key-here
   ```

4. **Start the service**:
   ```bash
   python gpu_service.py
   ```

   For production, use gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 gpu_service:app
   ```

## Usage

### Quick Start Example

On the coding machine:

```bash
cd remote_gpu/client

# Set server URL and API key
export GPU_SERVER_URL=http://your-gpu-server:5000
export GPU_API_KEY=your-secret-api-key-here

# Run example
python example_spmv.py
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
print(f"Execution time: {exec_time:.3f} ms")
```

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
