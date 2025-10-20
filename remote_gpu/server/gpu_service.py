from flask import Flask, request, jsonify
import base64
import msgpack
from cuda_executor import CUDAExecutor
from mtx_utils import MTXConverter
import logging
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

executor = CUDAExecutor()
mtx_converter = MTXConverter()


def verify_api_key():
    """Verify API key from request headers"""
    if not Config.REQUIRE_AUTH:
        return True

    api_key = request.headers.get('X-API-Key')
    if api_key != Config.API_KEY:
        return False
    return True


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'GPU Execution Service',
        'version': '1.0.0'
    })


@app.route('/execute_spmv', methods=['POST'])
def execute_spmv():
    """
    Execute SpMV kernel on GPU

    Request format (JSON):
    {
        "kernel": "base64-encoded cubin file",
        "matrix_data": {
            "num_rows": int,
            "num_cols": int,
            "nnz": int,
            "values": [float],
            "col_indices": [int],
            "row_ptr": [int],
            "x": [float]
        },
        "format": "json" or "msgpack" (optional, default: json)
    }

    Response format:
    {
        "success": bool,
        "result": {
            "y": [float],
            "execution_time_ms": float,
            "transfer_time_ms": float,
            "total_time_ms": float,
            "num_rows": int,
            "num_cols": int,
            "nnz": int
        },
        "error": str (if success=false)
    }
    """
    # Verify authentication
    if not verify_api_key():
        return jsonify({
            'success': False,
            'error': 'Unauthorized: Invalid API key'
        }), 401

    try:
        # Parse request
        if request.content_type == 'application/msgpack':
            data = msgpack.unpackb(request.data, raw=False)
        else:
            data = request.json

        if not data:
            return jsonify({
                'success': False,
                'error': 'Invalid request: empty data'
            }), 400

        # Extract kernel and matrix data
        kernel_b64 = data.get('kernel')
        matrix_data = data.get('matrix_data')

        if not kernel_b64 or not matrix_data:
            return jsonify({
                'success': False,
                'error': 'Missing kernel or matrix_data'
            }), 400

        # Decode kernel
        kernel_binary = base64.b64decode(kernel_b64)

        # Get number of runs (default: 1)
        num_runs = data.get('num_runs', 1)

        logger.info(f"Executing SpMV: rows={matrix_data['num_rows']}, "
                   f"cols={matrix_data['num_cols']}, nnz={matrix_data['nnz']}, runs={num_runs}")

        # Execute on GPU
        result = executor.execute_spmv_csr(kernel_binary, matrix_data, num_runs=num_runs)

        logger.info(f"Execution completed: {result['execution_time_ms']:.3f} ms, "
                   f"{result['gflops']:.2f} GFLOPS")

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        logger.error(f"Error executing SpMV: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/info', methods=['GET'])
def info():
    """Get GPU device information"""
    try:
        import pycuda.driver as cuda

        # Initialize CUDA driver if not already done
        cuda.init()
        device = cuda.Device(0)
        attrs = device.get_attributes()

        info = {
            'device_name': device.name(),
            'compute_capability': device.compute_capability(),
            'total_memory_mb': device.total_memory() // (1024 * 1024),
            'multiprocessor_count': attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT],
        }

        return jsonify({
            'success': True,
            'info': info
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/upload_mtx', methods=['POST'])
def upload_mtx():
    """
    Upload MTX file to server

    Request format (multipart/form-data or JSON):
    - If multipart: file field with MTX file
    - If JSON: {"filename": str, "content": base64-encoded content}

    Response:
    {
        "success": bool,
        "filename": str,
        "info": {
            "num_rows": int,
            "num_cols": int,
            "nnz": int,
            "exists": bool
        }
    }
    """
    if not verify_api_key():
        return jsonify({
            'success': False,
            'error': 'Unauthorized: Invalid API key'
        }), 401

    try:
        # Handle multipart form data
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename
            content = file.read()
        # Handle JSON with base64 content
        elif request.json:
            data = request.json
            filename = data.get('filename')
            content_b64 = data.get('content')
            content = base64.b64decode(content_b64)
        else:
            return jsonify({
                'success': False,
                'error': 'No file or data provided'
            }), 400

        # Save MTX file
        mtx_converter.save_mtx_file(filename, content)

        # Get matrix info
        info = mtx_converter.get_matrix_info(filename)

        logger.info(f"Uploaded MTX file: {filename} ({info['num_rows']}x{info['num_cols']}, nnz={info['nnz']})")

        return jsonify({
            'success': True,
            'filename': filename,
            'info': info
        })

    except Exception as e:
        logger.error(f"Error uploading MTX: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/check_mtx/<filename>', methods=['GET'])
def check_mtx(filename):
    """Check if MTX file exists on server"""
    if not verify_api_key():
        return jsonify({
            'success': False,
            'error': 'Unauthorized: Invalid API key'
        }), 401

    try:
        info = mtx_converter.get_matrix_info(filename)

        if info:
            return jsonify({
                'success': True,
                'exists': True,
                'info': info
            })
        else:
            return jsonify({
                'success': True,
                'exists': False
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/list_mtx', methods=['GET'])
def list_mtx():
    """List all cached MTX files"""
    if not verify_api_key():
        return jsonify({
            'success': False,
            'error': 'Unauthorized: Invalid API key'
        }), 401

    try:
        matrices = mtx_converter.list_cached_matrices()
        return jsonify({
            'success': True,
            'matrices': matrices
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/execute_spmv_mtx', methods=['POST'])
def execute_spmv_mtx():
    """
    Execute SpMV using MTX file

    Request format (JSON):
    {
        "kernel": "base64-encoded cubin file" (optional if method='cusparse'),
        "mtx_filename": "matrix.mtx",
        "x": [float array - input vector],
        "method": "custom" or "cusparse" (default: "custom")
    }

    Response: Same as execute_spmv
    """
    if not verify_api_key():
        return jsonify({
            'success': False,
            'error': 'Unauthorized: Invalid API key'
        }), 401

    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'Invalid request: empty data'
            }), 400

        mtx_filename = data.get('mtx_filename')
        x = data.get('x')
        method = data.get('method', 'custom')
        num_runs = data.get('num_runs', 1)

        if not mtx_filename or not x:
            return jsonify({
                'success': False,
                'error': 'Missing mtx_filename or x'
            }), 400

        # Convert MTX to CSR (uses cache if available)
        logger.info(f"Loading MTX file: {mtx_filename}")
        csr_data = mtx_converter.convert_mtx_to_csr(mtx_filename)

        # Add input vector
        csr_data['x'] = x

        logger.info(f"Executing SpMV ({method}): rows={csr_data['num_rows']}, "
                   f"cols={csr_data['num_cols']}, nnz={csr_data['nnz']}, runs={num_runs}")

        # Execute on GPU based on method
        if method == 'cusparse':
            result = executor.execute_spmv_cusparse(csr_data, num_runs=num_runs)
        else:
            # Custom kernel method
            kernel_b64 = data.get('kernel')
            if not kernel_b64:
                return jsonify({
                    'success': False,
                    'error': 'Missing kernel for custom method'
                }), 400

            kernel_binary = base64.b64decode(kernel_b64)
            result = executor.execute_spmv_csr(kernel_binary, csr_data, num_runs=num_runs)
            result['method'] = 'custom'

        logger.info(f"Execution completed ({result.get('method', method)}): "
                   f"{result['execution_time_ms']:.3f} ms, {result['gflops']:.2f} GFLOPS")

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        logger.error(f"Error executing SpMV with MTX: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    import atexit

    # Register cleanup function
    atexit.register(CUDAExecutor.cleanup_all_contexts)

    logger.info("Starting GPU Execution Service")
    logger.info(f"Authentication: {'enabled' if Config.REQUIRE_AUTH else 'disabled'}")

    try:
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG
        )
    finally:
        # Clean up CUDA contexts
        CUDAExecutor.cleanup_all_contexts()
