from flask import Flask, request, jsonify
import base64
import msgpack
from cuda_executor import CUDAExecutor
import logging
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

executor = CUDAExecutor()


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

        logger.info(f"Executing SpMV: rows={matrix_data['num_rows']}, "
                   f"cols={matrix_data['num_cols']}, nnz={matrix_data['nnz']}")

        # Execute on GPU
        result = executor.execute_spmv_csr(kernel_binary, matrix_data)

        logger.info(f"Execution completed: {result['execution_time_ms']:.3f} ms")

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


if __name__ == '__main__':
    logger.info("Starting GPU Execution Service")
    logger.info(f"Authentication: {'enabled' if Config.REQUIRE_AUTH else 'disabled'}")

    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
