#!/usr/bin/env python3
"""
Test connection to GPU server and check GPU architecture
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from remote_client import RemoteGPUClient

# Load .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

def main():
    SERVER_URL = os.environ.get('GPU_SERVER_URL', 'http://120.25.194.125:2110')
    API_KEY = os.environ.get('GPU_API_KEY', 'your-secret-api-key-change-this')

    print("=" * 60)
    print("GPU Server Connection Test")
    print("=" * 60)

    print(f"\nServer URL: {SERVER_URL}")
    print(f"API Key: {API_KEY[:10]}..." if len(API_KEY) > 10 else f"API Key: {API_KEY}")

    # Initialize client
    client = RemoteGPUClient(SERVER_URL, API_KEY)

    # Health check
    print("\n1. Testing health endpoint...")
    if client.health_check():
        print("   ✓ Server is healthy")
    else:
        print("   ✗ Server is not responding")
        return

    # Get GPU info
    print("\n2. Getting GPU information...")
    gpu_info = client.get_gpu_info()

    if gpu_info:
        print("   ✓ GPU Info retrieved successfully:")
        print(f"   - Device: {gpu_info['device_name']}")
        print(f"   - Compute Capability: {gpu_info['compute_capability']}")
        print(f"   - Total Memory: {gpu_info['total_memory_mb']} MB")
        print(f"   - Multiprocessors: {gpu_info['multiprocessor_count']}")

        # Get architecture
        arch = client.get_server_arch()
        if arch:
            print(f"   - Architecture: {arch}")
            print(f"\n   Use this architecture when compiling CUDA kernels")
    else:
        print("   ✗ Failed to get GPU info")
        return

    print("\n" + "=" * 60)
    print("Connection test completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
