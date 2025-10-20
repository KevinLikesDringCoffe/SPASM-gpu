import os


class Config:
    """GPU Service Configuration"""

    # Server settings
    HOST = os.environ.get('GPU_SERVICE_HOST', '0.0.0.0')
    PORT = int(os.environ.get('GPU_SERVICE_PORT', 5000))
    DEBUG = os.environ.get('GPU_SERVICE_DEBUG', 'False').lower() == 'true'

    # Authentication
    REQUIRE_AUTH = os.environ.get('GPU_SERVICE_REQUIRE_AUTH', 'True').lower() == 'true'
    API_KEY = os.environ.get('GPU_SERVICE_API_KEY', 'your-secret-api-key-change-this')

    # Request limits
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB max request size
