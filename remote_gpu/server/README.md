# GPU Execution Server

Flask-based REST API service for executing CUDA kernels on GPU.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import pycuda.autoinit; print('GPU OK')"
```

### Configuration

Environment variables (optional):
```bash
export GPU_SERVICE_HOST=0.0.0.0        # Bind address
export GPU_SERVICE_PORT=5000           # Port
export GPU_SERVICE_API_KEY=secret-key  # API key for authentication
export GPU_SERVICE_REQUIRE_AUTH=True   # Enable/disable auth
export GPU_SERVICE_DEBUG=False         # Debug mode
```

Or edit `config.py` directly.

### Running the Service

**Development:**
```bash
python gpu_service.py
```

**Production (with gunicorn):**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 300 gpu_service:app
```

**As a systemd service:**

Create `/etc/systemd/system/gpu-service.service`:
```ini
[Unit]
Description=GPU Execution Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/remote_gpu/server
Environment="GPU_SERVICE_API_KEY=your-secret-key"
ExecStart=/usr/bin/python3 gpu_service.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gpu-service
sudo systemctl start gpu-service
sudo systemctl status gpu-service
```

## Testing

```bash
# Health check
curl http://localhost:5000/health

# GPU info
curl http://localhost:5000/info

# Test SpMV execution (from client)
cd ../client
python example_spmv.py
```

## Logs

Check logs for debugging:
```bash
# If running with systemd
sudo journalctl -u gpu-service -f

# If running manually, logs go to stdout
```

## Performance Tuning

### Gunicorn Workers
- Use 1-2 workers per GPU
- Each worker uses one GPU context
- More workers = more concurrent requests

### Request Timeout
- Default: 300 seconds (5 minutes)
- Adjust based on kernel execution time
- Set via `--timeout` flag in gunicorn

### Memory Limits
- Default max request size: 500MB
- Adjust `MAX_CONTENT_LENGTH` in config.py

## Security

1. **Always change the default API key**
2. Use HTTPS in production (nginx reverse proxy)
3. Restrict access via firewall
4. Monitor logs for unauthorized access attempts

Example nginx configuration:
```nginx
server {
    listen 443 ssl;
    server_name gpu-server.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 500M;
    }
}
```
