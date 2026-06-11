import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

bind = os.getenv("RSM_GUNICORN_BIND", "0.0.0.0:8000")
workers = int(os.getenv("RSM_GUNICORN_WORKERS", "2"))
worker_class = "uvicorn_worker.UvicornWorker"

# RSM inspections are CPU-heavy and may take longer than ordinary API calls.
timeout = int(os.getenv("RSM_GUNICORN_TIMEOUT", "300"))
graceful_timeout = int(os.getenv("RSM_GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("RSM_GUNICORN_KEEPALIVE", "5"))

# Do not preload the app: each worker must own its ProcessPoolExecutor lifecycle.
preload_app = False

certfile = os.getenv("RSM_SSL_CERTFILE", str(PROJECT_ROOT / "localhost+1.pem"))
keyfile = os.getenv("RSM_SSL_KEYFILE", str(PROJECT_ROOT / "localhost+1-key.pem"))

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("RSM_GUNICORN_LOG_LEVEL", "info")
capture_output = True
