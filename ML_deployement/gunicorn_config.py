import multiprocessing

bind = "0.0.0.0:8000"
workers = min(multiprocessing.cpu_count() * 2, 8)  # Limit max workers
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
errorlog = "-"
accesslog = "-"
loglevel = "info"

# TensorFlow optimization
worker_tmp_dir = "/dev/shm"
worker_class = "uvicorn.workers.UvicornWorker"
