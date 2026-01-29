import os

workers = 1
worker_class = 'sync'
worker_connections = 1000
timeout = 120
keepalive = 5
threads = 4
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
accesslog = '-'
errorlog = '-'
loglevel = 'info'