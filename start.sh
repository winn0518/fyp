#!/bin/bash
# start.sh

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p templates uploads

# Start the application with gunicorn for production
gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 2 app:app