#!/bin/bash
# Entrypoint script for krknctl Lightspeed FAISS-only RAG service

set -e

echo "Starting krknctl Lightspeed FAISS service..."

# Set environment variables
export PYTHONPATH="/app:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export RELEVANCE_THRESHOLD="${RELEVANCE_THRESHOLD:-0.4}"

echo "Configuration:"
echo "  RELEVANCE_THRESHOLD: $RELEVANCE_THRESHOLD"
echo "  PERSIST_DIR: ${PERSIST_DIR:-/app/faiss_index}"
echo "  FORCE_REINDEX: ${FORCE_REINDEX:-false}"

# Change to app directory
cd /app

# Check if FORCE_REINDEX is enabled
if [ "${FORCE_REINDEX}" = "true" ]; then
    echo "FORCE_REINDEX=true - Rebuilding FAISS index..."
    python3 /app/build_index.py
    echo "Index rebuild completed"
elif [ ! -f "/app/faiss_index/index.faiss" ]; then
    echo "WARNING: FAISS index not found at /app/faiss_index/index.faiss"
    echo "Building index now..."
    python3 /app/build_index.py
    echo "Index build completed"
else
    echo "Using existing FAISS index"
    ls -lh /app/faiss_index/index.faiss
fi

# Verify FastAPI app exists
if [ ! -f "/app/fastapi_app.py" ]; then
    echo "ERROR: FastAPI application not found at /app/fastapi_app.py"
    echo "Available files in /app:"
    ls -la /app/
    exit 1
fi

# Start the FastAPI server
echo "Starting FAISS FastAPI service on port 8080..."
echo "Endpoint: http://0.0.0.0:8080/v1/chat/completions"
exec python3 fastapi_app.py