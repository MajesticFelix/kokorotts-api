# KokoroTTS API - Multi-stage Docker Build
# Supports both CPU and GPU deployments
# Build arguments: 
#   - PYTHON_VERSION (default: 3.12-slim)
#   - TORCH_INDEX_URL (default: https://download.pytorch.org/whl/cpu for CPU builds)

ARG PYTHON_VERSION=3.12-slim
FROM python:${PYTHON_VERSION} as base

# Build arguments
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
ARG BUILD_ENV="production"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    pkg-config \
    # Audio processing
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    espeak-ng \
    espeak-ng-data \
    # System utilities
    curl \
    git \
    wget \
    unzip \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r kokorotts && useradd -r -g kokorotts -s /bin/bash kokorotts

# Set work directory
WORKDIR /app

# ================================
# Stage 2: Python Dependencies
# ================================
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Install PyTorch with specified index URL (CPU or GPU)
    pip install --no-cache-dir torch torchaudio --index-url ${TORCH_INDEX_URL} && \
    # Install other dependencies
    pip install --no-cache-dir -r requirements.txt

# ================================
# Stage 3: Application
# ================================
FROM dependencies as application

# Copy application code
COPY app/ ./app/
COPY static/ ./static/
COPY LICENSE ./

# Create directories for runtime
RUN mkdir -p /app/logs /app/cache /app/models && \
    chown -R kokorotts:kokorotts /app

# ================================
# Stage 4: Runtime (Final)
# ================================
FROM application as runtime

# Health check dependencies
COPY docker/healthcheck.py ./docker/
COPY docker/entrypoint.sh ./docker/

# Make scripts executable
RUN chmod +x ./docker/entrypoint.sh ./docker/healthcheck.py

# Switch to non-root user
USER kokorotts

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python docker/healthcheck.py

# Default environment variables
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    LOG_LEVEL=info \
    KOKORO_CACHE_DIR=/app/cache \
    KOKORO_MODEL_DIR=/app/models

# Use entrypoint script for flexibility
ENTRYPOINT ["./docker/entrypoint.sh"]

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Build-time labels
LABEL maintainer="KokoroTTS API" \
      description="Production-ready KokoroTTS API with OpenAI compatibility" \
      version="1.0.0" \
      org.opencontainers.image.title="KokoroTTS API" \
      org.opencontainers.image.description="FastAPI wrapper for KokoroTTS model with voice blending and streaming" \
      org.opencontainers.image.url="https://github.com/MajesticFelix/kokorotts-api" \
      org.opencontainers.image.source="https://github.com/MajesticFelix/kokorotts-api" \
      org.opencontainers.image.vendor="KokoroTTS" \
      org.opencontainers.image.licenses="Apache-2.0"