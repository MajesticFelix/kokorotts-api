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
RUN groupadd -r kokorotts && useradd -r -g kokorotts -s /bin/bash -m kokorotts

# Set work directory
WORKDIR /app

# ================================
# Stage 2: Python Dependencies
# ================================
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Create platform-compatible requirements
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Filter out platform-incompatible packages
    grep -v "triton" requirements.txt > requirements_filtered.txt && \
    # Install PyTorch with specified index URL (CPU or GPU)
    pip install --no-cache-dir torch torchaudio --index-url ${TORCH_INDEX_URL} && \
    # Install other dependencies (filtered)
    pip install --no-cache-dir -r requirements_filtered.txt

# ================================
# Stage 3: Application
# ================================
FROM dependencies as application

# Copy application code
COPY app/ ./app/
COPY static/ ./static/
COPY LICENSE ./

# Create directories for runtime
RUN mkdir -p /app/logs /app/cache /app/models /app/cache/huggingface /app/cache/transformers && \
    chown -R kokorotts:kokorotts /app && \
    chown -R kokorotts:kokorotts /home/kokorotts

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

# Default environment variables for KokoroTTS API
ENV \
    # Application settings
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    LOG_LEVEL=info \
    DEBUG=false \
    RELOAD=false \
    # API configuration
    API_TITLE="Kokoro TTS API" \
    API_VERSION="1.0.0" \
    API_DESCRIPTION="OpenAI-compatible TTS API using Kokoro model with voice blending support" \
    API_DOCS_URL="/docs" \
    API_REDOC_URL="/redoc" \
    API_STATIC_DIRECTORY="static" \
    API_SUPPORTED_FORMATS="wav,mp3,flac,ogg,opus" \
    API_MIN_SPEED=0.25 \
    API_MAX_SPEED=4.0 \
    API_DEFAULT_LANGUAGE="a" \
    API_DEFAULT_VOICE="af_heart" \
    API_DEFAULT_FORMAT="mp3" \
    API_VOICE_CACHE_DURATION=3600 \
    API_CORS_ORIGINS="*" \
    API_CORS_METHODS="*" \
    API_CORS_HEADERS="*" \
    API_ALLOW_CREDENTIALS=true \
    # TTS Engine configuration
    KOKORO_DEVICE="auto" \
    KOKORO_SAMPLE_RATE=24000 \
    KOKORO_DEFAULT_CHUNK_SIZE=800 \
    KOKORO_MAX_CHUNK_SIZE=1000 \
    KOKORO_STREAMING_CHUNK_SIZE=800 \
    KOKORO_MP3_BITRATE="192k" \
    KOKORO_OPUS_CODEC="libopus" \
    KOKORO_MEMORY_LIMIT_MB=1024 \
    KOKORO_BATCH_SIZE=5 \
    KOKORO_CACHE_DIR=/app/cache \
    KOKORO_MODEL_DIR=/app/models \
    KOKORO_REPO_ID="hexgrad/Kokoro-82M" \
    # Security configuration
    SECURITY_API_KEY_ENABLED=false \
    SECURITY_RATE_LIMIT_ENABLED=false \
    SECURITY_RATE_LIMIT_PER_MINUTE=60 \
    SECURITY_RATE_LIMIT_BURST=10 \
    # Monitoring configuration
    MONITORING_HEALTH_CHECK_ENABLED=true \
    MONITORING_HEALTH_CHECK_PATH="/health" \
    MONITORING_METRICS_ENABLED=true \
    MONITORING_METRICS_PATH="/metrics" \
    MONITORING_LOG_FORMAT="json" \
    MONITORING_SENTRY_ENVIRONMENT="production" \
    # System paths and cache
    HOME=/home/kokorotts \
    HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/transformers

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