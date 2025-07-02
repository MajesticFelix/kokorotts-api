# KokoroTTS API Dockerfile
# Supports both local development and cloud deployment with configurable rate limiting

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Default configuration for local development (no rate limiting)
ENV RATE_LIMITING_ENABLED=false \
    DEPLOYMENT_MODE=local \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    CORS_ORIGINS="*" \
    TRUST_PROXY_HEADERS=true \
    MAX_TEXT_LENGTH=50000 \
    DEFAULT_LANGUAGE=a \
    VOICE_CACHE_DURATION=3600

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]