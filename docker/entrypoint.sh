#!/bin/bash
# KokoroTTS API Docker Entrypoint Script
# Handles container initialization, configuration, and startup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration defaults
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export WORKERS=${WORKERS:-1}
export LOG_LEVEL=${LOG_LEVEL:-info}
export KOKORO_CACHE_DIR=${KOKORO_CACHE_DIR:-/app/cache}
export KOKORO_MODEL_DIR=${KOKORO_MODEL_DIR:-/app/models}
export KOKORO_DEVICE=${KOKORO_DEVICE:-auto}
export HOME=${HOME:-/home/kokorotts}
export HF_HOME=${HF_HOME:-/app/cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/app/cache/transformers}

# Function to detect device capability
detect_device() {
    if [ "$KOKORO_DEVICE" = "auto" ]; then
        log_info "Auto-detecting compute device..."
        
        # Check for NVIDIA GPU
        if command -v nvidia-smi >/dev/null 2>&1; then
            if nvidia-smi >/dev/null 2>&1; then
                log_success "NVIDIA GPU detected, using CUDA"
                export KOKORO_DEVICE="cuda"
                return 0
            fi
        fi
        
        # Check for CUDA availability via Python
        if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            log_success "CUDA available via PyTorch, using GPU"
            export KOKORO_DEVICE="cuda"
            return 0
        fi
        
        log_info "No GPU detected, using CPU"
        export KOKORO_DEVICE="cpu"
    else
        log_info "Device set to: $KOKORO_DEVICE"
    fi
}

# Function to create required directories
create_directories() {
    log_info "Creating required directories..."
    
    directories=(
        "$KOKORO_CACHE_DIR"
        "$KOKORO_MODEL_DIR"
        "/app/logs"
        "$HF_HOME"
        "$TRANSFORMERS_CACHE"
        "/home/kokorotts"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_success "Created directory: $dir"
        else
            log_info "Directory exists: $dir"
        fi
    done
}

# Function to set appropriate permissions
set_permissions() {
    log_info "Setting permissions for application directories..."
    
    # Only change ownership if we're running as root
    if [ "$(id -u)" = "0" ]; then
        chown -R kokorotts:kokorotts "$KOKORO_CACHE_DIR" "$KOKORO_MODEL_DIR" "/app/logs" 2>/dev/null || true
        log_success "Permissions set for kokorotts user"
    else
        log_info "Running as non-root user, skipping ownership changes"
    fi
}

# Function to validate environment
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Check if port is numeric and in valid range
    if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
        log_error "Invalid port number: $PORT"
        exit 1
    fi
    
    # Check if workers is numeric and reasonable
    if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -lt 1 ] || [ "$WORKERS" -gt 16 ]; then
        log_error "Invalid worker count: $WORKERS"
        exit 1
    fi
    
    # Validate log level
    valid_log_levels=("debug" "info" "warning" "error" "critical")
    if [[ ! " ${valid_log_levels[@]} " =~ " ${LOG_LEVEL} " ]]; then
        log_error "Invalid log level: $LOG_LEVEL"
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Function to wait for dependencies
wait_for_dependencies() {
    log_info "Checking for external dependencies..."
    
    # If Redis is configured, wait for it
    if [ -n "$REDIS_URL" ]; then
        log_info "Waiting for Redis connection..."
        # Extract host and port from Redis URL
        redis_host=$(echo "$REDIS_URL" | sed -n 's|redis://\([^:]*\):\([0-9]*\)/.*|\1|p')
        redis_port=$(echo "$REDIS_URL" | sed -n 's|redis://\([^:]*\):\([0-9]*\)/.*|\2|p')
        
        if [ -n "$redis_host" ] && [ -n "$redis_port" ]; then
            wait_for_service "$redis_host" "$redis_port" "Redis"
        fi
    fi
    
    # If database URL is configured, wait for it
    if [ -n "$DATABASE_URL" ] && [[ "$DATABASE_URL" != sqlite* ]]; then
        log_info "External database detected, checking connectivity..."
        # Add database connectivity check here if needed
    fi
}

# Function to wait for a service to be available
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    local counter=0
    
    log_info "Waiting for $service_name at $host:$port..."
    
    while ! nc -z "$host" "$port" 2>/dev/null; do
        counter=$((counter + 1))
        if [ $counter -gt $timeout ]; then
            log_error "Timeout waiting for $service_name"
            exit 1
        fi
        sleep 1
    done
    
    log_success "$service_name is available"
}

# Function to perform health check before starting
initial_health_check() {
    log_info "Performing initial system check..."
    
    # Check Python availability
    if ! command -v python >/dev/null 2>&1; then
        log_error "Python not found"
        exit 1
    fi
    
    # Check if we can import required modules
    if ! python -c "import fastapi, uvicorn, torch" >/dev/null 2>&1; then
        log_error "Required Python modules not available"
        exit 1
    fi
    
    # Check KokoroTTS availability
    if ! python -c "import kokoro" >/dev/null 2>&1; then
        log_error "KokoroTTS module not available"
        exit 1
    fi
    
    log_success "Initial health check passed"
}

# Function to optimize for device
optimize_for_device() {
    log_info "Optimizing configuration for device: $KOKORO_DEVICE"
    
    if [ "$KOKORO_DEVICE" = "cpu" ]; then
        # CPU optimizations
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
        export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
        export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-2}
        
        # Memory optimizations
        export PYTHONMALLOC=${PYTHONMALLOC:-malloc}
        export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
        
        log_success "CPU optimizations applied"
    elif [ "$KOKORO_DEVICE" = "cuda" ]; then
        # GPU optimizations
        export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
        export TORCH_CUDNN_ENABLED=${TORCH_CUDNN_ENABLED:-1}
        
        log_success "GPU optimizations applied"
    fi
}

# Function to display startup information
display_startup_info() {
    log_info "=== KokoroTTS API Container Starting ==="
    log_info "Host: $HOST"
    log_info "Port: $PORT"
    log_info "Workers: $WORKERS"
    log_info "Log Level: $LOG_LEVEL"
    log_info "Device: $KOKORO_DEVICE"
    log_info "Cache Dir: $KOKORO_CACHE_DIR"
    log_info "Model Dir: $KOKORO_MODEL_DIR"
    log_info "Python Version: $(python --version)"
    log_info "PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Unknown')"
    log_info "========================================"
}

# Function to handle shutdown signals
shutdown_handler() {
    log_info "Received shutdown signal, gracefully stopping..."
    
    # If uvicorn is running, send it a termination signal
    if [ -n "$UVICORN_PID" ]; then
        kill -TERM "$UVICORN_PID" 2>/dev/null || true
        wait "$UVICORN_PID" 2>/dev/null || true
    fi
    
    log_success "Application stopped gracefully"
    exit 0
}

# Main execution
main() {
    # Set up signal handlers
    trap shutdown_handler SIGTERM SIGINT SIGQUIT
    
    log_info "Starting KokoroTTS API container..."
    
    # Run initialization steps
    detect_device
    validate_environment
    create_directories
    set_permissions
    wait_for_dependencies
    initial_health_check
    optimize_for_device
    display_startup_info
    
    # If no command specified, start the default application
    if [ $# -eq 0 ]; then
        log_info "Starting default uvicorn server..."
        exec uvicorn app.main:app \
            --host "$HOST" \
            --port "$PORT" \
            --workers "$WORKERS" \
            --log-level "$LOG_LEVEL" \
            --access-log \
            --use-colors
    else
        # Execute the provided command
        log_info "Executing command: $*"
        exec "$@"
    fi
}

# Run main function with all arguments
main "$@"