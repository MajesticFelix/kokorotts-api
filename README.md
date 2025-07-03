# 🎤 KokoroTTS API

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-red.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-green.svg)](https://platform.openai.com/docs/api-reference/audio)

A production-ready, open-source FastAPI wrapper for the **Kokoro TTS model** (82M parameters) with enterprise-grade features including rate limiting, API key authentication, voice blending, streaming, and comprehensive monitoring.

## ✨ Key Features

🎭 **Advanced Voice Control**
- **Voice Blending**: Mix multiple voices with custom weights (`af_bella(2)+af_sky(1)`)
- **9 Languages**: English (US/UK), Japanese, Chinese, Spanish, French, Hindi, Italian, Portuguese
- **Multiple Formats**: WAV, MP3, FLAC, OGG, OPUS audio output

🚀 **Production Ready**
- **OpenAI Compatible**: Drop-in replacement for OpenAI TTS API
- **Enterprise Security**: Rate limiting, API key authentication, CORS protection
- **Streaming Support**: Real-time audio generation with optional word-level captions
- **Cloud Deployment**: Kubernetes, AWS ECS, Google Cloud Run, Azure ready

📊 **Monitoring & Observability**
- **Prometheus Metrics**: Request rates, rate limiting, system performance
- **Grafana Dashboards**: Visual monitoring and alerting
- **Health Checks**: Comprehensive diagnostics and debugging endpoints
- **Abuse Detection**: Automatic pattern recognition and blocking

⚡ **Performance Optimized**
- **Memory Efficient**: Automatic batch processing for long texts
- **GPU Optional**: Runs on CPU or GPU with automatic detection
- **Lightweight**: Only 350MB model size with 82M parameters
- **Async Processing**: Non-blocking I/O with FastAPI

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/MajesticFelix/kokorotts-api.git
cd kokorotts-api
```

#### CPU Version
```bash
# Start CPU-optimized version (using helper script)
./docker-run.sh up

# OR manually with docker-compose
docker-compose -f docker/docker-compose.yml up -d
```

#### GPU-Accelerated Version
```bash
# Build and run GPU version (requires NVIDIA Docker runtime)
./docker-run.sh build-gpu

# Or manually
docker build -f docker/Dockerfile.gpu -t kokorotts-api:gpu .

# Run with GPU support (using helper script)
./docker-run.sh up-gpu

# Or manually run with docker
docker run -d \
  --name kokorotts-api-gpu \
  --gpus all \
  -p 8000:8000 \
  -v kokorotts-models:/app/models \
  -v kokorotts-cache:/app/cache \
  kokorotts-api:gpu

# Or use docker-compose with GPU profile
# Uncomment the GPU service in docker/docker-compose.yml, then:
# docker-compose -f docker/docker-compose.yml up -d kokorotts-api-gpu
```

#### Production Deployment
```bash
# Production with monitoring, nginx, and optimized settings (using helper script)
./docker-run.sh up-prod

# Or manually
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d

# Enable monitoring stack (using helper script)
./docker-run.sh monitoring

# Or manually
docker-compose -f docker/docker-compose.yml --profile monitoring up -d

# Check all services
./docker-run.sh status
# Or: docker-compose -f docker/docker-compose.yml ps
```

#### Test 
```bash
# Test the API
./docker-run.sh test

# OR manually test
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello, world! This is Kokoro TTS.",
    "voice": "af_heart",
    "response_format": "mp3",
    "speed": 1,
    "stream": false,
    "include_captions": false,
    "language": "a"
  }' \
  --output speech.mp3

# Open test interface
open http://localhost:8000/test
# Open docs
open http://localhost:8000/docs
```

### Option 2: Python Installation

```bash
# Clone repository
git clone https://github.com/MajesticFelix/kokorotts-api.git
cd kokorotts-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Start the API
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🎯 API Documentation

### Core Endpoints

#### Text-to-Speech Synthesis
```http
POST /v1/audio/speech
```

#### List Available Voices
```http
GET /v1/audio/voices
```

#### List Supported Languages
```http
GET /v1/audio/languages
```

### Monitoring Endpoints

```http
GET /health          # Health check
GET /metrics         # Prometheus metrics
GET /debug           # Diagnostic information
GET /pipeline/status # TTS model status
```

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Test Interface**: `http://localhost:8000/test`

## 💡 Examples

### Python Client

```python
import requests

response = requests.post("http://localhost:8000/v1/audio/speech", json={
    "model": "kokoro",
    "input": "Hello from Python!",
    "voice": "af_heart", # Voice Blend Example (add more voices using '+' and increase the weights as you wish): af_heart(weight)+af_bella(weight)  
    "response_format": "mp3",
    "speed": 1,
    "stream": False, # Set True if you want real-time streaming
    "include_captions": False, # Set True if you want per word timestamped captions
    "language": "a"
})

with open("output.mp3", "wb") as f:
    f.write(response.content)
```

### JavaScript/Node.js

```javascript
const response = await fetch('http://localhost:8000/v1/audio/speech', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'kokoro',
    input: 'Hello from JavaScript!',
    voice: 'af_heart', // Voice Blend Example (add more voices using '+' and increase the weights as you wish): af_heart(weight)+af_bella(weight)  
    response_format: 'mp3'
    speed: 1,
    stream: false, // Set true if you want real-time streaming
    include_captions: false, // Set true if you want per word timestamped captions
    language: 'a'
  })
});

const audioBuffer = await response.arrayBuffer();
```

### cURL with Advanced Features

```bash
# Streaming audio
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "This is streaming audio generation.",
    "voice": "af_heart",
    "response_format": "mp3",
    "speed": 1,
    "stream": true,
    "include_captions": false,
    "language": "a"
  }' \
  --output streaming.mp3

# Different language
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "こんにちは、世界！",
    "voice": "af_heart",
    "response_format": "mp3",
    "speed": 1,
    "stream": true,
    "include_captions": false,
    "language": "j"
  }' \
  --output japanese.mp3
```

## 🐳 Docker Deployment Guide

### Docker Helper Script

For convenience, use the included `docker-run.sh` script for common operations:

```bash
# Basic operations
./docker-run.sh up              # Start CPU version
./docker-run.sh up-gpu          # Start GPU version (after uncommenting in compose file)
./docker-run.sh up-dev          # Start development environment
./docker-run.sh up-prod         # Start production environment
./docker-run.sh down            # Stop all services

# Building
./docker-run.sh build           # Build all images
./docker-run.sh build-cpu       # Build CPU-only image
./docker-run.sh build-gpu       # Build GPU image

# Utilities
./docker-run.sh test            # Test API with sample request
./docker-run.sh logs            # Show application logs
./docker-run.sh shell           # Access container shell
./docker-run.sh status          # Show service status
./docker-run.sh health          # Check API health
./docker-run.sh monitoring      # Start monitoring stack
./docker-run.sh clean           # Clean up resources
```

### Available Docker Images

The project provides three optimized Docker configurations:

- **`docker/Dockerfile`** - Multi-stage, flexible build (CPU/GPU compatible)
- **`docker/Dockerfile.cpu`** - CPU-optimized (smaller, faster build)
- **`docker/Dockerfile.gpu`** - GPU-optimized with CUDA support

### Build Options

#### CPU-Optimized Build
```bash
# Build CPU version (recommended for most deployments)
docker build -f docker/Dockerfile.cpu -t kokorotts-api:cpu .

# Run CPU version
docker run -d \
  --name kokorotts-api \
  -p 8000:8000 \
  -v kokorotts-models:/app/models \
  -v kokorotts-cache:/app/cache \
  -e WORKERS=2 \
  -e LOG_LEVEL=info \
  kokorotts-api:cpu
```

#### GPU-Accelerated Build
```bash
# Prerequisites: NVIDIA Docker runtime installed
# Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Build GPU version
docker build -f docker/Dockerfile.gpu -t kokorotts-api:gpu .

# Run GPU version
docker run -d \
  --name kokorotts-api-gpu \
  --gpus all \
  -p 8000:8000 \
  -v kokorotts-models:/app/models \
  -v kokorotts-cache:/app/cache \
  -e KOKORO_DEVICE=cuda \
  -e WORKERS=1 \
  kokorotts-api:gpu
```

#### Multi-Platform Build
```bash
# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.cpu \
  -t kokorotts-api:latest \
  --push .
```

### Docker Compose Configurations

#### Development Setup
```bash
# Development with hot reload and debugging
cp .env.example .env  # Edit configuration as needed
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up

# Enable development tools (Jupyter, file watcher)
docker-compose -f docker/docker-compose.yml --profile jupyter --profile watcher up -d
```

#### Production Setup
```bash
# Create production directories
sudo mkdir -p /opt/kokorotts/{models,cache,logs}
sudo chown -R 1000:1000 /opt/kokorotts

# Production deployment
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d

# With full monitoring stack
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml --profile monitoring up -d
```

#### Scaling Setup
```bash
# Scale API instances
docker-compose -f docker/docker-compose.yml up -d --scale kokorotts-api=3

# With load balancer
docker-compose -f docker/docker-compose.yml --profile nginx up -d
```

### Environment Configuration

Copy and customize the environment file:
```bash
cp .env.example .env
```

Key configuration options:
```bash
# Basic settings
HOST=0.0.0.0
PORT=8000
WORKERS=2
LOG_LEVEL=info

# Device configuration
KOKORO_DEVICE=cpu  # or 'cuda' for GPU

# Performance tuning
OMP_NUM_THREADS=2
KOKORO_MEMORY_LIMIT_MB=1024

# Security (optional)
API_KEY_ENABLED=false
RATE_LIMIT_ENABLED=false

# Monitoring
METRICS_ENABLED=true
HEALTH_CHECK_ENABLED=true
```

### Production Checklist

**Before deployment:**
- [ ] Configure environment variables in `.env`
- [ ] Set up persistent volumes for models and cache
- [ ] Configure monitoring (Prometheus/Grafana)
- [ ] Set up reverse proxy (Nginx) with SSL
- [ ] Configure log rotation and backup
- [ ] Test health checks and auto-restart

**Security:**
- [ ] Enable API key authentication
- [ ] Configure rate limiting
- [ ] Set up firewall rules
- [ ] Use non-root user (already configured)
- [ ] Enable SSL/TLS certificates
- [ ] Restrict metrics endpoint access

**Monitoring:**
```bash
# Health check
curl http://localhost:8000/health

# System metrics
curl http://localhost:8000/metrics

# Container logs
docker-compose logs -f kokorotts-api

# Resource usage
docker stats kokorotts-api
```

### Cloud Deployment Examples

#### AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker build -f docker/Dockerfile.cpu -t kokorotts-api .
docker tag kokorotts-api:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/kokorotts-api:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/kokorotts-api:latest

# Deploy using ECS task definition
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/kokorotts-api --file docker/Dockerfile.cpu
gcloud run deploy kokorotts-api \
  --image gcr.io/PROJECT-ID/kokorotts-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

#### Azure Container Instances
```bash
# Create container group
az container create \
  --resource-group myResourceGroup \
  --name kokorotts-api \
  --image kokorotts-api:cpu \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables WORKERS=2 LOG_LEVEL=info
```

#### Kubernetes
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kokorotts-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kokorotts-api
  template:
    metadata:
      labels:
        app: kokorotts-api
    spec:
      containers:
      - name: kokorotts-api
        image: kokorotts-api:cpu
        ports:
        - containerPort: 8000
        env:
        - name: WORKERS
          value: "1"
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Code Style

- **Python**: Follow PEP 8, use type hints
- **API Design**: RESTful principles, OpenAPI compatibility
- **Error Handling**: Comprehensive error responses
- **Testing**: Unit tests for core functionality
- **Documentation**: Inline docstrings and API documentation

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the **Apache License 2.0** - the same license as the official Kokoro TTS model.

This means:
- ✅ **Commercial use allowed** - Build products and services
- ✅ **Modification allowed** - Adapt and extend the code
- ✅ **Distribution allowed** - Share and redistribute
- ✅ **Private use allowed** - Use in private projects
- ⚠️ **License and copyright notice required** - Include attribution

The Kokoro TTS model was trained exclusively on permissive, non-copyrighted, public domain audio data, ensuring freedom from legal restrictions.

See [LICENSE](LICENSE) for full license text.

## 🔧 Troubleshooting

### Docker Issues

**Container not starting:**
```bash
# Check container logs
docker logs kokorotts-api
docker-compose -f docker/docker-compose.yml logs kokorotts-api

# Check container status
docker ps -a

# Rebuild container
docker-compose -f docker/docker-compose.yml down
docker-compose -f docker/docker-compose.yml build --no-cache
docker-compose -f docker/docker-compose.yml up -d
```

**GPU not detected:**
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check CUDA availability in container
docker exec kokorotts-api python -c "import torch; print(torch.cuda.is_available())"

# Install NVIDIA Container Toolkit if needed
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Memory issues:**
```bash
# Check memory usage
docker stats kokorotts-api

# Increase memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G

# Enable swap accounting
echo 'GRUB_CMDLINE_LINUX="cgroup_enable=memory swapaccount=1"' | sudo tee -a /etc/default/grub
sudo update-grub && sudo reboot
```

**Permission errors:**
```bash
# Fix volume permissions
sudo chown -R 1000:1000 /opt/kokorotts
docker-compose -f docker/docker-compose.yml down && docker-compose -f docker/docker-compose.yml up -d

# Check if running as correct user
docker exec kokorotts-api id
```

**Network connectivity:**
```bash
# Check if port is accessible
curl http://localhost:8000/health

# Check Docker network
docker network ls
docker network inspect kokorotts_kokorotts-network

# Test internal connectivity
docker exec kokorotts-api curl http://localhost:8000/health
```

### Application Issues

**API not starting:**
```bash
# Check detailed logs
docker logs kokorotts-api --follow

# Verify environment variables
docker exec kokorotts-api env | grep KOKORO

# Test manually
docker exec -it kokorotts-api python -c "from app.main import app; print('App loads successfully')"

# Restart with debugging
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up
```

**Model loading errors:**
```bash
# Clear model cache
docker volume rm kokorotts_kokorotts-models kokorotts_kokorotts-cache
docker-compose -f docker/docker-compose.yml up -d

# Check available disk space
docker exec kokorotts-api df -h

# Manually download models
docker exec kokorotts-api python -c "from kokoro import KPipeline; KPipeline('a')"
```

**Audio generation issues:**
```bash
# Test basic TTS functionality
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "test", "voice": "af_heart"}' \
  -o test.mp3

# Check available voices
curl http://localhost:8000/v1/audio/voices

# Verify audio format support
docker exec kokorotts-api python -c "import soundfile as sf; print(sf.available_formats())"
```

**Performance issues:**
```bash
# Monitor resource usage
docker stats kokorotts-api

# Check worker processes
docker exec kokorotts-api ps aux

# Optimize worker count for CPU
# Edit docker/docker-compose.yml: WORKERS=<number_of_cpu_cores>

# For GPU, use single worker
# Edit docker/docker-compose.yml: WORKERS=1
```

### Health Check Failures

**Health check timeouts:**
```bash
# Increase health check timeout
# In docker/Dockerfile:
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3

# Test health check manually
docker exec kokorotts-api python docker/healthcheck.py

# Check application startup time
docker logs kokorotts-api | grep "startup"
```

### Development Issues

**Hot reload not working:**
```bash
# Ensure development compose is used
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up

# Check volume mounts
docker inspect kokorotts-api | grep Mounts -A 20

# Restart development environment
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml down
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up --build
```

**Debugging in container:**
```bash
# Access container shell
docker exec -it kokorotts-api bash

# Run with debugger
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up
# Then connect debugger to port 5678

# View real-time logs
docker-compose -f docker/docker-compose.yml logs -f kokorotts-api
```

### Audio Quality Issues

**Poor audio quality:**
- Ensure proper voice selection for target language
- Adjust speed parameter (0.25-4.0)
- Try different audio formats (WAV for best quality)
- Check if GPU acceleration is working properly

**Clipped or distorted audio:**
```bash
# Check volume levels in generated audio
ffprobe -v quiet -show_streams test.mp3

# Reduce synthesis speed
curl -X POST http://localhost:8000/v1/audio/speech \
  -d '{"input": "test", "voice": "af_heart", "speed": 0.8}'

# Try different voice
curl http://localhost:8000/v1/audio/voices
```

### Monitoring and Logging

**Enable debug logging:**
```bash
# Set debug level
docker-compose -f docker/docker-compose.yml up -d --env LOG_LEVEL=debug

# View detailed logs
docker-compose -f docker/docker-compose.yml logs -f kokorotts-api

# Access metrics
curl http://localhost:8000/metrics
```

**Check system resources:**
```bash
# View Prometheus metrics
open http://localhost:9090

# View Grafana dashboard
open http://localhost:3000

# Check container resource usage
docker exec kokorotts-api cat /proc/meminfo
docker exec kokorotts-api cat /proc/cpuinfo
```