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

### Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/MajesticFelix/kokorotts-api
cd kokorotts-api

# Start the API
docker-compose up -d

# Test the API
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

### Voice Blending Example

```bash
# Equal blend of two voices
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This speech uses blended voices for unique sound.",
    "voice": "af_bella+af_heart",
    "response_format": "mp3"
    "speed": 1,
    "stream": false,
    "include_captions": false,
    "language": "a"
  }' \
  --output blended.mp3

# Weighted blend (2:1 ratio)
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Custom voice blend with specific weights.",
    "voice": "af_bella(2)+af_heart(1)",
    "response_format": "mp3"
    "speed": 1,
    "stream": false,
    "include_captions": false,
    "language": "a"
  }' \
  --output weighted_blend.mp3
```

## 📦 Installation

### Option 1: Docker (Recommended)

```bash
# Basic setup
docker-compose up -d

# With Redis for rate limiting
docker-compose --profile redis up -d

# With PostgreSQL for API keys
docker-compose --profile database up -d

# Full production setup
docker-compose --profile redis --profile database up -d
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

### Option 3: Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env.local

# Edit configuration as needed
nano .env.local

# Start with development settings
DEPLOYMENT_MODE=development python -m uvicorn app.main:app --reload
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

## ⚙️ Configuration

### Environment Variables

The API uses a structured configuration system with environment variable prefixes:

#### TTS Configuration (`TTS_*`)
```env
TTS_MAX_TEXT_LENGTH=50000
TTS_DEFAULT_LANGUAGE=a
TTS_SAMPLE_RATE=24000
TTS_SUPPORTED_FORMATS=wav,mp3,flac,ogg,opus
TTS_ENABLE_GPU=true
TTS_MEMORY_LIMIT_MB=1024
```

#### Security Configuration (`SECURITY_*`)
```env
SECURITY_CORS_ORIGINS=*
SECURITY_TRUST_PROXY_HEADERS=true
SECURITY_REQUEST_TIMEOUT=300
SECURITY_ENABLE_METRICS=true
```

#### Rate Limiting
```env
RATE_LIMITING_ENABLED=false
RATE_LIMIT_REQUESTS_PER_MINUTE=30
RATE_LIMIT_CHARS_PER_REQUEST=5000
RATE_LIMIT_USE_REDIS=false
REDIS_URL=redis://localhost:6379/0
```

#### API Key Authentication
```env
API_KEY_AUTHENTICATION=false
API_KEY_REQUIRED=false
API_KEY_DATABASE_URL=sqlite:///./api_keys.db
```

See [`.env.example`](.env.example) for complete configuration options.

## 🚢 Deployment

### Docker Compose Production

```bash
# Use production configuration
docker-compose -f docker/docker-compose.prod.yml up -d
```

### Cloud Deployments

#### Kubernetes
```bash
kubectl apply -f cloud/k8s-deployment.yaml
```

#### AWS ECS Fargate
```bash
aws ecs register-task-definition --cli-input-json file://cloud/aws-ecs-task.json
```

#### Google Cloud Run
```bash
gcloud run services replace cloud/gcp-cloud-run.yaml --region=us-central1
```

#### Azure Container Instances
```bash
az deployment group create \
  --resource-group your-rg \
  --template-file cloud/azure-container-instance.json
```

For detailed deployment instructions, see the [Cloud Deployment Guide](cloud/README.md).

## 📊 Monitoring

### Prometheus + Grafana Stack

```bash
# Start monitoring stack
docker-compose -f monitoring/docker-compose-monitoring.yml up -d

# Access Grafana
open http://localhost:3000  # admin/kokorotts123
```

### Key Metrics

- Request rates and response times
- Rate limiting effectiveness
- Character usage and limits
- System resource utilization
- GPU memory usage (if available)

For complete monitoring setup, see the [Monitoring Guide](monitoring/README.md).

## 💡 Examples

### Python Client

```python
import requests

response = requests.post("http://localhost:8000/v1/audio/speech", json={
    "model": "kokoro",
    "input": "Hello from Python!",
    "voice": "af_heart",
    "response_format": "mp3",
    "speed": 1,
    "stream": False,
    "include_captions": False,
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
    voice: 'af_bella+af_heart',
    response_format: 'mp3'
    speed: 1,
    stream: false,
    include_captions: false,
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

### Troubleshooting

**API not starting:**
```bash
# Check logs
docker logs kokorotts-api

# Verify dependencies
pip install -r requirements.txt
```

**Audio quality issues:**
- Ensure proper voice selection for target language
- Adjust speed parameter (0.25-4.0)
- Try different audio formats

**Rate limiting not working:**
- Verify Redis connection
- Check `RATE_LIMITING_ENABLED=true`
- Review configuration in `/admin/rate-limits/config`