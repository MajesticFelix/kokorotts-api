# Docker Configuration for KokoroTTS API

This directory contains Docker configurations for both local development and production deployment.

## Files

- **`Dockerfile`** - Container definition for the KokoroTTS API
- **`docker-compose.yml`** - Local development setup (rate limiting disabled)
- **`docker-compose.prod.yml`** - Production setup with Redis and rate limiting

## Quick Start

### Local Development
```bash
# From project root
docker-compose -f docker/docker-compose.yml up -d

# Access the API
curl http://localhost:8000/health
```

### Production Deployment
```bash
# From project root
docker-compose -f docker/docker-compose.prod.yml up -d

# Access with rate limits
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello","voice":"af_heart"}'
```

## Configuration

### Local Development Features
- ✅ No rate limiting
- ✅ Debug mode enabled
- ✅ Source code mounting
- ✅ Higher text limits (100,000 chars)
- ✅ Hot reload support

### Production Features
- ✅ Rate limiting enabled
- ✅ Redis for distributed limiting
- ✅ Security hardening
- ✅ Resource limits
- ✅ Health monitoring

## Environment Variables

The Docker containers use environment variables for configuration. See the main `.env.example` file for all available options.

### Key Variables for Docker

| Variable | Local | Production | Description |
|----------|--------|------------|-------------|
| `RATE_LIMITING_ENABLED` | `false` | `true` | Enable rate limiting |
| `DEPLOYMENT_MODE` | `local` | `cloud` | Deployment environment |
| `DEBUG` | `true` | `false` | Debug mode |
| `MAX_TEXT_LENGTH` | `100000` | `50000` | Max characters per request |

## Scaling

### Horizontal Scaling
```bash
# Scale API containers
docker-compose -f docker/docker-compose.prod.yml up -d --scale kokoro-api=3
```

### Resource Monitoring
```bash
# Monitor resource usage
docker stats

# View logs
docker-compose -f docker/docker-compose.prod.yml logs -f kokoro-api
```

## Testing Rate Limiting

### Local (No Limits)
```bash
# This will succeed indefinitely
for i in {1..100}; do
  curl -X POST localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d "{\"input\":\"Test ${i}\",\"voice\":\"af_heart\"}"
done
```

### Production (With Limits)
```bash
# This will hit rate limits after 30 requests
for i in {1..50}; do
  curl -X POST localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d "{\"input\":\"Test ${i}\",\"voice\":\"af_heart\"}"
done
```

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check logs
docker-compose logs kokoro-api

# Check configuration
docker-compose config
```

**Rate limiting not working:**
```bash
# Check Redis connection
docker exec kokorotts-api_redis_1 redis-cli ping

# Verify environment variables
docker exec kokorotts-api_kokoro-api_1 env | grep RATE_LIMIT
```

**High memory usage:**
```bash
# Monitor resources
docker stats --no-stream

# Check model loading
docker logs kokorotts-api_kokoro-api_1 | grep pipeline
```

## Security Notes

- The production configuration runs containers as non-root users
- Redis is isolated on an internal network
- CORS is restricted to specified domains
- SSL termination should be handled by a reverse proxy

## Development Tips

### Code Changes
With the local setup, your code changes will be reflected immediately since the source directory is mounted into the container.

### Database Access
```bash
# Connect to Redis in development
docker exec -it kokorotts-api_redis_1 redis-cli

# View rate limiting data
KEYS "rate_limit:*"
```

### Performance Testing
```bash
# Load testing with Apache Bench
ab -n 100 -c 10 http://localhost:8000/health

# For TTS endpoint testing
# (Use a tool like curl in a loop or a proper load testing tool)
```