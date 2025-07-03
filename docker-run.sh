#!/bin/bash
# KokoroTTS API Docker Helper Script
# This script provides convenient commands to run Docker operations from the root directory

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo -e "${BLUE}KokoroTTS API Docker Helper${NC}"
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  up                 Start services (CPU version)"
    echo "  up-gpu             Start services (GPU version)"
    echo "  up-dev             Start development environment"
    echo "  up-prod            Start production environment"
    echo "  down               Stop services"
    echo "  build              Build images"
    echo "  build-cpu          Build CPU image"
    echo "  build-gpu          Build GPU image"
    echo "  logs               Show logs"
    echo "  shell              Access container shell"
    echo "  test               Test the API"
    echo "  clean              Clean up volumes and images"
    echo "  monitoring         Start monitoring stack"
    echo ""
    echo "Examples:"
    echo "  $0 up              # Start CPU version"
    echo "  $0 up-dev          # Start development environment"
    echo "  $0 up-prod         # Start production with monitoring"
    echo "  $0 logs            # Show application logs"
    echo "  $0 shell           # Access container bash shell"
}

# Main docker-compose file paths
COMPOSE_FILE="docker/docker-compose.yml"
COMPOSE_DEV="docker/docker-compose.dev.yml"
COMPOSE_PROD="docker/docker-compose.prod.yml"

case "$1" in
    "up")
        echo -e "${GREEN}Starting KokoroTTS API (CPU version)...${NC}"
        docker-compose -f $COMPOSE_FILE up -d
        echo -e "${GREEN}Services started! API available at http://localhost:8000${NC}"
        ;;
    "up-gpu")
        echo -e "${GREEN}Starting KokoroTTS API (GPU version)...${NC}"
        # First, check if GPU service is uncommented
        if grep -q "# kokorotts-api-gpu:" $COMPOSE_FILE; then
            echo -e "${YELLOW}GPU service is commented out in docker-compose.yml${NC}"
            echo "Please uncomment the GPU service section and comment out the CPU service"
            exit 1
        fi
        docker-compose -f $COMPOSE_FILE up -d kokorotts-api-gpu
        echo -e "${GREEN}GPU services started! API available at http://localhost:8000${NC}"
        ;;
    "up-dev")
        echo -e "${GREEN}Starting development environment...${NC}"
        docker-compose -f $COMPOSE_FILE -f $COMPOSE_DEV up
        ;;
    "up-prod")
        echo -e "${GREEN}Starting production environment...${NC}"
        docker-compose -f $COMPOSE_FILE -f $COMPOSE_PROD up -d
        echo -e "${GREEN}Production services started!${NC}"
        echo "API: http://localhost:8000"
        echo "Nginx: http://localhost:80"
        ;;
    "down")
        echo -e "${GREEN}Stopping services...${NC}"
        docker-compose -f $COMPOSE_FILE -f $COMPOSE_DEV -f $COMPOSE_PROD down
        ;;
    "build")
        echo -e "${GREEN}Building all images...${NC}"
        docker-compose -f $COMPOSE_FILE build
        ;;
    "build-cpu")
        echo -e "${GREEN}Building CPU image...${NC}"
        docker build -f docker/Dockerfile.cpu -t kokorotts-api:cpu .
        ;;
    "build-gpu")
        echo -e "${GREEN}Building GPU image...${NC}"
        docker build -f docker/Dockerfile.gpu -t kokorotts-api:gpu .
        ;;
    "logs")
        service=${2:-kokorotts-api}
        echo -e "${GREEN}Showing logs for $service...${NC}"
        docker-compose -f $COMPOSE_FILE logs -f $service
        ;;
    "shell")
        container=${2:-kokorotts-api}
        echo -e "${GREEN}Accessing $container shell...${NC}"
        docker exec -it $container bash
        ;;
    "test")
        echo -e "${GREEN}Testing API...${NC}"
        curl -X POST "http://localhost:8000/v1/audio/speech" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "kokoro",
            "input": "Hello, this is a test of the KokoroTTS API!",
            "voice": "af_heart",
            "response_format": "mp3",
            "speed": 1,
            "stream": false,
            "include_captions": false,
            "language": "a"
          }' \
          --output test-output.mp3 \
          --write-out "Status: %{http_code}\nTime: %{time_total}s\n"
        
        if [ -f "test-output.mp3" ]; then
            echo -e "${GREEN}✅ Test successful! Audio saved to test-output.mp3${NC}"
        else
            echo -e "${YELLOW}⚠️  No audio file generated${NC}"
        fi
        ;;
    "clean")
        echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
        docker-compose -f $COMPOSE_FILE -f $COMPOSE_DEV -f $COMPOSE_PROD down -v
        docker system prune -f
        echo -e "${GREEN}Cleanup complete!${NC}"
        ;;
    "monitoring")
        echo -e "${GREEN}Starting monitoring stack...${NC}"
        docker-compose -f $COMPOSE_FILE --profile monitoring up -d
        echo -e "${GREEN}Monitoring started!${NC}"
        echo "Prometheus: http://localhost:9090"
        echo "Grafana: http://localhost:3000 (admin/admin)"
        ;;
    "status")
        echo -e "${GREEN}Service status:${NC}"
        docker-compose -f $COMPOSE_FILE ps
        ;;
    "health")
        echo -e "${GREEN}Health check:${NC}"
        curl -s http://localhost:8000/health | python -m json.tool || echo "API not responding"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac