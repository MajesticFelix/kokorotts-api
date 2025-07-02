# Cloud Deployment Configurations

This directory contains cloud-specific deployment configurations for various platforms.

## Files

- **`k8s-deployment.yaml`** - Kubernetes deployment with auto-scaling
- **`nginx.conf`** - Nginx reverse proxy configuration
- **`aws-ecs-task.json`** - AWS ECS Fargate task definition
- **`gcp-cloud-run.yaml`** - Google Cloud Run service configuration
- **`azure-container-instance.json`** - Azure Container Instances template

## Platform-Specific Deployments

### Kubernetes (Generic)

**Features:**
- High availability with 3+ replicas
- Horizontal pod auto-scaling
- Ingress with SSL termination
- Redis for distributed rate limiting

**Deploy:**
```bash
# Apply all resources
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods -n kokorotts
kubectl get services -n kokorotts
kubectl get ingress -n kokorotts
```

**Scale:**
```bash
# Manual scaling
kubectl scale deployment kokorotts-api --replicas=5 -n kokorotts

# Auto-scaling is configured for CPU > 70% and Memory > 80%
```

### AWS ECS Fargate

**Features:**
- Serverless container hosting
- Integration with AWS services
- CloudWatch logging
- ALB integration

**Deploy:**
```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://aws-ecs-task.json

# Create service
aws ecs create-service \
  --cluster your-cluster \
  --service-name kokorotts-api \
  --task-definition kokorotts-api:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

**Prerequisites:**
- ECS cluster
- VPC with subnets
- Security groups
- IAM roles
- ElastiCache Redis cluster
- Application Load Balancer

### Google Cloud Run

**Features:**
- Fully managed serverless
- Auto-scaling to zero
- Pay-per-request pricing
- Integration with GCP services

**Deploy:**
```bash
# Deploy service
gcloud run services replace gcp-cloud-run.yaml --region=us-central1

# Get service URL
gcloud run services describe kokorotts-api --region=us-central1 --format="value(status.url)"
```

**Prerequisites:**
- Container image in GCR/Artifact Registry
- Cloud Memorystore (Redis) instance
- VPC connector for private Redis access

### Azure Container Instances

**Features:**
- Quick deployment
- Pay-per-second billing
- Easy integration with Azure services
- Public IP with DNS label

**Deploy:**
```bash
# Deploy using ARM template
az deployment group create \
  --resource-group your-rg \
  --template-file azure-container-instance.json \
  --parameters dnsNameLabel=your-unique-dns-name
```

**Prerequisites:**
- Azure Container Registry
- Azure Cache for Redis
- Resource group

## Nginx Reverse Proxy

The `nginx.conf` provides:
- Additional rate limiting (60/min general, 30/min for speech)
- SSL termination
- Load balancing
- Caching for information endpoints
- Security headers

**Usage with Docker:**
```bash
# Use with docker-compose
docker-compose -f docker/docker-compose.prod.yml --profile proxy up -d
```

**Standalone deployment:**
```bash
# Copy SSL certificates to ./ssl/
# Update server_name in nginx.conf
docker run -d \
  -p 80:80 -p 443:443 \
  -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro \
  -v $(pwd)/ssl:/etc/nginx/ssl:ro \
  nginx:alpine
```

## Rate Limiting Configuration

All cloud deployments enable rate limiting by default:

| Limit Type | Value | Scope |
|------------|-------|-------|
| Requests per minute | 30 | Per IP |
| Requests per hour | 200 | Per IP |
| Requests per day | 1000 | Per IP |
| Characters per request | 5000 | Per request |
| Characters per hour | 50000 | Per IP |
| Concurrent requests | 3 | Per IP |

### Adjusting Limits

Update environment variables in the deployment files:

```env
RATE_LIMIT_REQUESTS_PER_MINUTE=50    # Higher for premium users
RATE_LIMIT_MAX_CHARS_PER_REQUEST=10000  # Enterprise limits
```

## Monitoring and Observability

### Health Checks
All configurations include health checks on `/health` endpoint:
- Initial delay: 60 seconds
- Check interval: 30 seconds
- Timeout: 10 seconds
- Failure threshold: 3 attempts

### Metrics Collection
Monitor these endpoints:
- `/health` - Basic health status
- `/metrics` - System and rate limiting metrics
- `/debug` - Comprehensive diagnostic information

### Logging
Structured JSON logging is enabled in production mode:
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "message": "Rate limit applied",
  "ip": "192.168.1.1",
  "limit_type": "requests",
  "current_count": 25,
  "limit": 30
}
```

## Security Considerations

### Network Security
- Redis should be on private networks only
- Use VPC/VNET for internal communication
- Configure security groups/firewall rules properly
- Enable TLS encryption for Redis connections

### Application Security
- CORS is restricted to specific domains
- Rate limiting prevents abuse
- Input validation and sanitization
- Non-root container execution

### SSL/TLS
- Use valid SSL certificates
- Configure security headers
- Enable HSTS
- Use TLS 1.2+ only

## Cost Optimization

### AWS
- Use Fargate Spot for development
- Set up CloudWatch alarms for scaling
- Use reserved capacity for predictable workloads

### GCP
- Cloud Run scales to zero when not used
- Use preemptible instances for batch processing
- Configure appropriate CPU allocation

### Azure
- Use consumption-based pricing
- Implement auto-shutdown for development
- Monitor with Azure Cost Management

## Troubleshooting

### Common Issues

**Pod/Container not starting:**
```bash
# Check logs
kubectl logs deployment/kokorotts-api -n kokorotts
docker logs container-name

# Check resource limits
kubectl describe pod pod-name -n kokorotts
```

**Rate limiting not working:**
```bash
# Test Redis connectivity
kubectl exec -it deployment/redis -n kokorotts -- redis-cli ping

# Check environment variables
kubectl get configmap kokorotts-config -n kokorotts -o yaml
```

**High resource usage:**
```bash
# Monitor resources
kubectl top pods -n kokorotts
docker stats

# Check for memory leaks
kubectl exec -it deployment/kokorotts-api -n kokorotts -- ps aux
```

### Performance Tuning

**CPU-bound workloads:**
- Increase CPU requests/limits
- Enable horizontal auto-scaling
- Use faster instance types

**Memory-bound workloads:**
- Increase memory limits
- Monitor for memory leaks
- Optimize model loading

**Network-bound workloads:**
- Use CDN for static assets
- Optimize Redis connection pooling
- Enable compression