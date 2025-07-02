# KokoroTTS API Monitoring

This directory contains monitoring configurations for the KokoroTTS API, including Prometheus metrics, Grafana dashboards, and alerting rules.

## Overview

The monitoring stack includes:
- **Prometheus** - Metrics collection and storage
- **Grafana** - Visualization and dashboards
- **AlertManager** - Alert handling and notifications
- **Node Exporter** - System metrics
- **cAdvisor** - Container metrics
- **Redis Exporter** - Redis metrics

## Quick Start

### 1. Start Monitoring Stack
```bash
# Start all monitoring services
docker-compose -f monitoring/docker-compose-monitoring.yml up -d

# Check status
docker-compose -f monitoring/docker-compose-monitoring.yml ps
```

### 2. Access Dashboards
- **Grafana**: http://localhost:3000 (admin/kokorotts123)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

### 3. Configure KokoroTTS API
```bash
# Update your KokoroTTS API to use the monitoring Redis
export REDIS_URL=redis://localhost:6379/0
export RATE_LIMITING_ENABLED=true
export DEPLOYMENT_MODE=cloud
```

## Monitoring Features

### Rate Limiting Metrics
- Request rates and blocking rates
- Character usage statistics
- Concurrent connection monitoring
- IP-based violation tracking
- Abuse pattern detection

### System Metrics
- CPU, memory, and disk usage
- GPU metrics (if available)
- Container resource utilization
- Network statistics

### Custom Metrics Exposed
```
# Rate limiting metrics
kokorotts_requests_total{ip, endpoint, status}
kokorotts_requests_blocked_total{ip, limit_type, reason}
kokorotts_characters_processed_total{ip}
kokorotts_concurrent_requests
kokorotts_redis_connected

# System metrics
kokorotts_cpu_percent
kokorotts_memory_percent
kokorotts_gpu_memory_used_mb
kokorotts_gpu_memory_total_mb
```

## Dashboard Features

### Main Dashboard Panels
1. **Request Rate & Blocks** - Real-time request and blocking rates
2. **Block Rate (%)** - Percentage of requests being blocked
3. **Character Usage** - Character processing vs. limits
4. **Concurrent Connections** - Active connections vs. limits
5. **Top Rate Limited IPs** - Most frequently blocked IPs
6. **Rate Limit Types** - Distribution of block reasons
7. **Redis Connection Status** - Redis connectivity health

### Custom Queries
```promql
# Request rate by IP
rate(kokorotts_requests_total[1m]) by (ip)

# Block rate percentage
(rate(kokorotts_requests_blocked_total[5m]) / rate(kokorotts_requests_total[5m])) * 100

# Character usage per hour
rate(kokorotts_characters_processed_total[1h]) * 3600

# Top violating IPs
topk(10, sum by (ip) (rate(kokorotts_requests_blocked_total[1h])))
```

## Alerting Rules

### Critical Alerts
- **Service Down** - API service unavailable
- **Redis Connection Lost** - Rate limiting impaired
- **Critical Rate Limit Violations** - >50% requests blocked
- **Potential DDoS Attack** - >1000 requests/minute

### Warning Alerts
- **High Rate Limit Violations** - >25% requests blocked
- **High CPU/Memory Usage** - Resource constraints
- **Character Usage Spike** - Unusual character processing
- **Single IP High Request Rate** - Potential abuse

### Alert Channels
- Email notifications
- Slack integration
- Webhook endpoints
- PagerDuty integration (configurable)

## Configuration

### Prometheus Targets
Add your KokoroTTS API instances to `prometheus-config.yml`:
```yaml
scrape_configs:
  - job_name: 'kokorotts-api'
    static_configs:
      - targets: 
        - 'kokorotts-api-1:8000'
        - 'kokorotts-api-2:8000'
        - 'kokorotts-api-3:8000'
```

### Alert Notifications
Update `alertmanager.yml` with your notification preferences:
```yaml
receivers:
  - name: 'critical-alerts'
    email_configs:
      - to: 'your-email@domain.com'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK'
        channel: '#alerts'
```

### Custom Dashboards
1. Import the provided dashboard JSON in Grafana
2. Customize panels for your specific needs
3. Add additional data sources as needed

## Production Deployment

### High Availability Setup
```bash
# Use external storage for Prometheus
# Configure Grafana with external database
# Set up AlertManager clustering
```

### Security Considerations
- Change default Grafana password
- Configure authentication (LDAP, OAuth)
- Use TLS certificates
- Restrict network access
- Regular backup of configurations

### Performance Tuning
```yaml
# Prometheus retention
--storage.tsdb.retention.time=30d

# Scrape intervals
scrape_interval: 15s  # General metrics
scrape_interval: 5s   # Rate limiting metrics (more frequent)

# Memory optimization
--storage.tsdb.retention.size=10GB
```

## Troubleshooting

### Common Issues

**Metrics not appearing:**
```bash
# Check if API is exposing metrics
curl http://localhost:8000/metrics

# Verify Prometheus can reach the API
curl http://localhost:9090/api/v1/targets
```

**Grafana not showing data:**
```bash
# Check Prometheus datasource
curl http://localhost:9090/api/v1/query?query=up

# Verify dashboard queries
```

**Alerts not firing:**
```bash
# Check alert rules
curl http://localhost:9090/api/v1/rules

# Verify AlertManager configuration
curl http://localhost:9093/api/v1/status
```

### Log Analysis
```bash
# View container logs
docker logs kokorotts-prometheus
docker logs kokorotts-grafana
docker logs kokorotts-alertmanager

# Monitor in real-time
docker logs -f kokorotts-prometheus
```

## API Integration

### Custom Metrics Endpoint
The KokoroTTS API exposes additional monitoring endpoints:

```bash
# Enhanced system metrics with rate limiting data
GET /metrics

# Admin endpoints (when rate limiting enabled)
GET /admin/rate-limits/usage?hours=24
GET /admin/rate-limits/violations?hours=1
GET /admin/rate-limits/ip/192.168.1.100
GET /admin/rate-limits/abuse-patterns
GET /admin/rate-limits/export?format=json
GET /admin/rate-limits/config
```

### Programmatic Access
```python
import requests

# Get usage statistics
response = requests.get('http://localhost:8000/admin/rate-limits/usage?hours=1')
stats = response.json()

# Check for abuse patterns
response = requests.get('http://localhost:8000/admin/rate-limits/abuse-patterns')
patterns = response.json()

# Export monitoring data
response = requests.get('http://localhost:8000/admin/rate-limits/export?format=csv')
data = response.text
```

## Extending Monitoring

### Adding Custom Metrics
1. Modify the rate limiting middleware to expose new metrics
2. Update Prometheus configuration to scrape them
3. Create dashboard panels for visualization
4. Add alerting rules as needed

### Integration Examples
- **ELK Stack** - For log analysis
- **Jaeger** - For distributed tracing
- **New Relic/DataDog** - For APM
- **PagerDuty** - For incident management