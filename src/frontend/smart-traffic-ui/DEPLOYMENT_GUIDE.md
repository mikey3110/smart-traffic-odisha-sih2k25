# Deployment Guide - Smart Traffic Management Frontend

This guide covers all deployment options for the Smart Traffic Management System frontend.

## 🐳 Docker Deployment

### Prerequisites

- Docker installed
- Docker Compose (optional)

### Build Docker Image

```bash
# Build the image
npm run docker:build

# Or manually
docker build -t smart-traffic-frontend .
```

### Run Docker Container

```bash
# Run the container
npm run docker:run

# Or manually
docker run -p 3000:3000 smart-traffic-frontend
```

### Docker Compose

For local development with mock backend:

```bash
# Start all services
npm run docker:compose

# Stop services
npm run docker:compose:down
```

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster
- kubectl configured
- Docker image pushed to registry

### Deploy to Kubernetes

```bash
# Deploy all resources
npm run k8s:deploy

# Check deployment status
npm run k8s:status

# Delete deployment
npm run k8s:delete
```

### Manual Deployment

```bash
# Create namespace
kubectl create namespace smart-traffic

# Deploy frontend
kubectl apply -f k8s/frontend-deployment.yaml

# Check pods
kubectl get pods -n smart-traffic

# Check services
kubectl get services -n smart-traffic
```

### Configuration

The Kubernetes deployment includes:

- **Deployment**: 3 replicas with resource limits
- **Service**: ClusterIP for internal communication
- **Ingress**: External access configuration
- **Health Checks**: Liveness and readiness probes
- **Security**: Non-root user and security context

## 🌐 Production Deployment

### Environment Setup

1. **Set environment variables**:
   ```bash
   export VITE_API_BASE_URL=https://api.smarttraffic.com
   export VITE_WS_URL=wss://api.smarttraffic.com
   ```

2. **Build for production**:
   ```bash
   npm run build
   ```

3. **Deploy to your hosting platform**:
   - Copy `dist/` contents to your web server
   - Configure reverse proxy for API calls
   - Set up SSL certificates

### Nginx Configuration

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name smarttraffic.com;
    root /var/www/smart-traffic;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/css application/javascript application/json;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;

    # API proxy
    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # SPA routing
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

## 🔄 CI/CD Pipeline

### GitHub Actions

The project includes a complete CI/CD pipeline:

1. **Test Stage**: Runs unit tests, linting, and type checking
2. **Build Stage**: Builds Docker image
3. **Deploy Stage**: Deploys to Kubernetes

### Pipeline Triggers

- **Push to main**: Full deployment
- **Pull Request**: Test and build only
- **Push to develop**: Test and build

### Manual Deployment

```bash
# Trigger deployment via GitHub Actions
gh workflow run ci-cd.yml

# Or deploy manually
npm run k8s:deploy
```

## 📊 Monitoring

### Health Checks

The application includes health check endpoints:

- **Liveness**: `/health` - Basic health check
- **Readiness**: `/health` - Application ready check

### Metrics

Monitor the following metrics:

- **Response Time**: API response times
- **Error Rate**: 4xx and 5xx error rates
- **Throughput**: Requests per second
- **Resource Usage**: CPU and memory usage

### Logging

Logs are available through:

- **Container Logs**: `kubectl logs -f deployment/smart-traffic-frontend`
- **Application Logs**: Check browser console for client-side logs

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API URL | `http://localhost:8000` |
| `VITE_WS_URL` | WebSocket URL | `ws://localhost:8000` |
| `NODE_ENV` | Environment | `production` |

### Resource Limits

Kubernetes resource limits:

```yaml
resources:
  requests:
    memory: "128Mi"
    cpu: "100m"
  limits:
    memory: "256Mi"
    cpu: "200m"
```

## 🚨 Troubleshooting

### Common Issues

1. **Container won't start**
   ```bash
   # Check logs
   docker logs <container-id>
   
   # Check resource usage
   docker stats
   ```

2. **Kubernetes deployment fails**
   ```bash
   # Check pod status
   kubectl describe pod <pod-name> -n smart-traffic
   
   # Check events
   kubectl get events -n smart-traffic
   ```

3. **Health checks failing**
   ```bash
   # Check health endpoint
   curl http://localhost:3000/health
   
   # Check pod logs
   kubectl logs <pod-name> -n smart-traffic
   ```

### Debug Commands

```bash
# Docker debug
docker exec -it <container-id> /bin/sh

# Kubernetes debug
kubectl exec -it <pod-name> -n smart-traffic -- /bin/sh

# Check service connectivity
kubectl port-forward service/smart-traffic-frontend 3000:3000 -n smart-traffic
```

## 📈 Scaling

### Horizontal Scaling

Scale the deployment:

```bash
# Scale to 5 replicas
kubectl scale deployment smart-traffic-frontend --replicas=5 -n smart-traffic

# Auto-scaling (if HPA is configured)
kubectl autoscale deployment smart-traffic-frontend --min=3 --max=10 -n smart-traffic
```

### Vertical Scaling

Update resource limits in `k8s/frontend-deployment.yaml`:

```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "200m"
  limits:
    memory: "512Mi"
    cpu: "400m"
```

## 🔒 Security

### Security Headers

The application includes security headers:

- `X-Frame-Options: SAMEORIGIN`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: no-referrer-when-downgrade`

### Container Security

- Non-root user (UID 1001)
- Read-only root filesystem
- Minimal attack surface
- Regular security updates

## 📋 Checklist

### Pre-deployment

- [ ] All tests passing
- [ ] Code linted and formatted
- [ ] Docker image built successfully
- [ ] Environment variables configured
- [ ] Health checks working

### Post-deployment

- [ ] Application accessible
- [ ] API calls working
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Logs accessible

## 🆘 Support

For deployment issues:

1. Check the troubleshooting section
2. Review container logs
3. Check Kubernetes events
4. Verify network connectivity
5. Contact the development team

---

**Deployment Guide v1.0 - Smart India Hackathon 2025**
