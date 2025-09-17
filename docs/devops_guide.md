# DevOps Guide - Smart Traffic Management System

## Overview

This guide provides comprehensive instructions for deploying, monitoring, and maintaining the Smart Traffic Management System in production and demo environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Production Deployment](#production-deployment)
4. [Demo Environment](#demo-environment)
5. [Monitoring & Logging](#monitoring--logging)
6. [Backup & Recovery](#backup--recovery)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)

## Prerequisites

### System Requirements

- **Kubernetes Cluster**: v1.20+
- **Docker**: v20.10+
- **kubectl**: v1.20+
- **Helm**: v3.0+ (optional)
- **Minikube**: v1.20+ (for local development)

### Required Tools

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm (optional)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install Minikube (for local development)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

## Quick Start

### 1. Local Development with Minikube

```bash
# Start Minikube
minikube start --memory=4096 --cpus=2

# Enable required addons
minikube addons enable ingress
minikube addons enable metrics-server

# Deploy the application
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/demo.yaml

# Check deployment status
kubectl get pods -n smart-traffic-demo

# Get external URLs
minikube service list -n smart-traffic-demo
```

### 2. Access the Application

```bash
# Get service URLs
kubectl get services -n smart-traffic-demo

# Access frontend
minikube service frontend-demo -n smart-traffic-demo

# Access backend API
minikube service backend-demo -n smart-traffic-demo

# Access SUMO GUI
minikube service sumo-demo -n smart-traffic-demo
```

## Production Deployment

### 1. Create Production Namespace

```bash
kubectl create namespace smart-traffic-prod
kubectl label namespace smart-traffic-prod environment=production
```

### 2. Deploy Core Services

```bash
# Deploy PostgreSQL
kubectl apply -f k8s/postgres-deployment.yaml

# Deploy Redis
kubectl apply -f k8s/redis-deployment.yaml

# Deploy Backend API
kubectl apply -f k8s/backend-deployment.yaml

# Deploy Frontend
kubectl apply -f k8s/frontend-deployment.yaml

# Deploy ML API
kubectl apply -f k8s/ml-api-deployment.yaml

# Deploy CV Service
kubectl apply -f k8s/cv-service-deployment.yaml
```

### 3. Configure Ingress

```bash
# Deploy Ingress
kubectl apply -f k8s/ingress.yaml

# Check ingress status
kubectl get ingress -n smart-traffic-prod
```

### 4. Deploy Monitoring

```bash
# Deploy Prometheus and Grafana
kubectl apply -f k8s/monitoring-deployment.yaml

# Access Grafana
kubectl port-forward svc/grafana 3000:80 -n smart-traffic-prod
```

## Demo Environment

### 1. Deploy Demo Environment

```bash
# Deploy demo environment
kubectl apply -f k8s/demo.yaml

# Wait for all pods to be ready
kubectl wait --for=condition=ready pod -l app=backend-demo -n smart-traffic-demo --timeout=300s
kubectl wait --for=condition=ready pod -l app=frontend-demo -n smart-traffic-demo --timeout=300s
```

### 2. Access Demo Services

| Service | URL | Port | Description |
|---------|-----|------|-------------|
| Frontend | http://localhost:30000 | 30000 | Main dashboard |
| Backend API | http://localhost:30080 | 30080 | REST API |
| ML API | http://localhost:30081 | 30081 | ML services |
| CV Service | http://localhost:30082 | 30082 | Computer vision |
| SUMO GUI | http://localhost:30083 | 30083 | Traffic simulation |

### 3. Demo Startup Script

```bash
#!/bin/bash
# demo_start.sh

echo "Starting Smart Traffic Management Demo..."

# Deploy demo environment
kubectl apply -f k8s/demo.yaml

# Wait for services to be ready
echo "Waiting for services to start..."
kubectl wait --for=condition=ready pod -l app=backend-demo -n smart-traffic-demo --timeout=300s
kubectl wait --for=condition=ready pod -l app=frontend-demo -n smart-traffic-demo --timeout=300s

# Run smoke tests
echo "Running smoke tests..."
./scripts/smoke_test.sh --quick

# Display access information
echo "Demo is ready!"
echo "Frontend: http://localhost:30000"
echo "Backend API: http://localhost:30080"
echo "SUMO GUI: http://localhost:30083"
```

## Monitoring & Logging

### 1. Health Checks

```bash
# Run comprehensive health checks
./scripts/smoke_test.sh

# Check specific service health
kubectl get pods -n smart-traffic-prod
kubectl describe pod <pod-name> -n smart-traffic-prod
```

### 2. Logs

```bash
# View application logs
kubectl logs -f deployment/backend-demo -n smart-traffic-demo
kubectl logs -f deployment/frontend-demo -n smart-traffic-demo
kubectl logs -f deployment/ml-api-demo -n smart-traffic-demo

# View logs from all containers
kubectl logs -f -l app=backend-demo -n smart-traffic-demo
```

### 3. Metrics

```bash
# Access Prometheus
kubectl port-forward svc/prometheus 9090:80 -n smart-traffic-prod

# Access Grafana
kubectl port-forward svc/grafana 3000:80 -n smart-traffic-prod
```

### 4. Resource Usage

```bash
# Check resource usage
kubectl top pods -n smart-traffic-prod
kubectl top nodes

# Check resource limits
kubectl describe pod <pod-name> -n smart-traffic-prod
```

## Backup & Recovery

### 1. Automated Backups

```bash
# Create backup
./scripts/backup.sh backup

# Create backup with S3 upload
./scripts/backup.sh backup --s3

# List available backups
./scripts/backup.sh list

# Restore from backup
./scripts/backup.sh restore smart-traffic-backup-20241219_143022
```

### 2. Database Backup

```bash
# Manual database backup
kubectl exec -it deployment/postgres-demo -n smart-traffic-demo -- pg_dump -U traffic_user traffic_management > backup.sql

# Restore database
kubectl exec -i deployment/postgres-demo -n smart-traffic-demo -- psql -U traffic_user traffic_management < backup.sql
```

### 3. Model Backup

```bash
# Backup ML models
kubectl cp smart-traffic-demo/ml-api-demo-xxx:/app/models ./models-backup/

# Restore ML models
kubectl cp ./models-backup/ smart-traffic-demo/ml-api-demo-xxx:/app/models/
```

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl get pods -n smart-traffic-demo

# Check pod events
kubectl describe pod <pod-name> -n smart-traffic-demo

# Check logs
kubectl logs <pod-name> -n smart-traffic-demo
```

#### 2. Services Not Accessible

```bash
# Check service endpoints
kubectl get endpoints -n smart-traffic-demo

# Check service configuration
kubectl describe service <service-name> -n smart-traffic-demo

# Test service connectivity
kubectl run test-pod --image=busybox -it --rm -- nslookup <service-name>
```

#### 3. Database Connection Issues

```bash
# Check database status
kubectl exec -it deployment/postgres-demo -n smart-traffic-demo -- pg_isready -U traffic_user

# Check database logs
kubectl logs deployment/postgres-demo -n smart-traffic-demo

# Test database connection
kubectl exec -it deployment/backend-demo -n smart-traffic-demo -- python -c "import psycopg2; print('DB connection OK')"
```

#### 4. Resource Issues

```bash
# Check resource usage
kubectl top pods -n smart-traffic-demo

# Check resource limits
kubectl describe pod <pod-name> -n smart-traffic-demo

# Scale deployment
kubectl scale deployment backend-demo --replicas=3 -n smart-traffic-demo
```

### Debug Commands

```bash
# Get detailed pod information
kubectl get pods -o wide -n smart-traffic-demo

# Check node resources
kubectl describe nodes

# Check persistent volumes
kubectl get pv,pvc -n smart-traffic-demo

# Check ingress status
kubectl get ingress -n smart-traffic-demo
kubectl describe ingress <ingress-name> -n smart-traffic-demo
```

## Performance Tuning

### 1. Resource Optimization

```yaml
# Example resource configuration
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### 2. Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend-demo
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 3. Database Optimization

```bash
# Check database performance
kubectl exec -it deployment/postgres-demo -n smart-traffic-demo -- psql -U traffic_user -d traffic_management -c "SELECT * FROM pg_stat_activity;"

# Optimize database
kubectl exec -it deployment/postgres-demo -n smart-traffic-demo -- psql -U traffic_user -d traffic_management -c "VACUUM ANALYZE;"
```

## Security Considerations

### 1. Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: smart-traffic-netpol
spec:
  podSelector:
    matchLabels:
      app: backend-demo
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend-demo
    ports:
    - protocol: TCP
      port: 8000
```

### 2. Secrets Management

```bash
# Create secrets
kubectl create secret generic postgres-secret \
  --from-literal=username=traffic_user \
  --from-literal=password=secure_password \
  -n smart-traffic-prod

# Use secrets in deployment
env:
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: postgres-secret
      key: password
```

### 3. RBAC Configuration

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: smart-traffic-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
```

## Maintenance

### 1. Rolling Updates

```bash
# Update deployment
kubectl set image deployment/backend-demo backend=smart-traffic-backend:v2.0.0 -n smart-traffic-demo

# Check rollout status
kubectl rollout status deployment/backend-demo -n smart-traffic-demo

# Rollback if needed
kubectl rollout undo deployment/backend-demo -n smart-traffic-demo
```

### 2. Cleanup

```bash
# Clean up demo environment
kubectl delete namespace smart-traffic-demo

# Clean up old resources
kubectl delete pods --field-selector=status.phase=Succeeded
kubectl delete pods --field-selector=status.phase=Failed
```

### 3. Monitoring Maintenance

```bash
# Check Prometheus targets
kubectl port-forward svc/prometheus 9090:80 -n smart-traffic-prod
# Visit http://localhost:9090/targets

# Check Grafana dashboards
kubectl port-forward svc/grafana 3000:80 -n smart-traffic-prod
# Visit http://localhost:3000
```

## Support

### Emergency Contacts

- **Team Lead**: For deployment issues
- **Backend Dev**: For API problems
- **ML Engineer**: For optimizer issues
- **Frontend Dev**: For UI deployment

### Useful Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

---

**DevOps Guide v1.0 - Smart India Hackathon 2025**

*For additional support, please contact the development team or create an issue in the repository.*
