# Smart Traffic Management System - Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Development Setup](#development-setup)
4. [Production Deployment](#production-deployment)
5. [Configuration](#configuration)
6. [Monitoring Setup](#monitoring-setup)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance](#maintenance)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended (16GB+ for production)
- **Storage**: 50GB+ free space
- **Network**: Stable internet connection

### Software Requirements

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Python**: 3.11+
- **Node.js**: 18+
- **Git**: 2.30+

### Optional Requirements

- **Kubernetes**: 1.20+ (for production)
- **Helm**: 3.0+ (for Kubernetes deployment)
- **Prometheus**: 2.30+ (for monitoring)
- **Grafana**: 8.0+ (for dashboards)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/smart-traffic-management.git
cd smart-traffic-management
```

### 2. Environment Setup

```bash
# Copy environment configuration
cp .env.example .env

# Edit configuration
nano .env
```

### 3. Start the System

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Access the System

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Admin Interface**: http://localhost:8003
- **Grafana**: http://localhost:3001 (admin/admin123)
- **Prometheus**: http://localhost:9090

## Development Setup

### 1. Backend Development

```bash
# Navigate to backend directory
cd src/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend Development

```bash
# Navigate to frontend directory
cd src/frontend/smart-traffic-ui

# Install dependencies
npm install

# Start development server
npm run dev
```

### 3. ML Optimizer Development

```bash
# Navigate to ML engine directory
cd src/ml_engine

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### 4. SUMO Simulation Development

```bash
# Install SUMO
sudo apt-get update
sudo apt-get install sumo sumo-tools

# Navigate to simulation directory
cd src/simulation/sumo_integration

# Install Python dependencies
pip install -r requirements.txt

# Run simulation
python run_sumo_integration.py
```

## Production Deployment

### 1. Docker Compose Deployment

```bash
# Build and start services
docker-compose -f docker-compose.prod.yml up -d

# Scale services if needed
docker-compose up -d --scale backend=3

# Update services
docker-compose pull
docker-compose up -d
```

### 2. Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace traffic-management

# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/services/
kubectl apply -f k8s/ingress.yaml

# Check deployment status
kubectl get pods -n traffic-management
kubectl get services -n traffic-management
```

### 3. Helm Deployment

```bash
# Add Helm repository
helm repo add traffic-management ./helm-chart
helm repo update

# Install chart
helm install traffic-management ./helm-chart \
  --namespace traffic-management \
  --create-namespace \
  --values values/production.yaml

# Upgrade deployment
helm upgrade traffic-management ./helm-chart \
  --namespace traffic-management \
  --values values/production.yaml
```

## Configuration

### 1. Environment Variables

Create a `.env` file with the following variables:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/traffic_management
REDIS_URL=redis://localhost:6379

# API Configuration
JWT_SECRET=your-super-secret-jwt-key
API_VERSION=v1
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
GRAFANA_ADMIN_PASSWORD=admin123

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Slack Configuration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

### 2. System Configuration

Edit `config/system_config.yaml`:

```yaml
system:
  name: "Smart Traffic Management System"
  version: "2.1.0"
  environment: "production"
  log_level: "INFO"

components:
  backend:
    port: 8000
    health_check_url: "http://localhost:8000/health"
    max_restarts: 5

  # ... other components
```

### 3. Database Configuration

```bash
# Create database
createdb traffic_management

# Run migrations
python src/backend/manage.py migrate

# Create superuser
python src/backend/manage.py createsuperuser
```

## Monitoring Setup

### 1. Prometheus Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'traffic-management'
    static_configs:
      - targets: ['backend:8000', 'ml-optimizer:8001', 'sumo-simulation:8002']
```

### 2. Grafana Dashboards

```bash
# Import dashboards
curl -X POST \
  http://admin:admin123@localhost:3001/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/dashboards/traffic-overview.json
```

### 3. Log Aggregation

```bash
# Start ELK stack
docker-compose -f docker-compose.monitoring.yml up -d

# Configure log shipping
# Logs will be automatically shipped to Elasticsearch
```

## Troubleshooting

### 1. Common Issues

#### Service Won't Start

```bash
# Check logs
docker-compose logs service-name

# Check resource usage
docker stats

# Restart service
docker-compose restart service-name
```

#### Database Connection Issues

```bash
# Check database status
docker-compose exec postgres pg_isready

# Check connection string
echo $DATABASE_URL

# Test connection
python -c "import psycopg2; psycopg2.connect('$DATABASE_URL')"
```

#### Frontend Build Issues

```bash
# Clear node modules
rm -rf node_modules package-lock.json

# Reinstall dependencies
npm install

# Clear build cache
npm run build -- --no-cache
```

### 2. Health Checks

```bash
# Check all services
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:3000

# Check system status
curl http://localhost:8003/api/status
```

### 3. Performance Issues

```bash
# Check resource usage
docker stats

# Check database performance
docker-compose exec postgres psql -U traffic_user -d traffic_management -c "SELECT * FROM pg_stat_activity;"

# Check Redis performance
docker-compose exec redis redis-cli info memory
```

## Maintenance

### 1. Regular Maintenance Tasks

#### Daily
- Check system health
- Review alerts and logs
- Monitor resource usage
- Verify backups

#### Weekly
- Update dependencies
- Review performance metrics
- Clean up old logs
- Test disaster recovery

#### Monthly
- Security updates
- Performance optimization
- Capacity planning
- Documentation updates

### 2. Backup Procedures

```bash
# Database backup
docker-compose exec postgres pg_dump -U traffic_user traffic_management > backup_$(date +%Y%m%d).sql

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/

# Full system backup
docker-compose down
tar -czf system_backup_$(date +%Y%m%d).tar.gz .
docker-compose up -d
```

### 3. Update Procedures

```bash
# Update system
git pull origin main
docker-compose pull
docker-compose up -d

# Rollback if needed
git checkout previous-version
docker-compose up -d
```

### 4. Monitoring and Alerting

#### Set up alerts for:
- High CPU usage (>80%)
- High memory usage (>85%)
- High disk usage (>90%)
- Service failures
- Database connection issues
- API response time >5s

#### Notification channels:
- Email alerts
- Slack notifications
- PagerDuty integration
- SMS alerts (critical only)

## Security Considerations

### 1. Network Security
- Use HTTPS in production
- Configure firewall rules
- Implement VPN access
- Regular security audits

### 2. Data Security
- Encrypt sensitive data
- Regular security updates
- Access control and auditing
- Backup encryption

### 3. Application Security
- Input validation
- SQL injection prevention
- XSS protection
- CSRF protection

## Performance Optimization

### 1. Database Optimization
- Index optimization
- Query optimization
- Connection pooling
- Read replicas

### 2. Caching Strategy
- Redis caching
- CDN for static assets
- Application-level caching
- Database query caching

### 3. Load Balancing
- Nginx load balancer
- Health checks
- Session persistence
- Auto-scaling

## Disaster Recovery

### 1. Backup Strategy
- Daily database backups
- Configuration backups
- Code repository backups
- Off-site storage

### 2. Recovery Procedures
- Database restoration
- Service recovery
- Data validation
- System testing

### 3. Business Continuity
- Redundant systems
- Failover procedures
- Communication plans
- Recovery time objectives

## Support and Documentation

### 1. Getting Help
- Check documentation
- Review logs and metrics
- Contact support team
- Create issue tickets

### 2. Documentation
- API documentation
- User guides
- Admin guides
- Troubleshooting guides

### 3. Training
- System administration
- User training
- Developer onboarding
- Best practices
