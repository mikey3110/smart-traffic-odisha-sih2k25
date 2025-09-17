#!/bin/bash

# Smart Traffic Management System - Production Deployment Script
# This script handles the complete deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="smart-traffic-odisha-sih2k25"
ENVIRONMENT=${1:-production}
DOCKER_REGISTRY="ghcr.io"
NAMESPACE="smart-traffic-${ENVIRONMENT}"

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    success "Docker is available"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
        exit 1
    fi
    success "kubectl is available"
    
    # Check helm (optional)
    if command -v helm &> /dev/null; then
        success "Helm is available"
    else
        warning "Helm is not installed (optional)"
    fi
    
    # Check environment variables
    if [ -z "$POSTGRES_PASSWORD" ]; then
        error "POSTGRES_PASSWORD environment variable is not set"
        exit 1
    fi
    success "Required environment variables are set"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    services=("backend" "frontend" "ml-engine" "cv-service")
    
    for service in "${services[@]}"; do
        log "Building ${service}..."
        docker build -t "${DOCKER_REGISTRY}/${PROJECT_NAME}/${service}:latest" \
                     -f "src/${service}/Dockerfile" .
        success "Built ${service} image"
    done
}

# Push images to registry
push_images() {
    log "Pushing images to registry..."
    
    services=("backend" "frontend" "ml-engine" "cv-service")
    
    for service in "${services[@]}"; do
        log "Pushing ${service}..."
        docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}/${service}:latest"
        success "Pushed ${service} image"
    done
}

# Create namespace
create_namespace() {
    log "Creating namespace ${NAMESPACE}..."
    
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    success "Namespace ${NAMESPACE} created/updated"
}

# Deploy secrets
deploy_secrets() {
    log "Deploying secrets..."
    
    # Create secrets from environment variables
    kubectl create secret generic smart-traffic-secrets \
        --from-literal=postgres-password="${POSTGRES_PASSWORD}" \
        --from-literal=redis-password="${REDIS_PASSWORD:-}" \
        --from-literal=jwt-secret="${JWT_SECRET:-$(openssl rand -base64 32)}" \
        --namespace="${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    success "Secrets deployed"
}

# Deploy configmaps
deploy_configmaps() {
    log "Deploying configmaps..."
    
    # Create configmap for application configuration
    kubectl create configmap smart-traffic-config \
        --from-literal=environment="${ENVIRONMENT}" \
        --from-literal=log-level="INFO" \
        --from-literal=postgres-host="postgres" \
        --from-literal=redis-host="redis" \
        --namespace="${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    success "ConfigMaps deployed"
}

# Deploy database
deploy_database() {
    log "Deploying PostgreSQL database..."
    
    # Deploy PostgreSQL
    kubectl apply -f k8s/postgres-deployment.yaml -n "${NAMESPACE}"
    kubectl apply -f k8s/postgres-service.yaml -n "${NAMESPACE}"
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n "${NAMESPACE}" --timeout=300s
    
    success "Database deployed and ready"
}

# Deploy Redis
deploy_redis() {
    log "Deploying Redis cache..."
    
    kubectl apply -f k8s/redis-deployment.yaml -n "${NAMESPACE}"
    kubectl apply -f k8s/redis-service.yaml -n "${NAMESPACE}"
    
    # Wait for Redis to be ready
    log "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n "${NAMESPACE}" --timeout=300s
    
    success "Redis deployed and ready"
}

# Deploy backend services
deploy_backend_services() {
    log "Deploying backend services..."
    
    # Deploy Backend API
    kubectl apply -f k8s/backend-deployment.yaml -n "${NAMESPACE}"
    kubectl apply -f k8s/backend-service.yaml -n "${NAMESPACE}"
    
    # Deploy ML API
    kubectl apply -f k8s/ml-api-deployment.yaml -n "${NAMESPACE}"
    kubectl apply -f k8s/ml-api-service.yaml -n "${NAMESPACE}"
    
    # Deploy CV Service
    kubectl apply -f k8s/cv-service-deployment.yaml -n "${NAMESPACE}"
    kubectl apply -f k8s/cv-service.yaml -n "${NAMESPACE}"
    
    # Wait for services to be ready
    log "Waiting for backend services to be ready..."
    kubectl wait --for=condition=ready pod -l app=backend -n "${NAMESPACE}" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=ml-api -n "${NAMESPACE}" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=cv-service -n "${NAMESPACE}" --timeout=300s
    
    success "Backend services deployed and ready"
}

# Deploy frontend
deploy_frontend() {
    log "Deploying frontend..."
    
    kubectl apply -f k8s/frontend-deployment.yaml -n "${NAMESPACE}"
    kubectl apply -f k8s/frontend-service.yaml -n "${NAMESPACE}"
    
    # Wait for frontend to be ready
    log "Waiting for frontend to be ready..."
    kubectl wait --for=condition=ready pod -l app=frontend -n "${NAMESPACE}" --timeout=300s
    
    success "Frontend deployed and ready"
}

# Deploy ingress
deploy_ingress() {
    log "Deploying ingress..."
    
    kubectl apply -f k8s/ingress.yaml -n "${NAMESPACE}"
    
    success "Ingress deployed"
}

# Deploy monitoring
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Deploy Prometheus
    kubectl apply -f k8s/prometheus-deployment.yaml -n "${NAMESPACE}"
    kubectl apply -f k8s/prometheus-service.yaml -n "${NAMESPACE}"
    
    # Deploy Grafana
    kubectl apply -f k8s/grafana-deployment.yaml -n "${NAMESPACE}"
    kubectl apply -f k8s/grafana-service.yaml -n "${NAMESPACE}"
    
    success "Monitoring stack deployed"
}

# Run health checks
run_health_checks() {
    log "Running health checks..."
    
    # Get service URLs
    BACKEND_URL=$(kubectl get service backend -n "${NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    FRONTEND_URL=$(kubectl get service frontend -n "${NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$BACKEND_URL" ]; then
        BACKEND_URL="localhost:8000"
    fi
    
    if [ -z "$FRONTEND_URL" ]; then
        FRONTEND_URL="localhost:3000"
    fi
    
    # Test backend health
    log "Testing backend health..."
    if curl -f "http://${BACKEND_URL}/health" > /dev/null 2>&1; then
        success "Backend health check passed"
    else
        error "Backend health check failed"
        return 1
    fi
    
    # Test frontend
    log "Testing frontend..."
    if curl -f "http://${FRONTEND_URL}" > /dev/null 2>&1; then
        success "Frontend health check passed"
    else
        error "Frontend health check failed"
        return 1
    fi
    
    success "All health checks passed"
}

# Run smoke tests
run_smoke_tests() {
    log "Running smoke tests..."
    
    # Run basic smoke tests
    python tests/smoke_test.py --environment="${ENVIRONMENT}"
    
    success "Smoke tests passed"
}

# Display deployment status
show_status() {
    log "Deployment Status:"
    echo ""
    
    # Show pods
    echo "📦 Pods:"
    kubectl get pods -n "${NAMESPACE}"
    echo ""
    
    # Show services
    echo "🌐 Services:"
    kubectl get services -n "${NAMESPACE}"
    echo ""
    
    # Show ingress
    echo "🚪 Ingress:"
    kubectl get ingress -n "${NAMESPACE}"
    echo ""
    
    # Show access URLs
    echo "🔗 Access URLs:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend API: http://localhost:8000"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  ML API: http://localhost:8001"
    echo "  CV Service: http://localhost:5001"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana: http://localhost:3001"
    echo ""
}

# Main deployment function
deploy() {
    log "Starting deployment to ${ENVIRONMENT} environment..."
    
    check_prerequisites
    build_images
    push_images
    create_namespace
    deploy_secrets
    deploy_configmaps
    deploy_database
    deploy_redis
    deploy_backend_services
    deploy_frontend
    deploy_ingress
    deploy_monitoring
    
    # Wait a bit for everything to settle
    log "Waiting for services to stabilize..."
    sleep 30
    
    run_health_checks
    run_smoke_tests
    show_status
    
    success "Deployment completed successfully! 🎉"
}

# Rollback function
rollback() {
    log "Rolling back deployment..."
    
    # Delete the namespace (this will remove all resources)
    kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true
    
    success "Rollback completed"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "rollback")
        rollback
        ;;
    "status")
        show_status
        ;;
    "health")
        run_health_checks
        ;;
    "smoke")
        run_smoke_tests
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|status|health|smoke}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the application (default)"
        echo "  rollback - Rollback the deployment"
        echo "  status   - Show deployment status"
        echo "  health   - Run health checks"
        echo "  smoke    - Run smoke tests"
        exit 1
        ;;
esac