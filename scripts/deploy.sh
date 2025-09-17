#!/bin/bash

# Production Deployment Script
# Smart Traffic Management System - Phase 4

set -e

# Configuration
NAMESPACE="traffic-ml"
REGISTRY="your-registry.com"
VERSION="v1.0.0"
ENVIRONMENT="production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_warning "helm is not installed, some features may not work"
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl apply -f k8s/namespace.yaml
        log_success "Namespace $NAMESPACE created"
    fi
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets already exist
    if kubectl get secret postgres-secret -n $NAMESPACE &> /dev/null; then
        log_warning "Secrets already exist, skipping..."
        return
    fi
    
    kubectl apply -f k8s/secrets.yaml
    log_success "Secrets deployed"
}

# Deploy persistent volumes
deploy_persistent_volumes() {
    log_info "Deploying persistent volumes..."
    
    kubectl apply -f k8s/persistent-volumes.yaml
    log_success "Persistent volumes deployed"
}

# Deploy database
deploy_database() {
    log_info "Deploying PostgreSQL database..."
    
    kubectl apply -f k8s/postgres-deployment.yaml
    
    # Wait for database to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s
    
    log_success "PostgreSQL deployed and ready"
}

# Deploy Redis
deploy_redis() {
    log_info "Deploying Redis cache..."
    
    kubectl apply -f k8s/redis-deployment.yaml
    
    # Wait for Redis to be ready
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s
    
    log_success "Redis deployed and ready"
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    # Build ML API image
    log_info "Building ML API image..."
    docker build -t $REGISTRY/traffic-ml-api:$VERSION -f src/ml_engine/Dockerfile.ml-api .
    docker push $REGISTRY/traffic-ml-api:$VERSION
    
    # Build Streaming API image
    log_info "Building Streaming API image..."
    docker build -t $REGISTRY/traffic-streaming-api:$VERSION -f src/ml_engine/Dockerfile.streaming .
    docker push $REGISTRY/traffic-streaming-api:$VERSION
    
    # Build ML Engine image
    log_info "Building ML Engine image..."
    docker build -t $REGISTRY/traffic-ml-engine:$VERSION -f src/ml_engine/Dockerfile.ml-engine .
    docker push $REGISTRY/traffic-ml-engine:$VERSION
    
    log_success "Docker images built and pushed"
}

# Deploy ML services
deploy_ml_services() {
    log_info "Deploying ML services..."
    
    # Update image tags in deployment files
    sed -i "s|image: traffic-ml-api:latest|image: $REGISTRY/traffic-ml-api:$VERSION|g" k8s/ml-api-deployment.yaml
    sed -i "s|image: traffic-streaming-api:latest|image: $REGISTRY/traffic-streaming-api:$VERSION|g" k8s/streaming-api-deployment.yaml
    sed -i "s|image: traffic-ml-engine:latest|image: $REGISTRY/traffic-ml-engine:$VERSION|g" k8s/ml-api-deployment.yaml
    
    # Deploy ML API
    kubectl apply -f k8s/ml-api-deployment.yaml
    
    # Deploy Streaming API
    kubectl apply -f k8s/streaming-api-deployment.yaml
    
    # Wait for services to be ready
    log_info "Waiting for ML services to be ready..."
    kubectl wait --for=condition=ready pod -l app=ml-api -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=streaming-api -n $NAMESPACE --timeout=300s
    
    log_success "ML services deployed and ready"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    kubectl apply -f k8s/monitoring-deployment.yaml
    
    # Wait for monitoring to be ready
    log_info "Waiting for monitoring to be ready..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=grafana -n $NAMESPACE --timeout=300s
    
    log_success "Monitoring stack deployed and ready"
}

# Deploy ingress
deploy_ingress() {
    log_info "Deploying ingress..."
    
    kubectl apply -f k8s/ingress.yaml
    
    log_success "Ingress deployed"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Check ML API health
    ML_API_POD=$(kubectl get pods -l app=ml-api -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
    if kubectl exec $ML_API_POD -n $NAMESPACE -- curl -f http://localhost:8001/health &> /dev/null; then
        log_success "ML API health check passed"
    else
        log_error "ML API health check failed"
        exit 1
    fi
    
    # Check Streaming API health
    STREAMING_POD=$(kubectl get pods -l app=streaming-api -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
    if kubectl exec $STREAMING_POD -n $NAMESPACE -- curl -f http://localhost:8002/stream/stats &> /dev/null; then
        log_success "Streaming API health check passed"
    else
        log_error "Streaming API health check failed"
        exit 1
    fi
    
    # Check database connectivity
    POSTGRES_POD=$(kubectl get pods -l app=postgres -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
    if kubectl exec $POSTGRES_POD -n $NAMESPACE -- pg_isready -U traffic_user -d traffic_ml &> /dev/null; then
        log_success "PostgreSQL health check passed"
    else
        log_error "PostgreSQL health check failed"
        exit 1
    fi
    
    # Check Redis connectivity
    REDIS_POD=$(kubectl get pods -l app=redis -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
    if kubectl exec $REDIS_POD -n $NAMESPACE -- redis-cli ping &> /dev/null; then
        log_success "Redis health check passed"
    else
        log_error "Redis health check failed"
        exit 1
    fi
}

# Display deployment information
display_deployment_info() {
    log_info "Deployment completed successfully!"
    echo
    echo "=== Deployment Information ==="
    echo "Namespace: $NAMESPACE"
    echo "Version: $VERSION"
    echo "Environment: $ENVIRONMENT"
    echo
    echo "=== Services ==="
    kubectl get services -n $NAMESPACE
    echo
    echo "=== Pods ==="
    kubectl get pods -n $NAMESPACE
    echo
    echo "=== Ingress ==="
    kubectl get ingress -n $NAMESPACE
    echo
    echo "=== Access URLs ==="
    echo "ML API: https://api.traffic-ml.com"
    echo "Streaming API: https://stream.traffic-ml.com"
    echo "Monitoring: https://monitoring.traffic-ml.com"
    echo
    echo "=== Next Steps ==="
    echo "1. Configure DNS records for the domains"
    echo "2. Set up SSL certificates"
    echo "3. Configure monitoring alerts"
    echo "4. Run integration tests"
    echo "5. Set up backup procedures"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add cleanup commands here
}

# Main deployment function
main() {
    log_info "Starting deployment of Smart Traffic Management System"
    log_info "Version: $VERSION"
    log_info "Environment: $ENVIRONMENT"
    echo
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    create_namespace
    deploy_secrets
    deploy_persistent_volumes
    deploy_database
    deploy_redis
    
    # Only build and push images if not using existing ones
    if [ "$1" != "--skip-build" ]; then
        build_and_push_images
    fi
    
    deploy_ml_services
    deploy_monitoring
    deploy_ingress
    
    # Wait a bit for everything to stabilize
    log_info "Waiting for services to stabilize..."
    sleep 30
    
    run_health_checks
    display_deployment_info
    
    log_success "Deployment completed successfully!"
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--skip-build] [--help]"
        echo
        echo "Options:"
        echo "  --skip-build    Skip building and pushing Docker images"
        echo "  --help, -h      Show this help message"
        exit 0
        ;;
    --skip-build)
        main --skip-build
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
