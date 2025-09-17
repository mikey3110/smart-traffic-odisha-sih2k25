#!/bin/bash

# Smart Traffic Management System - Complete Project Startup Script
# This script starts all components of the system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="project_startup_$(date +%Y%m%d_%H%M%S).log"

# Logging
log() {
    echo "$1" | tee -a "$LOG_FILE"
    echo -e "$1"
}

# Check prerequisites
check_prerequisites() {
    log "${BLUE}Checking prerequisites...${NC}"
    
    local missing_deps=()
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        missing_deps+=("Node.js")
    else
        NODE_VERSION=$(node --version)
        log "${GREEN}✓ Node.js: $NODE_VERSION${NC}"
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("Python 3")
    else
        PYTHON_VERSION=$(python3 --version)
        log "${GREEN}✓ Python: $PYTHON_VERSION${NC}"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("Docker")
    else
        DOCKER_VERSION=$(docker --version)
        log "${GREEN}✓ Docker: $DOCKER_VERSION${NC}"
    fi
    
    # Check kubectl (optional)
    if command -v kubectl &> /dev/null; then
        KUBECTL_VERSION=$(kubectl version --client --short 2>/dev/null || echo "kubectl available")
        log "${GREEN}✓ kubectl: $KUBECTL_VERSION${NC}"
    else
        log "${YELLOW}⚠ kubectl not found (optional for Kubernetes deployment)${NC}"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log "${RED}Missing dependencies: ${missing_deps[*]}${NC}"
        log "Please install the missing dependencies and try again."
        exit 1
    fi
    
    log "${GREEN}✓ All prerequisites satisfied${NC}"
}

# Start Backend Services
start_backend() {
    log "${BLUE}Starting Backend Services...${NC}"
    
    cd "$PROJECT_ROOT/src/backend"
    
    # Install Python dependencies
    if [ ! -d "venv" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Start PostgreSQL and Redis with Docker Compose
    log "Starting PostgreSQL and Redis..."
    docker-compose up -d postgres redis
    
    # Wait for services to be ready
    log "Waiting for database services..."
    sleep 10
    
    # Start Backend API
    log "Starting Backend API..."
    python main.py &
    BACKEND_PID=$!
    echo $BACKEND_PID > backend.pid
    
    # Start ML API
    log "Starting ML API..."
    cd ../ml_engine
    python api/ml_api.py &
    ML_API_PID=$!
    echo $ML_API_PID > ml_api.pid
    
    cd "$PROJECT_ROOT"
    log "${GREEN}✓ Backend services started${NC}"
}

# Start Computer Vision Service
start_cv_service() {
    log "${BLUE}Starting Computer Vision Service...${NC}"
    
    cd "$PROJECT_ROOT/src/computer_vision"
    
    # Install CV dependencies
    pip install -r requirements.txt
    
    # Start HLS Streaming Service
    log "Starting HLS Streaming Service..."
    python hls_streaming.py &
    HLS_PID=$!
    echo $HLS_PID > hls_streaming.pid
    
    # Start CV Demo Integration
    log "Starting CV Demo Integration..."
    python demo_integration.py &
    CV_DEMO_PID=$!
    echo $CV_DEMO_PID > cv_demo.pid
    
    cd "$PROJECT_ROOT"
    log "${GREEN}✓ Computer Vision services started${NC}"
}

# Start Frontend
start_frontend() {
    log "${BLUE}Starting Frontend...${NC}"
    
    cd "$PROJECT_ROOT/src/frontend/smart-traffic-ui"
    
    # Install Node.js dependencies
    if [ ! -d "node_modules" ]; then
        log "Installing Node.js dependencies..."
        npm install
    fi
    
    # Start development server
    log "Starting React development server..."
    npm run dev &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > frontend.pid
    
    cd "$PROJECT_ROOT"
    log "${GREEN}✓ Frontend started${NC}"
}

# Start SUMO Simulation
start_simulation() {
    log "${BLUE}Starting SUMO Simulation...${NC}"
    
    cd "$PROJECT_ROOT/sumo"
    
    # Check if SUMO is installed
    if ! command -v sumo &> /dev/null; then
        log "${YELLOW}⚠ SUMO not found, skipping simulation${NC}"
        log "To install SUMO: https://sumo.dlr.de/docs/Downloads.php"
        return 0
    fi
    
    # Start SUMO simulation
    log "Starting SUMO simulation..."
    python launch_scenarios.py &
    SUMO_PID=$!
    echo $SUMO_PID > sumo.pid
    
    cd "$PROJECT_ROOT"
    log "${GREEN}✓ SUMO simulation started${NC}"
}

# Run health checks
run_health_checks() {
    log "${BLUE}Running health checks...${NC}"
    
    # Wait for services to start
    sleep 15
    
    # Check Backend API
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        log "${GREEN}✓ Backend API is healthy${NC}"
    else
        log "${RED}✗ Backend API is not responding${NC}"
    fi
    
    # Check ML API
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        log "${GREEN}✓ ML API is healthy${NC}"
    else
        log "${RED}✗ ML API is not responding${NC}"
    fi
    
    # Check CV Service
    if curl -s http://localhost:5001/cv/streams > /dev/null 2>&1; then
        log "${GREEN}✓ CV Service is healthy${NC}"
    else
        log "${RED}✗ CV Service is not responding${NC}"
    fi
    
    # Check Frontend
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        log "${GREEN}✓ Frontend is healthy${NC}"
    else
        log "${RED}✗ Frontend is not responding${NC}"
    fi
}

# Display access information
show_access_info() {
    log "${BLUE}================================================${NC}"
    log "${GREEN}Smart Traffic Management System is running!${NC}"
    log "${BLUE}================================================${NC}"
    log ""
    log "${YELLOW}Access URLs:${NC}"
    log "  Frontend Dashboard: http://localhost:3000"
    log "  Backend API: http://localhost:8000"
    log "  ML API: http://localhost:8001"
    log "  CV Service: http://localhost:5001"
    log "  API Documentation: http://localhost:8000/docs"
    log ""
    log "${YELLOW}Services Status:${NC}"
    log "  Backend PID: $(cat src/backend/backend.pid 2>/dev/null || echo 'Not running')"
    log "  ML API PID: $(cat src/ml_engine/ml_api.pid 2>/dev/null || echo 'Not running')"
    log "  CV Service PID: $(cat src/computer_vision/hls_streaming.pid 2>/dev/null || echo 'Not running')"
    log "  Frontend PID: $(cat src/frontend/smart-traffic-ui/frontend.pid 2>/dev/null || echo 'Not running')"
    log ""
    log "${YELLOW}Log Files:${NC}"
    log "  Project Log: $LOG_FILE"
    log "  Backend Log: src/backend/logs/"
    log "  ML Log: src/ml_engine/logs/"
    log "  CV Log: src/computer_vision/logs/"
    log ""
    log "${YELLOW}To stop the system:${NC}"
    log "  ./scripts/stop_project.sh"
    log ""
    log "${GREEN}System is ready for use!${NC}"
}

# Main execution
main() {
    log "${BLUE}Starting Smart Traffic Management System...${NC}"
    log "Project Root: $PROJECT_ROOT"
    log "Log File: $LOG_FILE"
    log ""
    
    # Check prerequisites
    check_prerequisites
    log ""
    
    # Start services
    start_backend
    log ""
    
    start_cv_service
    log ""
    
    start_frontend
    log ""
    
    start_simulation
    log ""
    
    # Run health checks
    run_health_checks
    log ""
    
    # Show access information
    show_access_info
}

# Handle script arguments
case "$1" in
    "backend")
        check_prerequisites
        start_backend
        ;;
    "frontend")
        check_prerequisites
        start_frontend
        ;;
    "cv")
        check_prerequisites
        start_cv_service
        ;;
    "simulation")
        check_prerequisites
        start_simulation
        ;;
    "health")
        run_health_checks
        ;;
    "full"|"")
        main
        ;;
    "--help"|"-h")
        echo "Usage: $0 [service]"
        echo ""
        echo "Services:"
        echo "  backend     Start only backend services"
        echo "  frontend    Start only frontend"
        echo "  cv          Start only computer vision services"
        echo "  simulation  Start only SUMO simulation"
        echo "  health      Run health checks"
        echo "  full        Start all services (default)"
        echo ""
        echo "Options:"
        echo "  --help, -h  Show this help message"
        ;;
    *)
        echo "Unknown service: $1"
        echo "Use '$0 --help' for more information"
        exit 1
        ;;
esac
