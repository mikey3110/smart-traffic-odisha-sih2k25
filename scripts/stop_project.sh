#!/bin/bash

# Smart Traffic Management System - Stop Script
# This script stops all components of the system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Logging
log() {
    echo -e "$1"
}

# Stop service by PID file
stop_service() {
    local service_name="$1"
    local pid_file="$2"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping $service_name (PID: $pid)..."
            kill "$pid"
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                log "Force stopping $service_name..."
                kill -9 "$pid"
            fi
            log "${GREEN}✓ $service_name stopped${NC}"
        else
            log "${YELLOW}⚠ $service_name was not running${NC}"
        fi
        rm -f "$pid_file"
    else
        log "${YELLOW}⚠ $service_name PID file not found${NC}"
    fi
}

# Stop all services
stop_all_services() {
    log "${BLUE}Stopping Smart Traffic Management System...${NC}"
    log ""
    
    # Stop Frontend
    stop_service "Frontend" "$PROJECT_ROOT/src/frontend/smart-traffic-ui/frontend.pid"
    
    # Stop CV Services
    stop_service "CV Demo Integration" "$PROJECT_ROOT/src/computer_vision/cv_demo.pid"
    stop_service "HLS Streaming Service" "$PROJECT_ROOT/src/computer_vision/hls_streaming.pid"
    
    # Stop ML API
    stop_service "ML API" "$PROJECT_ROOT/src/ml_engine/ml_api.pid"
    
    # Stop Backend API
    stop_service "Backend API" "$PROJECT_ROOT/src/backend/backend.pid"
    
    # Stop SUMO Simulation
    stop_service "SUMO Simulation" "$PROJECT_ROOT/sumo/sumo.pid"
    
    # Stop Docker services
    log "Stopping Docker services..."
    cd "$PROJECT_ROOT/src/backend"
    docker-compose down 2>/dev/null || true
    cd "$PROJECT_ROOT"
    
    # Kill any remaining processes
    log "Cleaning up remaining processes..."
    
    # Kill processes by port
    for port in 3000 5001 8000 8001 8813; do
        local pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ ! -z "$pid" ]; then
            log "Killing process on port $port (PID: $pid)"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    # Kill Python processes related to our project
    pkill -f "python.*main.py" 2>/dev/null || true
    pkill -f "python.*ml_api.py" 2>/dev/null || true
    pkill -f "python.*hls_streaming.py" 2>/dev/null || true
    pkill -f "python.*demo_integration.py" 2>/dev/null || true
    pkill -f "python.*launch_scenarios.py" 2>/dev/null || true
    
    # Kill Node.js processes related to our project
    pkill -f "npm.*dev" 2>/dev/null || true
    pkill -f "vite" 2>/dev/null || true
    
    log ""
    log "${GREEN}✓ All services stopped${NC}"
}

# Stop specific service
stop_service_by_name() {
    local service="$1"
    
    case "$service" in
        "frontend")
            stop_service "Frontend" "$PROJECT_ROOT/src/frontend/smart-traffic-ui/frontend.pid"
            ;;
        "backend")
            stop_service "Backend API" "$PROJECT_ROOT/src/backend/backend.pid"
            ;;
        "ml")
            stop_service "ML API" "$PROJECT_ROOT/src/ml_engine/ml_api.pid"
            ;;
        "cv")
            stop_service "CV Demo Integration" "$PROJECT_ROOT/src/computer_vision/cv_demo.pid"
            stop_service "HLS Streaming Service" "$PROJECT_ROOT/src/computer_vision/hls_streaming.pid"
            ;;
        "simulation")
            stop_service "SUMO Simulation" "$PROJECT_ROOT/sumo/sumo.pid"
            ;;
        "docker")
            log "Stopping Docker services..."
            cd "$PROJECT_ROOT/src/backend"
            docker-compose down
            cd "$PROJECT_ROOT"
            ;;
        *)
            log "${RED}Unknown service: $service${NC}"
            echo "Available services: frontend, backend, ml, cv, simulation, docker"
            exit 1
            ;;
    esac
}

# Check if services are running
check_services() {
    log "${BLUE}Checking running services...${NC}"
    log ""
    
    local services=(
        "Frontend:3000"
        "Backend API:8000"
        "ML API:8001"
        "CV Service:5001"
        "SUMO:8813"
    )
    
    for service_info in "${services[@]}"; do
        local service_name=$(echo "$service_info" | cut -d: -f1)
        local port=$(echo "$service_info" | cut -d: -f2)
        
        if lsof -ti:$port >/dev/null 2>&1; then
            local pid=$(lsof -ti:$port)
            log "${YELLOW}⚠ $service_name is still running on port $port (PID: $pid)${NC}"
        else
            log "${GREEN}✓ $service_name is stopped${NC}"
        fi
    done
}

# Force stop all
force_stop() {
    log "${RED}Force stopping all services...${NC}"
    
    # Kill all processes on our ports
    for port in 3000 5001 8000 8001 8813; do
        local pids=$(lsof -ti:$port 2>/dev/null || true)
        if [ ! -z "$pids" ]; then
            log "Force killing processes on port $port: $pids"
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
    done
    
    # Kill all related processes
    pkill -9 -f "python.*main.py" 2>/dev/null || true
    pkill -9 -f "python.*ml_api.py" 2>/dev/null || true
    pkill -9 -f "python.*hls_streaming.py" 2>/dev/null || true
    pkill -9 -f "python.*demo_integration.py" 2>/dev/null || true
    pkill -9 -f "python.*launch_scenarios.py" 2>/dev/null || true
    pkill -9 -f "npm.*dev" 2>/dev/null || true
    pkill -9 -f "vite" 2>/dev/null || true
    
    # Stop Docker services
    cd "$PROJECT_ROOT/src/backend"
    docker-compose down --remove-orphans 2>/dev/null || true
    cd "$PROJECT_ROOT"
    
    log "${GREEN}✓ Force stop completed${NC}"
}

# Clean up PID files
cleanup_pid_files() {
    log "Cleaning up PID files..."
    
    local pid_files=(
        "$PROJECT_ROOT/src/frontend/smart-traffic-ui/frontend.pid"
        "$PROJECT_ROOT/src/backend/backend.pid"
        "$PROJECT_ROOT/src/ml_engine/ml_api.pid"
        "$PROJECT_ROOT/src/computer_vision/hls_streaming.pid"
        "$PROJECT_ROOT/src/computer_vision/cv_demo.pid"
        "$PROJECT_ROOT/sumo/sumo.pid"
    )
    
    for pid_file in "${pid_files[@]}"; do
        if [ -f "$pid_file" ]; then
            rm -f "$pid_file"
            log "Removed $pid_file"
        fi
    done
    
    log "${GREEN}✓ PID files cleaned up${NC}"
}

# Main execution
main() {
    case "$1" in
        "frontend"|"backend"|"ml"|"cv"|"simulation"|"docker")
            stop_service_by_name "$1"
            ;;
        "check")
            check_services
            ;;
        "force")
            force_stop
            cleanup_pid_files
            ;;
        "cleanup")
            cleanup_pid_files
            ;;
        "all"|"")
            stop_all_services
            cleanup_pid_files
            ;;
        "--help"|"-h")
            echo "Usage: $0 [service|command]"
            echo ""
            echo "Services:"
            echo "  frontend    Stop frontend service"
            echo "  backend     Stop backend service"
            echo "  ml          Stop ML API service"
            echo "  cv          Stop computer vision services"
            echo "  simulation  Stop SUMO simulation"
            echo "  docker      Stop Docker services"
            echo ""
            echo "Commands:"
            echo "  check       Check which services are running"
            echo "  force       Force stop all services"
            echo "  cleanup     Clean up PID files"
            echo "  all         Stop all services (default)"
            echo ""
            echo "Options:"
            echo "  --help, -h  Show this help message"
            ;;
        *)
            echo "Unknown service or command: $1"
            echo "Use '$0 --help' for more information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
