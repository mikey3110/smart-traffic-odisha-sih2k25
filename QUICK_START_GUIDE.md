# 🚀 Quick Start Guide - Smart Traffic Management System

## 🎯 **Get the Project Running in 5 Minutes!**

This guide will help you start the complete Smart Traffic Management System quickly and easily.

---

## 📋 **Prerequisites**

### Required Software
- **Node.js** 18+ - [Download](https://nodejs.org/)
- **Python** 3.8+ - [Download](https://python.org/)
- **Docker** - [Download](https://docker.com/)
- **Git** - [Download](https://git-scm.com/)

### Optional (for full functionality)
- **SUMO** - [Download](https://sumo.dlr.de/docs/Downloads.php)
- **kubectl** - [Download](https://kubernetes.io/docs/tasks/tools/)

---

## 🚀 **Quick Start (Choose Your Platform)**

### **Option 1: Windows (Recommended for Windows Users)**

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/smart-traffic-odisha-sih2k25.git
cd smart-traffic-odisha-sih2k25

# 2. Start the entire system
scripts\start_project.bat

# 3. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### **Option 2: Linux/macOS**

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/smart-traffic-odisha-sih2k25.git
cd smart-traffic-odisha-sih2k25

# 2. Make scripts executable
chmod +x scripts/*.sh

# 3. Start the entire system
./scripts/start_project.sh

# 4. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### **Option 3: Docker Compose (All Platforms)**

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/smart-traffic-odisha-sih2k25.git
cd smart-traffic-odisha-sih2k25

# 2. Start with Docker Compose
docker-compose up -d

# 3. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## 🎮 **What You'll See**

### **1. Frontend Dashboard** (http://localhost:3000)
- **Real-time Traffic Map** with Leaflet integration
- **Live Camera Feeds** from traffic intersections
- **Vehicle Counts** updated every 30 seconds
- **Signal Control** interface
- **Performance Metrics** dashboard

### **2. Backend API** (http://localhost:8000)
- **RESTful APIs** for all traffic management functions
- **Interactive API Documentation** at `/docs`
- **Health Check** endpoint at `/health`
- **Real-time Data** ingestion and processing

### **3. ML API** (http://localhost:8001)
- **Machine Learning** optimization endpoints
- **Q-Learning** traffic signal optimization
- **Performance Analytics** and metrics
- **Model Management** and versioning

### **4. Computer Vision Service** (http://localhost:5001)
- **HLS Video Streaming** for camera feeds
- **Vehicle Detection** using YOLO v8
- **Real-time Counts** API endpoints
- **Multi-camera** processing

---

## 🔧 **Individual Service Control**

### **Start Individual Services**

```bash
# Start only backend
./scripts/start_project.sh backend

# Start only frontend
./scripts/start_project.sh frontend

# Start only computer vision
./scripts/start_project.sh cv

# Start only simulation
./scripts/start_project.sh simulation
```

### **Stop Services**

```bash
# Stop all services
./scripts/stop_project.sh

# Stop specific service
./scripts/stop_project.sh frontend
./scripts/stop_project.sh backend
./scripts/stop_project.sh ml
./scripts/stop_project.sh cv
./scripts/stop_project.sh simulation
```

---

## 🧪 **Testing the System**

### **1. Health Checks**
```bash
# Run comprehensive health checks
./scripts/smoke_test.sh

# Quick health check
./scripts/smoke_test.sh --quick
```

### **2. API Testing**
```bash
# Test backend API
curl http://localhost:8000/health

# Test ML API
curl http://localhost:8001/health

# Test CV service
curl http://localhost:5001/cv/streams
```

### **3. Frontend Testing**
- Open http://localhost:3000 in your browser
- Check if the dashboard loads
- Verify real-time data updates
- Test camera feed integration

---

## 🐛 **Troubleshooting**

### **Common Issues**

#### **1. Port Already in Use**
```bash
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Linux/macOS
lsof -ti:3000 | xargs kill -9
```

#### **2. Services Not Starting**
```bash
# Check logs
tail -f logs/backend.log
tail -f logs/frontend.log

# Check Docker
docker ps
docker logs <container_name>
```

#### **3. Database Connection Issues**
```bash
# Check PostgreSQL
docker ps | grep postgres
docker logs <postgres_container>

# Check Redis
docker ps | grep redis
docker logs <redis_container>
```

#### **4. Frontend Build Issues**
```bash
# Clear cache and reinstall
cd src/frontend/smart-traffic-ui
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### **Reset Everything**
```bash
# Stop all services
./scripts/stop_project.sh

# Clean Docker
docker-compose down -v
docker system prune -f

# Restart
./scripts/start_project.sh
```

---

## 📊 **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   ML API        │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (Python)      │
│   Port: 3000    │    │   Port: 8000    │    │   Port: 8001    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CV Service    │    │   PostgreSQL    │    │   Redis         │
│   (HLS + YOLO)  │    │   (Database)    │    │   (Cache)       │
│   Port: 5001    │    │   Port: 5432    │    │   Port: 6379    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   SUMO          │
│   (Simulation)  │
│   Port: 8813    │
└─────────────────┘
```

---

## 🎯 **Demo Scenarios**

### **1. Traffic Optimization Demo**
1. Open http://localhost:3000
2. Navigate to the traffic map
3. Watch real-time vehicle detection
4. Observe ML-optimized signal timing
5. View performance metrics

### **2. Camera Feed Demo**
1. Go to the camera section
2. View live HLS video streams
3. See vehicle counting in action
4. Monitor detection accuracy

### **3. API Testing Demo**
1. Open http://localhost:8000/docs
2. Test the interactive API
3. Submit traffic data
4. View optimization results

---

## 📈 **Performance Monitoring**

### **Key Metrics to Watch**
- **API Response Time**: <300ms
- **Frontend Load Time**: <2s
- **Vehicle Detection**: 30-second updates
- **Stream Latency**: <2s
- **Memory Usage**: <2GB total

### **Monitoring URLs**
- **Grafana**: http://localhost:3000 (if monitoring enabled)
- **Prometheus**: http://localhost:9090 (if monitoring enabled)
- **Health Checks**: http://localhost:8000/health

---

## 🚀 **Production Deployment**

### **Kubernetes Deployment**
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/demo.yaml

# Check deployment
kubectl get pods -n smart-traffic-demo

# Access services
kubectl get services -n smart-traffic-demo
```

### **Docker Production**
```bash
# Build production images
docker-compose -f docker-compose.production.yml up -d

# Scale services
docker-compose up -d --scale backend=3
```

---

## 📚 **Additional Resources**

### **Documentation**
- [API Documentation](docs/api.yaml)
- [Deployment Guide](docs/devops_guide.md)
- [CV Integration Guide](docs/cv_integration.md)
- [Testing Guide](src/frontend/smart-traffic-ui/TESTING_GUIDE.md)

### **Development**
- [Frontend README](src/frontend/smart-traffic-ui/README.md)
- [Backend README](src/backend/README.md)
- [ML Engine README](src/ml_engine/README.md)
- [Simulation README](sumo/README.md)

### **Support**
- Check the troubleshooting section above
- Review the logs in the `logs/` directory
- Check the GitHub Issues page
- Contact the development team

---

## 🎉 **You're Ready!**

The Smart Traffic Management System is now running and ready for:

- ✅ **Demo Presentations**
- ✅ **Development Work**
- ✅ **Testing & Validation**
- ✅ **Production Deployment**

**Happy coding! 🚀**

---

**Quick Start Guide v1.0 - Smart India Hackathon 2025**

*For additional help, check the troubleshooting section or contact the development team.*
