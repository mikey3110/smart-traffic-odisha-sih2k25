# 🚦 Smart Traffic Management System - SIH 2025

**Government of Odisha Problem Statement**  
AI-based traffic management system to optimize signal timings and reduce urban congestion by 10%

## 🎯 Project Overview
Smart Traffic Orchestration Platform that combines:
- **Real-time AI traffic signal control** using computer vision
- **Multi-intersection coordination** for smooth traffic flow  
- **Predictive analytics** to prevent congestion
- **Professional dashboard** for traffic authorities

## 👥 Team Members
| Name | Role | GitHub | Responsibility |
|------|------|--------|----------------|
| vijay | Team Leader | [@mikey3110] | Integration & Coordination |
| siva ganesh | Computer Vision Engineer | [@GANESH2006-web] | Vehicle Detection |
| vijay | AI/ML Engineer | [@mikey3110] | Signal Optimization |
| vijayalakshmi | Backend Developer | [@Vijaya72252] | APIs & Data Pipeline |
| zebnoor | Frontend Developer | [@Zebnoor620] | Dashboard & UI |
| vamsi | SUMO Specialist | [@Nukanaboyinavamsi] | Traffic Simulation |
| voshan | DevOps Engineer | [@scanvoshan] | Deployment & Testing |

## 🛠️ Technology Stack
- **Backend:** Python, FastAPI, Redis
- **Computer Vision:** OpenCV, YOLO
- **AI/ML:** Reinforcement Learning (Stable-Baselines3)
- **Frontend:** React, Chart.js
- **Simulation:** SUMO Traffic Simulator

## 🚀 Quick Start

### Option 1: Automated Installation (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd smart-traffic-odisha-sih2025

# Run automated installation
python install.py

# Start the complete system
python run_system.py
```

### Option 2: Manual Installation
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Node.js dependencies
cd src/frontend/smart-traffic-ui
npm install
cd ../../..

# 3. Start the system
python run_system.py
```

## 🎮 System Components

### 1. Backend API Server
- **Location:** `src/backend/`
- **Port:** 8000
- **Features:** REST API, Redis integration, real-time data processing
- **Start:** `python src/backend/main.py`

### 2. Frontend Dashboard
- **Location:** `src/frontend/smart-traffic-ui/`
- **Port:** 5173
- **Features:** Real-time charts, intersection monitoring, AI status
- **Start:** `cd src/frontend/smart-traffic-ui && npm run dev`

### 3. Computer Vision Module
- **Location:** `src/computer_vision/`
- **Features:** YOLO vehicle detection, lane counting, API integration
- **Start:** `python src/computer_vision/vehicle_count.py`

### 4. ML Optimization Engine
- **Location:** `src/ml_engine/`
- **Features:** Signal timing optimization, continuous learning
- **Start:** `python src/ml_engine/continuous_optimizer.py`

### 5. Traffic Simulation
- **Location:** `src/simulation/`
- **Features:** SUMO integration, performance validation
- **Start:** `python src/simulation/traffic_simulator_mock.py`

## 🌐 System URLs

When running, access these URLs:
- **🎨 Dashboard:** http://localhost:5173
- **📡 Backend API:** http://localhost:8000
- **📚 API Documentation:** http://localhost:8000/docs
- **📖 Alternative Docs:** http://localhost:8000/redoc

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | System status and available endpoints |
| POST | `/traffic/ingest` | Send vehicle count data |
| GET | `/traffic/status/{id}` | Get traffic status for intersection |
| PUT | `/signal/optimize/{id}` | Update signal timing |
| GET | `/signal/status/{id}` | Get current signal status |
| GET | `/intersections` | List all intersections |
| GET | `/health` | Health check |

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:
```env
# Backend Configuration
API_HOST=localhost
API_PORT=8000
REDIS_URL=redis://localhost:6379

# Computer Vision
VIDEO_PATH=data/videos/
YOLO_MODEL=yolov8n.pt

# ML Engine
OPTIMIZATION_INTERVAL=10
CONFIDENCE_THRESHOLD=0.8
```

### Video Files
Place traffic video files in `data/videos/` directory:
- Supported formats: `.mp4`, `.avi`, `.mov`
- The system will automatically detect and use available videos

## 🧪 Testing

### Test Individual Components
```bash
# Test backend API
python src/backend/test_api.py

# Test computer vision
python src/computer_vision/vehicle_count.py

# Test ML engine
python src/ml_engine/signal_optimizer.py

# Test simulation
python src/simulation/test_sumo_mock.py
```

### Test Complete System
```bash
# Start all components
python run_system.py

# Choose option 1 to start all components
# Monitor the dashboard at http://localhost:5173
```

## 📈 Expected Results

- **Traffic Improvement:** 15-25% reduction in wait times
- **Throughput Increase:** 20-30% more vehicles processed
- **Real-time Processing:** 5-second data refresh cycles
- **AI Accuracy:** 85%+ confidence in optimizations

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill processes on ports 8000 and 5173
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall all dependencies
   pip install -r requirements.txt
   cd src/frontend/smart-traffic-ui && npm install
   ```

3. **Video Files Not Found**
   - Add video files to `data/videos/` directory
   - Supported formats: MP4, AVI, MOV

4. **Redis Connection Issues**
   - System works without Redis (uses in-memory storage)
   - Install Redis for better performance

### Logs and Debugging
- Backend logs: Check terminal output
- Frontend logs: Browser console (F12)
- System logs: `logs/` directory

## 🚀 Deployment

### Production Setup
1. Install production dependencies
2. Configure environment variables
3. Set up Redis server
4. Use production WSGI server (Gunicorn)
5. Configure reverse proxy (Nginx)

### Docker Deployment
```bash
# Build and run with Docker
docker-compose up -d
```

## 📚 Documentation

- **API Documentation:** http://localhost:8000/docs
- **Component READMEs:** Check individual component folders
- **Architecture:** See `docs/` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is developed for SIH 2025 - Government of Odisha

## 🆘 Support

For issues and questions:
- Check the troubleshooting section
- Review component READMEs
- Create an issue in the repository

---

**🎯 Ready to optimize traffic in Odisha! 🚦**
