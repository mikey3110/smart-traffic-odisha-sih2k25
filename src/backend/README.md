# Smart Traffic Management System - Enhanced Backend

A comprehensive FastAPI-based backend system for AI-powered traffic signal optimization, featuring advanced logging, error handling, data validation, Redis caching, and comprehensive monitoring.

## 🚀 Features

### Core Functionality
- **Traffic Data Ingestion**: Real-time traffic data collection from computer vision systems
- **Signal Optimization**: AI-powered traffic signal timing optimization
- **Real-time Monitoring**: Live traffic status and signal performance tracking
- **Analytics & Reporting**: Comprehensive traffic analytics and performance metrics

### Enhanced Backend Features
- **Comprehensive Logging**: Structured logging with multiple handlers and log levels
- **Advanced Error Handling**: Custom exception classes with proper HTTP status codes
- **Data Validation**: Advanced Pydantic models with field validation and custom validators
- **Redis Integration**: Connection pooling, retry logic, and caching for reliability
- **Middleware Stack**: Request timing, rate limiting, and security logging
- **Database Models**: SQLAlchemy models with relationships and indexes
- **API Versioning**: v1 endpoints with backward compatibility
- **Health Monitoring**: Comprehensive health checks for all system components
- **Docker Support**: Multi-stage Docker builds with production configuration

## 📁 Project Structure

```
src/backend/
├── api/                          # API endpoints
│   ├── v1/                      # Version 1 API endpoints
│   │   ├── traffic.py           # Traffic data endpoints
│   │   ├── signals.py           # Signal optimization endpoints
│   │   └── health.py            # Health check endpoints
│   ├── dependencies.py          # API dependencies
│   └── signals.py               # Signal-related endpoints
├── config/                      # Configuration management
│   ├── settings.py              # Pydantic settings
│   └── logging_config.py        # Logging configuration
├── database/                    # Database layer
│   ├── connection.py            # Database connection management
│   ├── models.py                # SQLAlchemy models
│   └── enhanced_database.py     # Enhanced database utilities
├── exceptions/                  # Custom exception handling
│   ├── custom_exceptions.py     # Custom exception classes
│   └── handlers.py              # Exception handlers
├── middleware/                  # Custom middleware
│   ├── logging_middleware.py    # Request/response logging
│   ├── rate_limiting.py         # Rate limiting middleware
│   └── request_timing.py        # Request timing middleware
├── models/                      # Pydantic models
│   ├── schemas.py               # API schemas
│   ├── enhanced_models.py       # Enhanced validation models
│   └── enums.py                 # Enumerations and constants
├── services/                    # Business logic services
│   ├── redis_service.py         # Redis service with pooling
│   └── optimization_service.py  # Signal optimization service
├── tests/                       # Unit tests
│   ├── test_enhanced_features.py # Comprehensive feature tests
│   ├── test_redis_service.py    # Redis service tests
│   └── test_middleware.py       # Middleware tests
├── docker/                      # Docker configuration
│   ├── production/              # Production Docker configs
│   ├── nginx/                   # Nginx configuration
│   ├── postgres/                # PostgreSQL configuration
│   └── redis/                   # Redis configuration
├── main.py                      # FastAPI application entry point
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Multi-stage Docker build
└── docker-compose.yml           # Docker Compose configuration
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd smart-traffic-odisha-sih2k25/src/backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Set up database**
   ```bash
   # Create PostgreSQL database
   createdb traffic_management_db
   
   # Run migrations (if using Alembic)
   alembic upgrade head
   ```

6. **Start Redis**
   ```bash
   redis-server
   ```

7. **Run the application**
   ```bash
   python main.py
   ```

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Run in development mode**
   ```bash
   docker-compose --profile dev up --build
   ```

3. **Run tests**
   ```bash
   docker-compose exec backend pytest
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment name | `development` |
| `API_DEBUG` | Enable debug mode | `false` |
| `API_HOST` | API host | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `DB_URL` | Database URL | `sqlite:///./traffic_management.db` |
| `REDIS_HOST` | Redis host | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `REDIS_PASSWORD` | Redis password | `None` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FILE_PATH` | Log file path | `logs/traffic_api.log` |

### Database Configuration

The system supports multiple database backends:
- **PostgreSQL** (recommended for production)
- **SQLite** (development)
- **MySQL** (alternative)

### Redis Configuration

Redis is used for:
- Caching traffic data
- Rate limiting
- Session storage
- Real-time data storage

## 📚 API Documentation

### Base URL
- Development: `http://localhost:8000`
- Production: `https://your-domain.com`

### API Endpoints

#### Traffic Data
- `POST /api/v1/traffic/ingest` - Ingest traffic data
- `GET /api/v1/traffic/status/{intersection_id}` - Get traffic status
- `GET /api/v1/traffic/history/{intersection_id}` - Get traffic history
- `GET /api/v1/traffic/analytics/{intersection_id}` - Get traffic analytics

#### Signal Optimization
- `PUT /api/v1/signals/optimize/{intersection_id}` - Optimize signal timings
- `GET /api/v1/signals/status/{intersection_id}` - Get signal status
- `GET /api/v1/signals/history/{intersection_id}` - Get optimization history
- `GET /api/v1/signals/performance/{intersection_id}` - Get performance metrics

#### Health & Monitoring
- `GET /api/v1/health/` - Comprehensive health check
- `GET /api/v1/health/database` - Database health
- `GET /api/v1/health/redis` - Redis health
- `GET /api/v1/health/system` - System health
- `GET /api/v1/health/ready` - Readiness probe
- `GET /api/v1/health/live` - Liveness probe

### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_enhanced_features.py

# Run with verbose output
pytest -v
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Middleware Tests**: Middleware functionality testing
- **Service Tests**: Business logic testing

## 📊 Monitoring & Logging

### Logging
- **Structured Logging**: JSON-formatted logs with context
- **Multiple Handlers**: Console, file, and error-specific handlers
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Request Tracking**: Request ID correlation across services

### Health Monitoring
- **Database Health**: Connection status and performance metrics
- **Redis Health**: Cache status and performance metrics
- **System Health**: CPU, memory, and disk usage
- **API Health**: Response times and error rates

### Metrics
- **Prometheus Integration**: Metrics collection and export
- **Custom Metrics**: Traffic data, optimization performance
- **Grafana Dashboards**: Visualization and alerting

## 🚀 Deployment

### Production Deployment

1. **Using Docker Compose**
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

2. **Using Kubernetes**
   ```bash
   kubectl apply -f k8s/
   ```

3. **Manual Deployment**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run migrations
   alembic upgrade head
   
   # Start with Gunicorn
   gunicorn main:app -c gunicorn.conf.py
   ```

### Environment-Specific Configurations

- **Development**: SQLite, debug logging, hot reload
- **Staging**: PostgreSQL, structured logging, monitoring
- **Production**: PostgreSQL, optimized logging, full monitoring

## 🔒 Security Features

- **Rate Limiting**: Per-endpoint rate limiting with Redis
- **Input Validation**: Comprehensive data validation with Pydantic
- **Security Headers**: CORS, XSS protection, content type validation
- **Error Handling**: Secure error responses without information leakage
- **Request Logging**: Security-focused request logging and monitoring

## 📈 Performance Features

- **Connection Pooling**: Database and Redis connection pooling
- **Caching**: Multi-level caching with Redis
- **Async Operations**: Asynchronous request handling
- **Request Timing**: Performance monitoring and optimization
- **Compression**: Gzip compression for API responses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Code Quality
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the API documentation at `/docs`

## 🔄 Changelog

### v1.0.0
- Initial release with comprehensive backend features
- Advanced logging and error handling
- Redis integration with connection pooling
- Comprehensive API with versioning
- Docker support with multi-stage builds
- Complete test suite
- Production-ready configuration

---

**Smart Traffic Management System** - Built for Smart India Hackathon 2025