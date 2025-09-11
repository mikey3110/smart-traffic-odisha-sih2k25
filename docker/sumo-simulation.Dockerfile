FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including SUMO
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gcc \
    g++ \
    make \
    cmake \
    libxerces-c-dev \
    libproj-dev \
    libgdal-dev \
    libfox-1.6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglib2.0-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxext-dev \
    libxfixes-dev \
    libxrender-dev \
    libx11-dev \
    libxft-dev \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    libtiff-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install SUMO
RUN wget https://sumo.dlr.de/releases/1.15.0/sumo-src-1.15.0.tar.gz && \
    tar -xzf sumo-src-1.15.0.tar.gz && \
    cd sumo-1.15.0 && \
    make -f Makefile.cvs && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf sumo-1.15.0 sumo-src-1.15.0.tar.gz

# Copy requirements first for better caching
COPY requirements.txt .
COPY src/simulation/sumo_integration/requirements.txt ./sumo-requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r sumo-requirements.txt

# Copy application code
COPY src/simulation/sumo_integration/ ./sumo_integration/
COPY src/shared/ ./shared/

# Create necessary directories
RUN mkdir -p logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SUMO_HOME=/usr/local
ENV PATH=$PATH:/usr/local/bin

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["python", "sumo_integration/run_sumo_integration.py"]
