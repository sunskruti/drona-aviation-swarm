# ==============================================================================
# Drona Aviation Multi-Drone Swarm Execution Environment (Inter IIT 15.0)
# Containerized for Airlink Remote Drone Arena & Evaluation
# ==============================================================================
FROM ubuntu:22.04

LABEL maintainer="Inter IIT Tech Meet 15.0 Swarm Team"
LABEL description="Synchronized Multi-Drone Swarm Execution on Pluto Drones"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DISTRO=humble

# Install essential system utilities and Python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up project workspace
WORKDIR /drona_swarm

# Install Python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy complete swarm codebase
COPY config/ ./config/
COPY src/ ./src/
COPY tests/ ./tests/

# Set Python path to include src
ENV PYTHONPATH="/drona_swarm/src:${PYTHONPATH}"

# Default entrypoint runs the full verification test suite and mission manager
CMD ["python3", "-m", "pytest", "-v", "tests/"]

