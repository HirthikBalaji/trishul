FROM python:3.12-slim

LABEL maintainer="TRISHUL AI Team"
LABEL description="Physics-Gated Multimodal Flood Damage Mapping & Isolation Analysis for Himalayan Valleys"

WORKDIR /app

# Install system dependencies for geospatial packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reproducible dependency resolution
RUN pip install --no-cache-dir uv

# Copy configuration and project definitions
COPY pyproject.toml README.md Makefile LICENSE LICENSE-DATA.md ./

# Install project dependencies
RUN uv pip install --system -e ".[report,ai]"

# Copy application code and assets
COPY trishul/ ./trishul/
COPY eval/ ./eval/
COPY models/ ./models/
COPY web/ ./web/

# Expose FastAPI backend port
EXPOSE 8000

# Default command: launch backend API server
CMD ["uvicorn", "trishul.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
