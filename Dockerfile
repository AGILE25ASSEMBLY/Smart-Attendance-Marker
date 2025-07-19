# Use a minimal Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OpenCV dependencies (needed for face recognition)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast dependency installer)
RUN pip install uv

# Copy dependency files and install
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Install OpenCV manually via pip (instead of `python3-opencv`)
RUN pip install opencv-python

# Copy app source code
COPY . .

# Create necessary directories
RUN mkdir -p uploads

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=main.py
ENV FLASK_ENV=production

# Expose port (used by gunicorn)
EXPOSE 5000

# Run using uv + gunicorn
CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:5000", "--workers=2", "--timeout=120", "main:app"]
