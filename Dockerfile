# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies via uv
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

# Manually install OpenCV (recommended in slim images)
RUN pip install opencv-python

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=main.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Run the application
CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "main:app"]
