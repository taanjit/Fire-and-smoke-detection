# Dockerfile for Fire and Smoke Detection API

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application requirements
COPY requirements_deploy.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy application files
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY config/ ./config/
COPY artifacts/ ./artifacts/
COPY params.yaml .
COPY schema.yaml .
COPY setup.py .
COPY README.md .
COPY app.py .

# Install the package properly
RUN pip install .

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
