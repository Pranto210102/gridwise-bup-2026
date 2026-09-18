# Use official lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential needed for some optimization tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/
COPY sample_cases.json .

# Expose standard service port
EXPOSE 8000

# Environment variables (Never bake secrets into images!)
ENV PORT=8000 \
    PYTHONUNBUFFERED=1

# Healthcheck for container orchestrators
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI service bound to 0.0.0.0 and dynamic PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
