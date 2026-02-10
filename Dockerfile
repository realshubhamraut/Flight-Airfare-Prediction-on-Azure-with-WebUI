# =============================================================================
# Flight Airfare Prediction - Production Docker Image
# =============================================================================
# Multi-stage build for optimized Azure deployment
# Supports both FastAPI backend and Streamlit frontend

FROM python:3.9-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# =============================================================================
# API Stage - FastAPI/Flask Backend
# =============================================================================
FROM base AS api

# Copy and install API-specific requirements
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY fastapi_app.py flask_app.py inference.py ./
COPY models/ ./models/

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Streamlit Stage - Frontend
# =============================================================================
FROM base AS streamlit

# Copy and install Streamlit-specific requirements
COPY requirements-streamlit.txt ./
RUN pip install --no-cache-dir -r requirements-streamlit.txt

# Copy application code
COPY app.py ./
COPY models/ ./models/
COPY .streamlit/ ./.streamlit/

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# =============================================================================
# Default: Full Application (API mode)
# =============================================================================
FROM base AS full

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]