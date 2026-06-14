# Stage 1: Build python dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compiler utilities for packages requiring C compiling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install packages to user local directory
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final lightweight runtime container
FROM python:3.11-slim AS runner

# Install system libraries required by PyMuPDF and Tesseract-OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Create non-root application user for execution security
RUN useradd -u 1000 -m appuser && \
    mkdir -p /app/data/uploads /app/data/faiss_index && \
    chown -R appuser:appuser /app

# Copy application codebase and configurations
COPY --chown=appuser:appuser requirements.txt .
COPY --chown=appuser:appuser app/ ./app/

# Switch to the non-root user
USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
