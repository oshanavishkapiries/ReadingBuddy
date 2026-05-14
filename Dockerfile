FROM python:3.12-slim

# System dependencies:
#   tesseract-ocr + eng pack  → OCR pipeline
#   libgl1, libglib2.0-0      → OpenCV (cv2)
#   libsm6, libxext6          → OpenCV display libs (headless still needs them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying source — maximises layer cache reuse
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Chromium + all its OS-level dependencies (used by Playwright for PDF rendering)
RUN playwright install --with-deps chromium

# Copy application source
COPY . .

# Persistent-data directories (overridden by named volumes at runtime)
RUN mkdir -p /app/data /app/workspace

ENV PYTHONUNBUFFERED=1
# Point the DB at the mounted data volume; can be overridden via env_file
ENV DATABASE_URL=sqlite:////app/data/readingbuddy.db

EXPOSE 8000

CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
