# -------------------------------------------
# 1. Base image
# -------------------------------------------
FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# -------------------------------------------
# 2. Install system dependencies
# -------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    ffmpeg \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -------------------------------------------
# 3. Set working directory
# -------------------------------------------
WORKDIR /usr/src/app/backend

# -------------------------------------------
# 4. Copy ONLY requirements first (for docker layer caching)
# -------------------------------------------
COPY backend/requirements.txt requirements.txt

# -------------------------------------------
# 5. Upgrade pip & install Python deps
# -------------------------------------------
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# -------------------------------------------
# 6. Copy entire backend source
# -------------------------------------------
COPY backend/ .

# -------------------------------------------
# 7. Create runtime folders
# -------------------------------------------
RUN mkdir -p /usr/src/app/data/uploads/enrollment_images \
    && mkdir -p /usr/src/app/logs

# -------------------------------------------
# 8. Env vars
# -------------------------------------------
ENV PORT=8000

# -------------------------------------------
# 9. Start FastAPI app
# -------------------------------------------
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]