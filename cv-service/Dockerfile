FROM python:3.11-slim

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY cv-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy CV extraction code from parent (needed for /extract-geometry endpoint)
COPY extract_board_cells_gridlines.py .

# Copy service code
COPY cv-service/main.py .
COPY cv-service/hybrid_extraction.py .

# Run service
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
