# 
# Dockerfile  (CPU-only, Python 3.11)
# 
FROM python:3.11-slim

# System libraries required by Pillow / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies 
# Install CPU-only PyTorch first (explicit wheel index keeps the image small)
RUN pip install --no-cache-dir \
        torch==2.3.0+cpu \
        torchvision==0.18.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies
RUN pip install --no-cache-dir \
        scikit-learn>=1.3.0 \
        matplotlib>=3.7.0 \
        Pillow>=10.0.0 \
        numpy>=1.24.0 \
        tqdm>=4.65.0

# Source code 
# Dataset and run artefacts are volume-mounted in docker-compose;
# they are excluded via .dockerignore to keep the build context small.
COPY ml/ ./ml/

# Default command 
CMD ["python", "ml/train.py"]
