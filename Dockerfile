# PPML - Precocious Puberty Machine Learning
# Base image with PyTorch + CUDA support (using mirror proxy)
FROM docker.1ms.run/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Install editable GitHub packages
RUN pip install --no-cache-dir \
    "git+https://github.com/Shen-YuFei/tabm.git@28e47ae301c92ec37787dde1ce923a0793f405b4" \
    "git+https://github.com/Shen-YuFei/TabPFN.git@89d80734cb673212a055b2451210b0a1398a576c" \
    "git+https://github.com/Shen-YuFei/tabpfn-extensions.git@7a900321c6aaea8e58ae23117628eaf18cc5a4ab"

# Copy project files
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Default command: Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
