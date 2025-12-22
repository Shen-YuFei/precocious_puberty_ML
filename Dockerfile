# PPML - Precocious Puberty Machine Learning
# Base image with PyTorch + CUDA support (using mirror proxy)
FROM docker.1ms.run/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Install code-server (VS Code Server)
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Configure SSH
RUN mkdir -p /var/run/sshd && \
    echo 'root:password' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Copy requirements first for better caching
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r requirements-docker.txt

# Copy and install local vendor packages (tabm, TabPFN, tabpfn-extensions)
COPY vendor/ /tmp/vendor/
RUN pip install --no-cache-dir /tmp/vendor/tabm /tmp/vendor/TabPFN /tmp/vendor/tabpfn-extensions \
    && rm -rf /tmp/vendor

# Copy project files
COPY . .

# Expose ports: SSH(22), Jupyter(8888), VS Code(8080)
EXPOSE 22 8888 8080

# Create startup script
RUN echo '#!/bin/bash\n\
    /usr/sbin/sshd\n\
    code-server --bind-addr 0.0.0.0:8080 --auth none &\n\
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=""\n' > /start.sh && chmod +x /start.sh

# Default command: Start all services
CMD ["/start.sh"]
