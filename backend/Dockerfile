FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3 and system tools
RUN apt-get update && \
    apt-get install -y \
    python3-full \
    python3-pip \
    ffmpeg \
    curl \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps globally
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy app code
COPY . .

# Default command
CMD ["python3"]
