FROM mcr.microsoft.com/devcontainers/base:noble

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    pkg-config \
    git \
    curl \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    tk-dev \
    uuid-dev && \
    rm -rf /var/lib/apt/lists/*
