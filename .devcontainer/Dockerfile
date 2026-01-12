FROM --platform=linux/x86_64 ghcr.io/prefix-dev/pixi:latest

# Install git (required by setuptools_scm)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
