FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/devuser/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m devuser
WORKDIR /usr/src
USER devuser

RUN pip install --no-cache-dir --upgrade pip

RUN git config --global init.defaultBranch main
