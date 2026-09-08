FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        gcc-aarch64-linux-gnu \
        git \
        libgomp1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements-container.txt /tmp/requirements-container.txt
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir -r /tmp/requirements-container.txt \
    && rm /tmp/requirements-container.txt \
    && mkdir -p /home/vscode/.cache/cpuattn \
    && chown -R vscode:vscode "${VIRTUAL_ENV}" /home/vscode/.cache/cpuattn

USER vscode
WORKDIR /workspace
