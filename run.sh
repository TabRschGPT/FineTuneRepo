#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
: "${NVIDIA_CONTAINER_TOOLKIT_VERSION:=1.17.8-1}"
: "${JUPYTER_PASSWORD:=mypassword}" 
: "${DOCKER_IMAGE:=unsloth/unsloth}"
: "${WORKDIR:=$PWD/work}"
CONTAINER_NAME=${CONTAINER_NAME:-unsloth}

# ========= Privileged runner =========
sudo_if_needed() { if [ "${EUID:-$UID}" -ne 0 ]; then sudo "$@"; else "$@"; fi; }

# ========= Package manager detection =========
PKG=yum
if command -v dnf >/dev/null 2>&1; then PKG=dnf; fi

# ========= Basic deps =========
sudo_if_needed $PKG -y update
sudo_if_needed $PKG -y install curl ca-certificates python3 python3-pip

if ! command -v docker >/dev/null 2>&1; then
  sudo_if_needed $PKG -y install docker
  sudo_if_needed systemctl enable --now docker
fi

# ========= Install cloudflared =========
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "Installing cloudflared..."
  sudo_if_needed mkdir -p /usr/local/bin
  sudo_if_needed curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o /usr/local/bin/cloudflared
  sudo_if_needed chmod +x /usr/local/bin/cloudflared
fi

# ========= NVIDIA Container Toolkit repo =========
distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.repo" \
 | sudo_if_needed tee /etc/yum.repos.d/libnvidia-container.repo >/dev/null

# ========= Install NVIDIA Container Toolkit =========
sudo_if_needed $PKG -y clean expire-cache || true
set +e
sudo_if_needed $PKG -y install \
  "nvidia-container-toolkit-${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
  "nvidia-container-toolkit-base-${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
  "libnvidia-container-tools-${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
  "libnvidia-container1-${NVIDIA_CONTAINER_TOOLKIT_VERSION}"
PIN_RC=$?
set -e
if [ $PIN_RC -ne 0 ]; then
  sudo_if_needed $PKG -y install nvidia-container-toolkit
fi

# ========= Docker: run Unsloth =========
mkdir -p "$WORKDIR"
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  sudo_if_needed docker rm -f "$CONTAINER_NAME"
fi

sudo_if_needed docker run -d --name "$CONTAINER_NAME" \
  -e JUPYTER_PASSWORD="$JUPYTER_PASSWORD" \
  -p 8888:8888 -p 2222:22 \
  -v "$WORKDIR":/workspace/work \
  --gpus all \
  "$DOCKER_IMAGE"

# ========= Cloudflare Tunnel (no login required) =========
echo "Starting Cloudflare Tunnel for port 8888…"
cloudflared tunnel --url http://localhost:8888 --no-autoupdate 2>&1 | tee cloudflared.log &
sleep 3

echo "===================================="
echo "🔗 Public Jupyter URL:"
grep -o "https://.*trycloudflare.com" cloudflared.log | head -n 1
echo "===================================="
echo "Cloudflare log: tail -f cloudflared.log"
