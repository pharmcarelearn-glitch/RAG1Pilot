#!/usr/bin/env bash
# Installs the rag1pilot agent worker as a systemd service on a small Linux VM
# (Ubuntu 22.04+ or Debian 12+). Run from inside a clone of the repository:
#
#   sudo MISTRAL_API_KEY=<your-key> bash scripts/setup-vm.sh
#
# Overridable variables: REPO_DIR, SERVICE_NAME (default rag1pilot-agent),
# RUN_USER (default rag1pilot).
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-rag1pilot-agent}"
RUN_USER="${RUN_USER:-rag1pilot}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_DIR="/etc/${SERVICE_NAME}"
ENV_FILE="${ENV_DIR}/.env"
UV_CACHE_DIR="/var/cache/${SERVICE_NAME}-uv"
UV_PYTHON_DIR="${REPO_DIR}/.uv-python"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root: sudo bash scripts/setup-vm.sh" >&2
  exit 1
fi

if [ -z "${MISTRAL_API_KEY:-}" ]; then
  read -rsp "MISTRAL_API_KEY (input hidden): " MISTRAL_API_KEY
  echo
fi
if [ -z "${MISTRAL_API_KEY}" ]; then
  echo "MISTRAL_API_KEY is required." >&2
  exit 1
fi

echo "==> Installing system packages"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq git curl ca-certificates
fi

echo "==> Installing uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh
fi
UV_BIN="$(command -v uv)"

echo "==> Creating service user '${RUN_USER}'"
id -u "${RUN_USER}" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "${RUN_USER}"

echo "==> Writing ${ENV_FILE}"
mkdir -p "${ENV_DIR}" "${UV_CACHE_DIR}"
if [ ! -f "${ENV_FILE}" ]; then
  printf 'MISTRAL_API_KEY=%s\n' "${MISTRAL_API_KEY}" > "${ENV_FILE}"
else
  echo "    (already exists, keeping it)"
fi
chmod 600 "${ENV_FILE}"
chown root:root "${ENV_FILE}"
chown root:root "${ENV_DIR}"

echo "==> Preparing ${REPO_DIR}"
chown -R "${RUN_USER}:${RUN_USER}" "${REPO_DIR}"
sudo -u "${RUN_USER}" env \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  UV_PYTHON_INSTALL_DIR="${UV_PYTHON_DIR}" \
  "${UV_BIN}" sync --frozen --no-dev

echo "==> Installing systemd unit ${SERVICE_NAME}"
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=rag1pilot Mistral agent worker (agents serve)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_USER}
WorkingDirectory=${REPO_DIR}
EnvironmentFile=${ENV_FILE}
Environment=UV_CACHE_DIR=${UV_CACHE_DIR}
Environment=UV_PYTHON_INSTALL_DIR=${UV_PYTHON_DIR}
ExecStart=${UV_BIN} run --frozen --env-file ${ENV_FILE} agents serve
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

echo "==> Done. Check it:"
echo "    systemctl status ${SERVICE_NAME}"
echo "    journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "Once the worker registers, the 'Try the agent' Vibe Code Web link is in the logs:"
echo "    journalctl -u ${SERVICE_NAME} --no-pager | grep 'Try the agent'"
