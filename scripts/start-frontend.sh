#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"
RUN_DIR="${ROOT_DIR}/.run"
LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${RUN_DIR}/frontend.pid"
LOG_FILE="${LOG_DIR}/frontend.log"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if (echo > /dev/tcp/127.0.0.1/5174) >/dev/null 2>&1; then
  echo "port 5174 is already in use. Stop the existing frontend first." >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "frontend already running (pid=${existing_pid})"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if [[ ! -d "${FRONTEND_DIR}" ]]; then
  echo "frontend directory not found: ${FRONTEND_DIR}" >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found." >&2
  exit 1
fi

if [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
  echo "frontend dependencies not installed. Run 'cd frontend && npm install' first." >&2
  exit 1
fi

VITE_BIN="${FRONTEND_DIR}/node_modules/.bin/vite"
if [[ ! -x "${VITE_BIN}" ]]; then
  echo "vite executable not found: ${VITE_BIN}" >&2
  exit 1
fi

cd "${FRONTEND_DIR}"

nohup "${VITE_BIN}" --host 0.0.0.0 --port 5174 --strictPort \
  > "${LOG_FILE}" 2>&1 < /dev/null &

pid=$!
echo "${pid}" > "${PID_FILE}"
sleep 2

if kill -0 "${pid}" 2>/dev/null; then
  echo "frontend started (pid=${pid})"
  echo "log: ${LOG_FILE}"
  echo "url: http://127.0.0.1:5174"
else
  rm -f "${PID_FILE}"
  echo "frontend failed to start. Check ${LOG_FILE}" >&2
  exit 1
fi
