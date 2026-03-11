#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/backend"
RUN_DIR="${ROOT_DIR}/.run"
LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${RUN_DIR}/backend.pid"
LOG_FILE="${LOG_DIR}/backend.log"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if (echo > /dev/tcp/127.0.0.1/8000) >/dev/null 2>&1; then
  echo "port 8000 is already in use. Stop the existing backend first." >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "backend already running (pid=${existing_pid})"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if [[ ! -d "${BACKEND_DIR}" ]]; then
  echo "backend directory not found: ${BACKEND_DIR}" >&2
  exit 1
fi

if [[ -x "${BACKEND_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${BACKEND_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "python not found. Create backend/.venv or install python3." >&2
  exit 1
fi

cd "${BACKEND_DIR}"

nohup env PYTHONPATH="${BACKEND_DIR}/src" PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" -m uvicorn main:app --host 0.0.0.0 --port 8000 \
  > "${LOG_FILE}" 2>&1 < /dev/null &

pid=$!
echo "${pid}" > "${PID_FILE}"
sleep 1

if kill -0 "${pid}" 2>/dev/null; then
  echo "backend started (pid=${pid})"
  echo "log: ${LOG_FILE}"
else
  rm -f "${PID_FILE}"
  echo "backend failed to start. Check ${LOG_FILE}" >&2
  exit 1
fi
