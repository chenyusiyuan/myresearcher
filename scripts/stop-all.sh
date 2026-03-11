#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/.run"
FRONTEND_PATTERN="${ROOT_DIR}/frontend/node_modules/.bin/vite --host 0.0.0.0 --port 5174 --strictPort"

stop_service() {
  local name="$1"
  local pid_file="${RUN_DIR}/${name}.pid"

  if [[ ! -f "${pid_file}" ]]; then
    echo "${name} not running (no pid file)"
    return
  fi

  local pid
  pid="$(cat "${pid_file}")"

  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    echo "${name} pid file was empty"
    return
  fi

  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${pid_file}"
    echo "${name} already stopped"
    return
  fi

  pkill -P "${pid}" 2>/dev/null || true
  kill "${pid}" 2>/dev/null || true

  for _ in {1..10}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${pid_file}"
      echo "${name} stopped"
      return
    fi
    sleep 1
  done

  kill -9 "${pid}" 2>/dev/null || true
  rm -f "${pid_file}"
  echo "${name} killed"
}

stop_frontend_residuals() {
  local pids
  pids="$(pgrep -f "${FRONTEND_PATTERN}" || true)"

  if [[ -z "${pids}" ]]; then
    return
  fi

  echo "${pids}" | xargs -r kill 2>/dev/null || true
  sleep 1

  pids="$(pgrep -f "${FRONTEND_PATTERN}" || true)"
  if [[ -n "${pids}" ]]; then
    echo "${pids}" | xargs -r kill -9 2>/dev/null || true
  fi
}

stop_service "frontend"
stop_frontend_residuals
stop_service "backend"
