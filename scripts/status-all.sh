#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/.run"
LOG_DIR="${ROOT_DIR}/logs"

show_status() {
  local name="$1"
  local port="$2"
  local pid_file="${RUN_DIR}/${name}.pid"
  local log_file="${LOG_DIR}/${name}.log"

  if [[ ! -f "${pid_file}" ]]; then
    if (echo > /dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1; then
      echo "${name}: listening on ${port} (not tracked by script)"
    else
      echo "${name}: stopped"
    fi
    return
  fi

  local pid
  pid="$(cat "${pid_file}")"

  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "${name}: running (pid=${pid})"
    echo "${name} log: ${log_file}"
  else
    if (echo > /dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1; then
      echo "${name}: listening on ${port} with stale pid file (${pid})"
    else
      echo "${name}: stale pid file (${pid})"
    fi
  fi
}

show_status "backend" "8000"
show_status "frontend" "5174"
