#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/start-backend.sh"
"${ROOT_DIR}/scripts/start-frontend.sh"

echo "all services started"
