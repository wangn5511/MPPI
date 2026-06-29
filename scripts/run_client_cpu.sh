#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python3 -m mppi.cli client \
  --url "${MPPI_CLIENT_URL:-ws://127.0.0.1:9010}" \
  --run-seconds "${MPPI_RUN_SECONDS:-10}" \
  --control-hz "${MPPI_CONTROL_HZ:-20}" \
  --open-loop-horizon "${MPPI_OPEN_LOOP_HORIZON:-8}"
