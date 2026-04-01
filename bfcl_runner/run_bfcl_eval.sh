#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${SCRIPT_DIR}/example_config.json"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="$1"
  shift
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "Using config: ${CONFIG_PATH}"
exec python3 "${SCRIPT_DIR}/run_bfcl_eval.py" --config "${CONFIG_PATH}" "$@"
