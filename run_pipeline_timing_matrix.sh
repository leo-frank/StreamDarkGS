#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec /home/zxn/anaconda3/envs/mvinverse-gsplat-cu118/bin/python tools/benchmark_pipeline.py "$@"
