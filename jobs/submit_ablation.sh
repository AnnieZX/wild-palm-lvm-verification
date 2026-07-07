#!/usr/bin/env bash
# Deprecated wrapper — use jobs/submit_qwen_ablation.sh
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_qwen_ablation.sh" "$@"
