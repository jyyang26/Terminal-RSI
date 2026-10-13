#!/bin/bash
set -euo pipefail
TAG="${1:-$(cat /tmp/tb2_pass3_fixed_tag)}"
BASE="/path/to/workspace/slime-terminal-RL/checkpoints/eval_only/35b-a3b-tp8/${TAG}"
TRIALS="/path/to/workspace/slime_harbor_trials/tb2_pass3_fixed_${TAG}"
LOG="${BASE}/logs/launcher.log"; METRICS="${TRIALS}/eval_0_metrics.json"
echo "tag=${TAG}"
[ ! -f "${LOG}" ] || { grep -aE 'eval 0:|Harbor eval metrics artifact|timed_out_groups|Parser warnings|Trial .*failed|Traceback' "${LOG}" | tail -100; }
if [ -s "${METRICS}" ]; then
  python3 -m json.tool "${METRICS}"
  python3 "$(dirname "$0")/tb2_pass3_gate.py" "${METRICS}" --min 0.35 --max 0.45
else
  printf 'eval_trial_dirs='; find "${TRIALS}" -maxdepth 1 -type d -name 'slime-eval-r0-*' 2>/dev/null | wc -l
  printf 'result_files='; find "${TRIALS}" -maxdepth 3 -type f -name result.json 2>/dev/null | wc -l
fi
