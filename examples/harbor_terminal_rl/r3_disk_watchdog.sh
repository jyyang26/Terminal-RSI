#!/bin/bash
# r3_disk_watchdog.sh
# --------------------------------------------------------------------------
# Read-only disk watchdog for the long R3 + partial-reward run.
#
# WHY THIS EXISTS
#   The run writes two things that grow without bound:
#     * checkpoints  ~455 GB per save point (actor 228 + critic 227, weights-only
#                    because --no-save-optim; it would be ~3.2 TB WITH optimizer)
#     * trial dirs   ~92 GB per rollout (285 trial dirs x agent logs/results)
#   slime has NO checkpoint retention/pruning knob -- every save point is kept
#   forever. And --no-save-optim means the checkpoints CANNOT be resumed from, so a
#   disk-full crash costs the whole run, not just the last few steps.
#
#   At 2.3 TB free that is roughly:  455 + 10*92 = 1375 GB per 10 steps  ->  ~17 steps.
#   The plan is to free/extend space mid-run, so this watchdog exists to make the
#   ceiling VISIBLE early instead of discovering it as a failed ckpt-save.
#
# WHAT IT DOES / DOES NOT DO
#   Prints a projection and a verdict. It NEVER stops the job and NEVER deletes
#   anything -- deletion on a shared volume is a human decision (there is a 10 TB
#   unrelated tmax15k run on this same disk that must NOT be touched).
#
# Usage:
#   examples/harbor_terminal_rl/r3_disk_watchdog.sh
#   WARN_GB=1500 examples/harbor_terminal_rl/r3_disk_watchdog.sh
#   watch -n 600 examples/harbor_terminal_rl/r3_disk_watchdog.sh
set -uo pipefail

_ROOT=/path/to/workspace
BASE="${BASE:-$(ls -1dt "$_ROOT"/slime_harbor_q35_122b_a10b_synthRTS_partialreward_* 2>/dev/null | head -1)}"
TRIALS="${TRIALS:-$(ls -1dt "$_ROOT"/slime_harbor_trials/synth_all_rts_122b_a10b_ppo_partial_reward_* 2>/dev/null | head -1)}"
# Alert threshold in GB of remaining space.
WARN_GB="${WARN_GB:-900}"
# Empirical unit costs (measured on the r3probe run, 2026-08-16).
CKPT_GB_PER_SAVE="${CKPT_GB_PER_SAVE:-455}"
TRIAL_GB_PER_ROLLOUT="${TRIAL_GB_PER_ROLLOUT:-92}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

echo "==================== DISK @ $(date '+%F %T') ===================="
avail_g=$(df -PBG /path/to/shared_storage 2>/dev/null | tail -1 | awk '{gsub(/G/,"",$4); print $4}')
used_pct=$(df -P /path/to/shared_storage 2>/dev/null | tail -1 | awk '{gsub(/%/,"",$5); print $5}')
echo "gy4: ${used_pct}% used, ${avail_g} GB available"

# Progress: how many steps done, how many save points written.
steps=0; saves=0
if [ -n "${BASE:-}" ] && [ -d "$BASE" ]; then
  _log=$(ls -t "$BASE"/launch_122b_*.log 2>/dev/null | head -1)
  if [ -n "$_log" ]; then steps=$(grep -ac 'Timer update_weights end' "$_log" 2>/dev/null); fi
  saves=$(ls -d "$BASE"/*/iter_* 2>/dev/null | grep -v '_critic/' | wc -l)
  echo "run: $(basename "$BASE")"
  echo "     steps_done=${steps:-0}  save_points=${saves:-0}"
fi
if [ -n "${TRIALS:-}" ] && [ -d "$TRIALS" ]; then
  echo "trials: $(basename "$TRIALS")  (dirs=$(ls "$TRIALS" 2>/dev/null | wc -l))"
fi

# Projection: how many MORE steps fit in the remaining space.
python3 - "$avail_g" "$CKPT_GB_PER_SAVE" "$TRIAL_GB_PER_ROLLOUT" "$SAVE_INTERVAL" "$WARN_GB" <<'PY'
import sys
avail, ckpt, trial, interval, warn = (float(x) for x in sys.argv[1:6])
per_step = trial + ckpt / interval           # amortised cost of one step
steps_left = avail / per_step if per_step > 0 else 0
print(f"cost model: {trial:.0f} GB/rollout (trials) + {ckpt:.0f} GB per save every {interval:.0f} steps"
      f"  =>  ~{per_step:.0f} GB per step")
print(f"HEADROOM: about {steps_left:.0f} MORE steps fit in {avail:.0f} GB"
      f"  (~{steps_left*1.5:.0f} h at ~90 min/step)")
if avail < warn:
    print(f"  DISK_ALERT: only {avail:.0f} GB left (< {warn:.0f} GB).")
    print("              --no-save-optim means checkpoints are NOT resumable, so a")
    print("              disk-full ckpt-save failure loses the run. Free space or")
    print("              redirect the save dir BEFORE that happens.")
    print("              Do NOT delete the unrelated 10 TB tmax15k_88cp4-84k run.")
PY
echo "=================================================================="
