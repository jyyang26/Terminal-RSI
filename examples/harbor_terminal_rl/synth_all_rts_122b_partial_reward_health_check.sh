#!/bin/bash
# synth_all_rts_122b_partial_reward_health_check.sh
# --------------------------------------------------------------------------
# Deterministic health probe for the 122B-A10B / RTS-37484 / PPO + CTRF-DENSE-reward
# run on TOPOLOGY=88cp2. Adapted from synth_all_38k_health_check.sh (the 27B/120-GPU
# probe), keeping ALL of its "don't misjudge a healthy job as dead" rules and adding a
# CTRF-hit-rate probe, which is the only direct observation of whether the DENSE REWARD
# is actually doing anything.
#
# It PRINTS a compact status block and a verdict line. It NEVER stops the job.
# The decision to stop is always escalated to a human — this only gathers facts.
#
# Design rules baked in (inherited, hard-won):
#   * File-system mtime is the ground truth for "alive", NOT ray-job-log timestamps
#     (which buffer/lag) and NOT GPU util (0% is NORMAL during rollout while agents
#     solve on remote Daytona).
#   * A "dead" verdict requires MULTIPLE hammers together (crash string in log AND
#     stalled artifact mtime), never a single soft signal.
#   * Probes are cheap and read-only; do NOT py-spy in a loop (it SIGSTOPs the target)
#     and do NOT walk the 37k task pool (observer effect slows the real glob).
#   * Agent chatter is stripped BEFORE crash-grep: the Terminus2 agent runs arbitrary
#     terminal commands whose text routinely contains "RuntimeError"/"Traceback".
#     Those are the AGENT SOLVING A TASK, not a training crash.
#
# 122B-specific notes vs the 27B probe:
#   * CP=2 means a Triton cold-compile storm on the first steps; per-microbatch times
#     of 400s+ decaying to ~20s are NORMAL warm-up, not a hang. Hence STALL_MIN
#     defaults to 45 min here (vs 30) — a legit CP2 compile window can exceed 30 min.
#   * The step to watch is rollout_id 2: NUM_CRITIC_ONLY_STEPS=2, so steps 0-1 are
#     critic-only and rollout_id 2 is the first actor+critic JOINT step — historically
#     where the smaller topologies OOM'd. Clearing it is the real validation.
#
# Usage:  examples/harbor_terminal_rl/synth_all_rts_122b_partial_reward_health_check.sh
#         BASE=/path/to/run ./...sh          # pin a specific run
#         STALL_MIN=60 ./...sh               # loosen the stall threshold
set -uo pipefail

# Auto-resolve to the NEWEST run so this survives restarts (each mints a fresh RUN_TAG).
_ROOT=/path/to/workspace
BASE="${BASE:-$(ls -1dt "$_ROOT"/slime_harbor_q35_122b_a10b_synthRTS_partialreward_* 2>/dev/null | head -1)}"
TRIALS="${TRIALS:-$(ls -1dt "$_ROOT"/slime_harbor_trials/synth_all_rts_122b_a10b_ppo_partial_reward_* 2>/dev/null | head -1)}"
RAY_SESS_LOGS="${RAY_SESS_LOGS:-/tmp/rsynthrts122b/session_latest/logs}"
# RAY_TEMP_DIR is only honoured when THIS run starts the head. On a long-lived cluster
# (CLEAN_START=0 / USE_EXISTING_RAY_HEAD=1) the head keeps whatever --temp-dir it was
# originally started with, so /tmp/rsynthrts122b may not exist at all. Fall back to
# discovering the newest job-driver log anywhere under /tmp.
if [ ! -d "$RAY_SESS_LOGS" ]; then
  _DISCOVERED=$(find /tmp -maxdepth 5 -name 'job-driver-raysubmit_*.log' -printf '%T@ %h\n' 2>/dev/null \
                 | sort -rn | head -1 | cut -d' ' -f2-)
  [ -n "$_DISCOVERED" ] && RAY_SESS_LOGS="$_DISCOVERED"
fi
_AUTOJOB="$(ls -1t "$RAY_SESS_LOGS"/job-driver-raysubmit_*.log 2>/dev/null | head -1 | sed -E 's#.*/job-driver-(raysubmit_[A-Za-z0-9]+)\.log#\1#')"
JOB="${JOB:-${_AUTOJOB:-raysubmit_UNKNOWN}}"
export RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
STALL_MIN="${STALL_MIN:-45}"     # CP2 Triton cold-compile can legitimately take >30min
NOW_EPOCH=$(date +%s)

echo "==================== HEALTH @ $(date '+%F %T') ===================="
echo "base:   ${BASE:-<none found>}"
echo "trials: ${TRIALS:-<none found>}"

# ---- 1. Ray job status (authoritative for FAILED/STOPPED) ------------------
JSTAT=$(ray job status "$JOB" 2>&1 | grep -oE 'RUNNING|FAILED|SUCCEEDED|STOPPED|PENDING' | head -1)
echo "job_status: ${JSTAT:-UNKNOWN}  (job=$JOB)"

# ---- 2. Freshest artifact mtime across the run (GROUND TRUTH for alive) ----
# Cheap mtime probes only, all timeout-guarded so a slow shared disk can never hang
# this check. For trials read the newest SUBDIR mtime (a dir's mtime bumps when its
# files are written) instead of a deep file walk over hundreds of trial dirs.
freshest=0; freshest_path=""
consider() { local ts=${1%.*}; [ -z "$ts" ] && return
  if [ "$ts" -gt "$freshest" ] 2>/dev/null; then freshest=$ts; freshest_path=$2; fi; }
if [ -n "${TRIALS:-}" ] && [ -d "$TRIALS" ]; then
  line=$(timeout 15 find "$TRIALS" -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1)
  [ -n "$line" ] && consider "${line%% *}" "${line#* }"
fi
for scan in "$BASE/rollout_dumps" "$BASE/wandb/wandb"; do
  [ -d "$scan" ] || continue
  line=$(timeout 10 find "$scan" -maxdepth 2 -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1)
  [ -n "$line" ] && consider "${line%% *}" "${line#* }"
done
if [ "$freshest" -gt 0 ]; then
  age_min=$(( (NOW_EPOCH - freshest) / 60 ))
  echo "freshest_artifact: ${age_min}min ago  ($freshest_path)"
else
  age_min=-1
  echo "freshest_artifact: NONE yet (pre-rollout init: 122B torch_dist load + engine cold-start, ~20min+)"
fi

# ---- 3. Checkpoints saved so far (progress marker) -------------------------
CKPT_DIR="$BASE/$(ls "$BASE" 2>/dev/null | grep -E '^Qwen3.5-122B-A10B_synthRTS' | head -1)"
if [ -d "$CKPT_DIR" ]; then
  last_iter=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo '-')
  n_ckpt=$(ls -d "$CKPT_DIR"/iter_* 2>/dev/null | wc -l)
  echo "checkpoints: latest_iter=${last_iter} count=${n_ckpt}  ($CKPT_DIR)"
else
  echo "checkpoints: none yet"
fi

# ---- 4. rollout_dumps count (one .pt per completed rollout step) -----------
echo "rollout_dumps(.pt): $(ls "$BASE/rollout_dumps/"*.pt 2>/dev/null | wc -l)"

# ---- 5. Log clock freshness (secondary; logs LAG so only a weak signal) ----
# Read the driver log FILE, not `ray job logs` (which refetches the whole multi-MB log
# every call and gets slow enough to hang a cron probe).
DRIVERLOG=$(ls -t "$RAY_SESS_LOGS"/job-driver-"$JOB".log 2>/dev/null | head -1)
if [ -n "$DRIVERLOG" ] && [ -f "$DRIVERLOG" ]; then
  LOGSRC_DESC="driver-file"; TAILBUF=$(tail -n 4000 "$DRIVERLOG" 2>/dev/null)
else
  # fall back to the wrapper's own nohup log, then to ray job logs
  WLOG=$(ls -t "$BASE"/launch_122b_*.log 2>/dev/null | head -1)
  if [ -n "$WLOG" ]; then LOGSRC_DESC="wrapper-log"; TAILBUF=$(tail -n 4000 "$WLOG" 2>/dev/null)
  else LOGSRC_DESC="ray-job-logs(fallback)"; TAILBUF=$(ray job logs "$JOB" 2>&1 | tail -n 4000); fi
fi
# The ray CLI's own "Job ... does not exist" / connection errors are ABOUT the probe,
# not from the training job. Left in TAILBUF they surface as a bogus CRASH_HAMMER
# (observed on a clean box before any run existed). Drop the buffer in that case.
if printf '%s\n' "$TAILBUF" | grep -qE 'does not exist|Request failed with status code 404|Failed to connect to Ray'; then
  TAILBUF=""; LOGSRC_DESC="${LOGSRC_DESC}/no-such-job"
fi
logclock=$(printf '%s\n' "$TAILBUF" | grep -oE '\[2026-[0-9-]+ [0-9:]+\]' | tail -1)
echo "latest_log_clock: ${logclock:-none}  (src=${LOGSRC_DESC}; logs may lag FS mtime)"

# ---- 6. Latest meaningful progress line ------------------------------------
prog=$(printf '%s\n' "$TAILBUF" | grep -iE 'rollout_id|perf/|Update weights|eval|save.*checkpoint|loaded checkpoint|reward' | tail -1 | cut -c1-160)
echo "latest_progress: ${prog:-<none>}"
# 88cp2-specific milestones. These are CUMULATIVE counts, so they must be grepped over
# the WHOLE log, not TAILBUF -- the interesting events (engine startup, checkpoint load)
# scroll far out of a 4000-line tail once agent chatter starts flowing.
_FULLLOG=$(ls -t "$BASE"/launch_122b_*.log 2>/dev/null | head -1)
if [ -n "$_FULLLOG" ] && [ -f "$_FULLLOG" ]; then
  echo "milestones: fired_up=$(grep -ac 'server is fired up' "$_FULLLOG")" \
       "loaded_ckpt=$(grep -ac 'successfully loaded checkpoint' "$_FULLLOG")" \
       "train_steps=$(grep -ac 'Timer update_weights end' "$_FULLLOG")" \
       "rollout_perf=$(grep -aoE 'perf [0-9]+:' "$_FULLLOG" | tail -1 | tr -d ':')" \
       "oom=$(grep -acE 'CUDA out of memory|torch\.OutOfMemoryError' "$_FULLLOG")  (oom MUST be 0)"
  # NUM_CRITIC_ONLY_STEPS=2 -> rollout_id 2 is the first actor+critic JOINT step.
  if grep -aq 'Timer update_weights end' "$_FULLLOG"; then
    _JOINT=$(grep -ac 'Timer update_weights end' "$_FULLLOG")
    [ "$_JOINT" -ge 3 ] && echo "  >>> PASSED rollout_id 2 (the first actor+critic joint step) <<<"
  fi
else
  echo "milestones: (wrapper log not found under $BASE)"
fi

# ---- 6b. ★ CTRF HIT RATE — is the DENSE REWARD actually firing? ------------
# partial_reward.test_count_* reads <trial>/verifier/ctrf.json. If that file is absent
# the trajectory silently falls back to the binary terminal reward and the whole dense
# reward becomes a NO-OP. This is invisible in loss curves, so probe it directly.
# Bounded scan (200 newest trial dirs) to avoid an observer effect on a large tree.
if [ -n "${TRIALS:-}" ] && [ -d "$TRIALS" ]; then
  timeout 60 python3 - "$TRIALS" <<'PY' 2>/dev/null || echo "ctrf_probe: (timed out / unavailable)"
import itertools, os, sys, json
root = sys.argv[1]
done = ctrf = 0; passed_tot = 0; partial = 0; vals = []
for e in itertools.islice(os.scandir(root), 200):
    v = os.path.join(e.path, "verifier")
    if not os.path.isdir(v):
        continue
    files = os.listdir(v)
    if not files:
        continue          # verifier never produced anything (agent failed / timed out)
    done += 1
    p = os.path.join(v, "ctrf.json")
    if not os.path.isfile(p):
        continue
    ctrf += 1
    try:
        s = json.load(open(p)).get("results", {}).get("summary", {})
    except Exception:
        continue
    ps = int(s.get("passed", 0) or 0)
    tot = ps + int(s.get("failed", 0) or 0) + int(s.get("other", 0) or 0) + int(s.get("pending", 0) or 0)
    if tot > 0:
        vals.append(ps / 20.0)          # HARBOR_TESTCOUNT_SCALE=20
        passed_tot += ps
        if 0 < ps < tot:
            partial += 1
pct = (100.0 * ctrf / done) if done else 0.0
msg = f"ctrf_probe: {ctrf}/{done} finished trials have ctrf.json ({pct:.0f}%)"
if vals:
    msg += f" | dense reward: n={len(vals)} mean={sum(vals)/len(vals):.3f} max={max(vals):.3f} strictly-partial={partial}"
print(msg)
if done >= 20 and pct < 50:
    print("  CTRF_ALERT: <50% of finished trials emit ctrf.json -> the DENSE REWARD is")
    print("              largely falling back to the binary terminal reward. Check that")
    print("              the task pool is the RTS pool (its test.sh runs pytest --ctrf).")
if vals and partial == 0 and len(vals) >= 20:
    print("  CTRF_NOTE: no strictly-partial trials yet (all all-or-nothing) -> dense")
    print("             credit is present but not yet differentiating; keep watching.")
PY
else
  echo "ctrf_probe: no trials dir yet"
fi

# ---- 7. HARD crash hammers (real signals only; exclude known-benign noise) -
# Strip AGENT and config chatter FIRST (see the header note), then look for
# framework-level (Megatron/SGLang/Ray/NCCL/CUDA) crashes only.
CRASH=$(printf '%s\n' "$TAILBUF" \
  | grep -viE 'Sending keys|tmux_session|HarborTrialWorker|terminus_2\.py|subagent|trial\.log|Answers subagent|server_args=ServerArgs|ServerArgs\(|No CUDA runtime is found|Triton is not supported|max_tokens_to_oom|watchdog_timeout=|_error[a-z_]*=|error_[a-z]*=|abort_on_' \
  | grep -iE 'Traceback \(most recent|CUDA error|NCCL error|NCCL.{0,20}(failed|unhandled|aborted)|Watchdog caught|watchdog.*timeout.*(kill|abort)|SIGABRT|Aborted \(core|core dumped|RuntimeError|CUDA out of memory|torch\.cuda\.OutOfMemoryError|uncorrectable|Xid [0-9]' \
  | tail -6)
if [ -n "$CRASH" ]; then
  echo "CRASH_HAMMERS:"; echo "$CRASH" | sed 's/^/   /' | cut -c1-160
else
  echo "CRASH_HAMMERS: none"
fi

# ---- 7b. Disk watermark on gy4 (ckpts + trials land here) -------------------
# A 122B actor+critic checkpoint with optimizer state is LARGE and SAVE_INTERVAL=10.
# If the disk fills, ckpt-save fails and takes the run down. Alert only; never delete.
DISK_WARN_PCT="${DISK_WARN_PCT:-95}"
disk_line=$(df -P /path/to/shared_storage 2>/dev/null | tail -1)
disk_pct=$(echo "$disk_line" | awk '{gsub(/%/,"",$5); print $5}')
disk_avail=$(df -h /path/to/shared_storage 2>/dev/null | tail -1 | awk '{print $4}')
if [ -n "$disk_pct" ] && [ "$disk_pct" -ge "$DISK_WARN_PCT" ] 2>/dev/null; then
  echo "DISK_ALERT: gy4 at ${disk_pct}% (avail ${disk_avail}) >= ${DISK_WARN_PCT}% -> ESCALATE:"
  echo "            ckpt-save may fail. Options: prune old trial dirs / prune old ckpts /"
  echo "            migrate save dir. Do NOT auto-delete without human OK."
else
  echo "disk_gy4: ${disk_pct}% used (avail ${disk_avail}); warn at ${DISK_WARN_PCT}%"
fi

# ---- 8. VERDICT (layered; a stop needs MULTIPLE hammers, never one soft) ---
echo "----"
verdict="HEALTHY"; reason=""
if [ "$JSTAT" = "FAILED" ] || [ "$JSTAT" = "STOPPED" ]; then
  verdict="DEAD"; reason="ray job status=$JSTAT (authoritative)"
elif [ "$JSTAT" = "SUCCEEDED" ]; then
  verdict="DONE"; reason="ray job status=SUCCEEDED"
elif [ -n "$CRASH" ] && [ "$age_min" -ge "$STALL_MIN" ]; then
  verdict="SUSPECT_DEAD"; reason="crash-hammer in log AND artifacts stalled ${age_min}min (>= ${STALL_MIN}) -> HUMAN REVIEW, do not auto-stop"
elif [ "$age_min" -ge 0 ] && [ "$age_min" -ge "$STALL_MIN" ]; then
  verdict="SLOW_OR_BACKPRESSURE"; reason="artifacts stalled ${age_min}min but NO crash-hammer -> likely slow/backpressure, or a CP2 Triton cold-compile window; keep watching"
elif [ "$age_min" -lt 0 ]; then
  verdict="STARTING"; reason="pre-rollout init (no artifacts yet); 122B + 88 GPU + CP2 startup can take ~30min+"
else
  verdict="HEALTHY"; reason="artifacts fresh (${age_min}min), job RUNNING, no crash-hammer"
fi
echo "VERDICT: $verdict"
echo "REASON:  $reason"
echo "=================================================================="
