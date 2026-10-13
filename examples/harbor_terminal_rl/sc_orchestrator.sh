#!/usr/bin/env bash
# sc_orchestrator.sh -- 2h watch-and-heal for the SC-only smoke, auto-launch formal on PASS.
#
# Robust (setsid/nohup, survives disconnect). Coordinates with the interactive session via a
# shared-FS mkdir lock so the formal run is launched exactly once.
#
# CRASH is judged by RAY JOB STATUS + driver-fatal strings + parent liveness -- NOT by generic
# "Traceback", because terminal-RL agents legitimately print tracebacks from sandbox code.
set -uo pipefail

ROOT=/path/to/workspace
REPO=$ROOT/slime-terminal-RL
SMOKE_LOG=$REPO/run_sc_i263c34_smoke_20260929_200808.log
SMOKE_PID=2553591
SMOKE_TRIALDIR=$ROOT/slime_harbor_trials/q38_27b_ppo_256k_SC_ITER263_CRIT34_SMOKE_20260929_200808
STAMP=$(date +%Y%m%d_%H%M%S)
STATE=$ROOT/sc_orch_state.txt
LOCK=$ROOT/sc_formal_launch.lock
ORCH_LOG=$REPO/sc_orchestrator_${STAMP}.log
DEADLINE=$(( $(date +%s) + 2*3600 ))
JID=$(grep -oE 'raysubmit_[A-Za-z0-9]+' "$SMOKE_LOG" | head -1)

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ORCH_LOG"; }
setstate(){ echo "$1" > "$STATE"; log "STATE -> $1"; }

jobstat(){ cd "$REPO" && ray job status "$JID" 2>/dev/null | grep -oiE "Status for job.*: (RUNNING|SUCCEEDED|FAILED|STOPPED|PENDING)" | grep -oiE 'RUNNING|SUCCEEDED|FAILED|STOPPED|PENDING' | head -1; }
parent_alive(){ kill -0 "$SMOKE_PID" 2>/dev/null; }
# driver-fatal strings only (launcher preflight die / rollout produced nothing trainable)
driver_fatal(){ grep -qE '_die:|Harbor rollout produced no trainable slime samples|RuntimeError: .*no trainable' "$SMOKE_LOG"; }
has_trainstep(){ grep -qiE 'model\.py:[0-9]+ - step [0-9]|train/pg_loss|train/value_loss|perf/train_time|perf/actor_train' "$SMOKE_LOG"; }
sc_applied_ok(){ grep -E 'sc_applied' "$SMOKE_LOG" | grep -viE 'grep|confirm SC|expect|watch|orchestrat|Sending keys|HarborTrialWorker' | grep -qE 'sc_applied[^A-Za-z].*(1\.0|0\.9[0-9]| 1$|=1)'; }
gradnorm_ok(){ grep -E 'grad_norm' "$SMOKE_LOG" | grep -viE 'None|nan|inf|grep|watch|Sending keys|HarborTrialWorker' | grep -qE 'grad_norm[^A-Za-z]*[0-9]'; }
daytona_err(){ grep -c 'DaytonaValidationError' "$SMOKE_LOG" 2>/dev/null || echo 0; }
trial_ok(){ find "$SMOKE_TRIALDIR" -name 'result.json' 2>/dev/null | wc -l; }

launch_formal(){
  if ! mkdir "$LOCK" 2>/dev/null; then log "lock held elsewhere -> NOT launching"; setstate "FORMAL_LAUNCHED_BY_OTHER"; return 0; fi
  log "acquired formal-launch lock"
  # 1. stop smoke job (never ray stop --force; keepalive untouched)
  log "stopping smoke ray job ${JID:-<none>}"
  [ -n "$JID" ] && (cd "$REPO" && ray job stop "$JID" >>"$ORCH_LOG" 2>&1)
  kill "$SMOKE_PID" 2>/dev/null
  # 2. wait GPU->0 via ray ledger (not ssh)
  for k in $(seq 1 30); do
    sleep 20
    local u; u=$(cd "$REPO" && ray status 2>/dev/null | grep -oE '[0-9.]+/120\.0 GPU' | head -1)
    log "gpu ledger: ${u:-?}"; echo "$u" | grep -qE '^0\.0/120' && { log "all 120 GPU released"; break; }
  done
  # 3. reclaim ONLY *SMOKE* temp caches (trials + smoke run dirs); never production/loadroots
  log "reclaiming *SMOKE* temp caches"; df -h "$ROOT" 2>/dev/null | tail -1 | tee -a "$ORCH_LOG"
  find "$ROOT/slime_harbor_trials" -maxdepth 1 -type d -name '*SMOKE*' 2>/dev/null | while read -r d; do log "rm trial $d"; rm -rf "$d"; done
  find "$ROOT/slime_harbor_runs"   -maxdepth 1 -type d -name '*SMOKE*' 2>/dev/null | while read -r d; do log "rm run   $d"; rm -rf "$d"; done
  sync
  local AVAIL_G; AVAIL_G=$(df -BG "$ROOT" 2>/dev/null | awk 'NR==2{print $4}' | tr -d G)
  [ -z "$AVAIL_G" ] && AVAIL_G=200
  log "post-cleanup avail=${AVAIL_G}G"
  # 4. disk-budget SAVE_INTERVAL (~1100G/save, 250G headroom)
  local NUM_ROLLOUT=100 budget SAVE_INTERVAL
  budget=$(( (AVAIL_G - 250) / 1100 )); [ "$budget" -lt 1 ] && budget=1
  SAVE_INTERVAL=$(( (NUM_ROLLOUT + budget - 1) / budget )); [ "$SAVE_INTERVAL" -lt 5 ] && SAVE_INTERVAL=5
  log "budget=${budget} saves -> SAVE_INTERVAL=${SAVE_INTERVAL} / NUM_ROLLOUT=${NUM_ROLLOUT}"
  # 5. launch FORMAL (batch 256 default, critic-only 2 default)
  local FSTAMP FORMAL_LOG; FSTAMP=$(date +%Y%m%d_%H%M%S); FORMAL_LOG=$REPO/run_sc_i263c34_FORMAL_${FSTAMP}.log
  echo "$FORMAL_LOG" > "$ROOT/sc_formal_log_path.txt"
  log "launching FORMAL -> $FORMAL_LOG"
  cd "$REPO"
  NUM_CRITIC_ONLY_STEPS=2 NUM_ROLLOUT=$NUM_ROLLOUT SAVE_INTERVAL=$SAVE_INTERVAL \
    RUN_TAG=SC_ITER263_CRIT34_FORMAL_${FSTAMP} \
    setsid nohup bash examples/harbor_terminal_rl/launch_27b_sc_iter263_crit34.sh > "$FORMAL_LOG" 2>&1 &
  log "FORMAL pid=$! log=$FORMAL_LOG"; setstate "FORMAL_LAUNCHED log=$FORMAL_LOG save_interval=$SAVE_INTERVAL"
}

setstate "WATCHING deadline=$(date -d @$DEADLINE +%H:%M) jid=$JID"
log "orchestrator up. deadline=$(date -d @$DEADLINE '+%Y-%m-%d %H:%M')"
while :; do
  now=$(date +%s); st=$(jobstat)
  # crash: job FAILED/STOPPED, or driver-fatal string, or parent dead w/o a train step
  if [ "$st" = "FAILED" ] || [ "$st" = "STOPPED" ] || driver_fatal || { ! parent_alive && ! has_trainstep; }; then
    if [ "$(daytona_err)" -gt 100 ] && [ "$(trial_ok)" -lt 3 ]; then
      setstate "FAIL_DAYTONA_DOWN jobstat=$st"; log "Daytona outage. Deferring to interactive Branch B (mechanical retry)."
    else
      setstate "FAIL_CRASH jobstat=$st"; log "Training crash (jobstat=$st). Deferring diagnosis to interactive Branch B."
      grep -E '_die:|produced no trainable|RayActorError|CUDA out of memory|NCCL' "$SMOKE_LOG" | grep -viE 'HarborTrialWorker|Sending keys|preflight' | tail -12 >> "$ORCH_LOG"
    fi
    break
  fi
  # pass: real train step (or job succeeded = all 3 steps done), no crash
  if has_trainstep || [ "$st" = "SUCCEEDED" ]; then
    if sc_applied_ok && gradnorm_ok; then
      setstate "PASS_DEFINITIVE jobstat=$st"; log "PASS: train step + sc_applied + finite grad_norm."; launch_formal; break
    else
      setstate "PASS_NEEDS_REVIEW jobstat=$st"; log "Train step/succeeded but sc/grad not parsed; keep watching for metric print."
      # if the job already fully succeeded, don't wait forever
      if [ "$st" = "SUCCEEDED" ]; then log "job SUCCEEDED -> proceeding to formal on train-step evidence."; launch_formal; break; fi
    fi
  fi
  if [ "$now" -ge "$DEADLINE" ]; then
    if has_trainstep && [ "$st" != "FAILED" ]; then setstate "DEADLINE_PASSish jobstat=$st"; log "2h: train step seen -> formal (fallback)."; launch_formal
    else setstate "DEADLINE_INCONCLUSIVE jobstat=$st"; log "2h: inconclusive (jobstat=$st) -> hold for review."; tail -20 "$SMOKE_LOG" >> "$ORCH_LOG"; fi
    break
  fi
  if [ $(( (now/60) % 3 )) -eq 0 ]; then
    log "hb jobstat=$st loglines=$(wc -l < "$SMOKE_LOG") trials_ok=$(trial_ok) daytona_err=$(daytona_err) trainstep=$(has_trainstep && echo Y || echo n)"
  fi
  sleep 60
done
log "orchestrator exiting. final: $(cat "$STATE")"
