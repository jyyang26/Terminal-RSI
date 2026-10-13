#!/bin/bash
# Convert a Megatron torch_dist ACTOR checkpoint -> loadable HF safetensors.
#
# See README_convert_qwen35_moe_to_hf.md in this directory for the full rationale,
# the incident history behind every guard below, and troubleshooting.
#
# Usage:
#   CKPT=<megatron --save dir> OUT_ROOT=<where HF dirs land> STEPS="9 19" ./convert_qwen35_actor_to_hf.sh
#
# Two mandatory steps (step 1 alone produces an UNLOADABLE checkpoint):
#   1. convert_torch_dist_to_hf_parallel.py  -> raw HF (per-expert layout, no mtp/visual)
#   2. 210_repair_qwen35_moe_converted_hf.py -> fuse experts, restore mtp/visual, fix dtypes
#
# Deliberately NOT 211_roll_optimizer_off_old_checkpoints.py: that tool DELETES the
# torch_dist tree after converting. This script never deletes it.

set -uo pipefail

# ---- required ----------------------------------------------------------------
CKPT="${CKPT:?set CKPT=<megatron --save dir containing iter_XXXXXXX/>}"
OUT_ROOT="${OUT_ROOT:-$(dirname "$CKPT")}"
STEPS="${STEPS:?set STEPS=\"9 19\" (iteration numbers, space separated)}"

# ---- tunables ----------------------------------------------------------------
REPO="${REPO:-/path/to/workspace/slime-terminal-RL}"
ORIGIN="${ORIGIN:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B}"
TAG="${TAG:-q35_122b}"
# WORKERS is the main lever on peak RAM: each --load-max-workers process holds its
# slice of the 227GB state dict PLUS its converted copy. Measured peaks:
#   16 workers -> >1TB (OOM-killed, deadlocked the Pool)
#    6 workers -> 831GB   (idle node only)
#    2 workers -> 400GB   (safe alongside a live training job)
WORKERS="${WORKERS:-2}"
SAVE_WORKERS="${SAVE_WORKERS:-4}"
STALL_SECS="${STALL_SECS:-3600}"
CG=/sys/fs/cgroup/memory/ckptconv
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- guard 1: memory containment --------------------------------------------
# HARD requirement when sharing the Ray head node. On 2026-08-15 the cgroup did not
# exist (it is not persistent across container restarts), the join silently degraded
# to a warning, the conversion ran uncontained, and Ray's node-memory monitor killed
# a MegatronTrainRayActor at 2180/2265GB -- taking down an 84-GPU PPO run.
if [ ! -d "$CG" ]; then
  echo "ERROR: $CG does not exist. Create it first:" >&2
  echo "         LIMIT_GB=400 ckptconv_cgroup_setup.sh" >&2
  echo "       Or set ALLOW_UNCONTAINED=1 (only when NO training is running)." >&2
  [ "${ALLOW_UNCONTAINED:-0}" = "1" ] || exit 1
elif ! echo $$ > "$CG/cgroup.procs" 2>/dev/null; then
  echo "ERROR: could not join $CG (not root?)." >&2
  [ "${ALLOW_UNCONTAINED:-0}" = "1" ] || exit 1
else
  echo "[cgroup] joined ckptconv (limit $(( $(cat "$CG/memory.limit_in_bytes") /1024/1024/1024 ))GB)"
fi

# ---- guard 2: don't compete with a live trainer -----------------------------
if pgrep -f "[t]rain_async.py" >/dev/null 2>&1; then
  echo "WARN: train_async.py is RUNNING on this node." >&2
  echo "      A concurrent conversion crashed the 88cp4-84k run twice (08-14, 08-15)." >&2
  echo "      Set ALLOW_CONCURRENT_TRAINING=1 to proceed inside the cgroup cap." >&2
  [ "${ALLOW_CONCURRENT_TRAINING:-0}" = "1" ] || exit 1
fi

cd "$REPO" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX=/tmp/pycache_convert
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""   # conversion is pure CPU

# Run a step under a stall watchdog; returns the step rc, or 99 if it stalled.
run_watched() {
  local steplog="$1"; shift
  : > "$steplog"
  nice -n 19 "$@" >> "$steplog" 2>&1 &
  local pid=$! last=0 now=0 idle=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    now=$(stat -c %Y "$steplog" 2>/dev/null || echo 0)
    if [ "$last" -ne 0 ] && [ "$now" -eq "$last" ]; then
      idle=$(( $(date +%s) - now ))
      if [ "$idle" -ge "$STALL_SECS" ]; then
        echo "[STALL] no output for ${idle}s -- killing $pid and its workers"
        pkill -9 -P "$pid" 2>/dev/null; kill -9 "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null; return 99
      fi
    fi
    last=$now
  done
  wait "$pid"
}

for STEP in $STEPS; do
  ITER_DIR="$CKPT/$(printf 'iter_%07d' "$STEP")"
  RAW="$OUT_ROOT/${TAG}_iter${STEP}_hf_raw"
  OUT="$OUT_ROOT/${TAG}_iter${STEP}_hf"

  echo "================ iter_$STEP  $(date) ================"
  [ -d "$ITER_DIR" ] || { echo "[skip] missing $ITER_DIR"; continue; }
  [ -f "$OUT/model.safetensors.index.json" ] && { echo "[skip] $OUT already present"; continue; }

  echo "--- step 1/2 torch_dist -> raw HF ($(date +%H:%M:%S), $WORKERS load workers)"
  run_watched "/tmp/conv_step1_iter${STEP}.log" \
    /usr/bin/python3 "$TOOLS/convert_torch_dist_to_hf_parallel.py" \
      --input-dir "$ITER_DIR" --output-dir "$RAW" --origin-hf-dir "$ORIGIN" \
      --load-max-workers "$WORKERS" --save-max-workers "$SAVE_WORKERS" --force
  rc=$?
  tail -3 "/tmp/conv_step1_iter${STEP}.log"
  if [ $rc -ne 0 ]; then
    echo "[FAIL] step1 iter_$STEP rc=$rc$([ $rc -eq 99 ] && echo ' (STALLED)')"
    rm -rf "$RAW"; continue
  fi

  # RAWCHECK: an OOM-killed worker leaves multiprocessing.Pool blocked forever while
  # its worker index still NAMES shards that were never written. Without this gate the
  # repair step would happily build a plausible-looking but corrupt checkpoint.
  /usr/bin/python3 - "$RAW" <<'PY'
import glob, json, os, sys
raw = sys.argv[1]
ref, keys = set(), set()
idx = glob.glob(os.path.join(raw, "*model.safetensors.index.json"))
for f in idx:
    wm = json.load(open(f))["weight_map"]
    ref |= set(wm.values()); keys |= set(wm)
absent = sorted(f for f in ref if not os.path.exists(os.path.join(raw, f)))
print(f"RAWCHECK indexes={len(idx)} keys={len(keys)} shards={len(ref)} absent={len(absent)}")
if absent:
    print("  absent e.g.", absent[:5])
sys.exit(1 if absent else 0)
PY
  [ $? -eq 0 ] || { echo "[FAIL] step1 iter_$STEP raw dir incomplete -- NOT repairing"; continue; }

  echo "--- step 2/2 repair (fuse experts + restore mtp/visual) ($(date +%H:%M:%S))"
  run_watched "/tmp/conv_step2_iter${STEP}.log" \
    /usr/bin/python3 "$TOOLS/210_repair_qwen35_moe_converted_hf.py" \
      --origin-hf-dir "$ORIGIN" --converted-hf-dir "$RAW" --output-dir "$OUT" --force
  rc=$?
  tail -6 "/tmp/conv_step2_iter${STEP}.log"
  [ $rc -eq 0 ] || { echo "[FAIL] step2 iter_$STEP rc=$rc (raw kept at $RAW)"; continue; }

  # VERIFY: an ACTOR must match origin's key set and total_size byte-for-byte.
  # (Only a CRITIC may legitimately differ -- by exactly one lm_head.)
  /usr/bin/python3 - "$ORIGIN" "$OUT" <<'PY'
import json, sys
from pathlib import Path
origin, out = Path(sys.argv[1]), Path(sys.argv[2])
o = json.loads((origin / "model.safetensors.index.json").read_text())
r = json.loads((out / "model.safetensors.index.json").read_text())
ok = set(o["weight_map"]) == set(r["weight_map"])
so, sr = int(o["metadata"]["total_size"]), int(r["metadata"]["total_size"])
absent = [f for f in set(r["weight_map"].values()) if not (out / f).is_file()]
print(f"VERIFY keys={len(r['weight_map'])}/{len(o['weight_map'])} keys_equal={ok} "
      f"total_size={sr} delta={so-sr} missing_shards={len(absent)}")
sys.exit(0 if (ok and so == sr and not absent) else 1)
PY
  [ $? -eq 0 ] || { echo "[FAIL] verification iter_$STEP -- raw kept at $RAW"; continue; }

  rm -rf "$RAW"   # intermediate only, superseded by the repaired dir
  echo "[ok] iter_$STEP -> $OUT   (torch_dist $ITER_DIR left untouched)"
  df -h "$OUT_ROOT" | tail -1
done

echo "================ all done $(date) ================"
ls -d "$OUT_ROOT/${TAG}"_iter*_hf 2>/dev/null
