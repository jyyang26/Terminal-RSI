#!/bin/bash
# setup_new_nodes.sh -- bring freshly-allocated nodes up to the cluster baseline.
#
#   bash envs/setup_new_nodes.sh 29.1.2.3 29.1.2.4 ...      # explicit IPs
#   bash envs/setup_new_nodes.sh --from-node-ip-list        # every IP in $NODE_IP_LIST
#                                                            that is not already in Ray
#
# WHY THIS EXISTS
#   /usr/local/lib/python3.12/dist-packages lives on overlayfs (NODE-LOCAL), not on the
#   shared filesystem. So every new node starts with a DIFFERENT package set and must be
#   patched up individually. History (envs/README_cluster_env_setup_zh.md §6): a new node
#   passed a 25-package version table, then still died in rollout with
#   `ModuleNotFoundError: shortuuid`; fixed that, then `litellm` was missing; fixed that,
#   then `dirhash` -- which harbor imports lazily INSIDE a function
#   (harbor/models/task/task.py:168), so no module-level probe can see it. The new node
#   turned out to be short 98 packages.
#
#   CONCLUSION, and the rule this script implements: do NOT hand-list package names.
#   Diff the ENTIRE package set against a worker that is already PROVEN to run rollout,
#   and install everything that is missing, with --no-deps so pip cannot "helpfully"
#   upgrade torch and desync the cluster.
#
# The reference is a known-good WORKER, never the head: the head carries extra dev tools
# (gradio / nemo / sklearn ...) and using it as the baseline makes every worker look broken.
set -uo pipefail

REF_IP="${REF_IP:-29.232.240.247}"     # proven rollout-capable worker
HEAD_IP="${HEAD_IP:-29.127.33.153}"
RAY_PORT="${RAY_PORT:-6379}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes"

# ---- resolve target list ------------------------------------------------------------
if [ "${1:-}" = "--from-node-ip-list" ]; then
  mapfile -t ALL < <(echo "${NODE_IP_LIST:-}" | tr ',' '\n' | sed 's/:.*//' | grep -E '^[0-9]')
  # already-joined nodes, so we only touch genuinely new ones.
  # NOTE: `ray status` prints node IDs, not IPs -- it cannot be grepped for addresses.
  # ray.nodes() is the only reliable source of joined node IPs.
  mapfile -t JOINED < <(timeout 90 python3 -c "
import ray
ray.init(address='auto', ignore_reinit_error=True, logging_level='ERROR')
print('\n'.join(sorted({n['NodeManagerAddress'] for n in ray.nodes() if n.get('Alive')})))
" 2>/dev/null)
  TARGETS=()
  for ip in "${ALL[@]}"; do
    skip=0
    for j in "${JOINED[@]:-}"; do [ "$ip" = "$j" ] && skip=1; done
    [ "$ip" = "$HEAD_IP" ] && skip=1
    [ "$skip" = 0 ] && TARGETS+=("$ip")
  done
else
  TARGETS=("$@")
fi

if [ ${#TARGETS[@]} -eq 0 ]; then
  echo "No target nodes. Pass IPs explicitly, or set NODE_IP_LIST and use --from-node-ip-list."
  echo "Current NODE_IP_LIST has $(echo "${NODE_IP_LIST:-}" | tr ',' '\n' | grep -c '^[0-9]') entries."
  exit 1
fi
echo "=== targets (${#TARGETS[@]}): ${TARGETS[*]}"
echo "=== reference worker: ${REF_IP}"
echo

# ---- 0) reachability ----------------------------------------------------------------
UNREACHABLE=()
for ip in "${TARGETS[@]}"; do
  n=$($SSH "$ip" 'nvidia-smi -L 2>/dev/null | wc -l' 2>/dev/null | tail -1)
  if [ -z "$n" ] || [ "$n" = "0" ]; then
    echo "  $ip  UNREACHABLE or 0 GPU"; UNREACHABLE+=("$ip")
  else
    echo "  $ip  ${n} GPU"
  fi
done
if [ ${#UNREACHABLE[@]} -gt 0 ]; then
  echo "ERROR: cannot reach: ${UNREACHABLE[*]}" >&2; exit 2
fi
echo

# ---- 1) capture the reference package set ------------------------------------------
REFLIST=$(mktemp)
$SSH "$REF_IP" 'pip list --format=freeze 2>/dev/null' | sort > "$REFLIST"
REFN=$(grep -c '==' "$REFLIST" || true)
[ "${REFN:-0}" -lt 50 ] && { echo "ERROR: reference pip list looks wrong (${REFN} pkgs)" >&2; exit 3; }
echo "=== reference has ${REFN} packages"

# ---- 2) per node: diff the FULL set, install what is missing ------------------------
for ip in "${TARGETS[@]}"; do
  echo
  echo "================ $ip ================"
  NEWLIST=$(mktemp)
  $SSH "$ip" 'pip list --format=freeze 2>/dev/null' | sort > "$NEWLIST"
  echo "  has $(grep -c '==' "$NEWLIST" || true) packages"

  SPEC=$(REF="$REFLIST" NEW="$NEWLIST" python3 - <<'PY'
import os
norm = lambda n: n.lower().replace('_', '-')
ref = {}
for l in open(os.environ['REF']):
    if '==' in l:
        ref[norm(l.split('==')[0])] = l.strip()
new = {norm(l.split('==')[0]) for l in open(os.environ['NEW']) if '==' in l}
missing = [ref[k] for k in sorted(set(ref) - new)]
print(' '.join(missing))
PY
)
  NMISS=$(echo "$SPEC" | wc -w)
  echo "  missing: ${NMISS} packages"
  if [ "${NMISS}" -gt 0 ]; then
    echo "  installing (--no-deps, so torch is never upgraded)..."
    # --no-deps is essential: a transitive resolve here can bump torch/triton on ONE node
    # and desync the cluster, which shows up much later as a numerics/NCCL mismatch.
    $SSH "$ip" "pip install --no-deps --quiet ${SPEC}" 2>&1 | tail -5
  fi

  # fla is the CP>1 hard dependency and must be EXACTLY 0.4.2 on every node
  # (0.4.1 has no fla.ops.cp at all -> every GDN layer raises at its first forward,
  #  ~20 min into the run, after the checkpoint load).
  echo "  pinning fla 0.4.2 + triton cache dirs..."
  $SSH "$ip" '
    pip install -U --no-deps --quiet "fla-core==0.4.2" "flash-linear-attention==0.4.2"
    mkdir -p /root/.triton_cache_122b_cp2 /root/.triton_cache_122b_cp4
    python3 -c "import fla, fla.ops.cp; print(\"    fla\", fla.__version__, \"+ fla.ops.cp OK\")"
  ' 2>&1 | tail -3
done

# ---- 3) verify ----------------------------------------------------------------------
echo
echo "================ verification ================"
echo "Running the full env self-check against the reference worker."
echo "Every target MUST come back IDENTICAL before you launch training."
bash "$(dirname "${BASH_SOURCE[0]}")/verify_node_env.sh" "${TARGETS[@]}"
RC=$?
echo
if [ $RC -eq 0 ]; then
  echo "ALL TARGETS IDENTICAL. Next: join them to Ray, e.g."
  for ip in "${TARGETS[@]}"; do
    echo "  ssh $ip 'ray stop --force; sleep 3; ray start --address=${HEAD_IP}:${RAY_PORT} --num-gpus 8'"
  done
  echo "Then confirm the size:  ray status | grep GPU"
else
  echo "MISMATCH REMAINS -- do NOT launch training yet (see the diff above)."
fi
exit $RC
