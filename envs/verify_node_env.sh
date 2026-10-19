#!/bin/bash
# 集群环境一致性自检。
#
#   REF_IP=<ref-worker-ip> HEAD_IP=<head-ip> ALL_NODE_IPS="<ip1> <ip2> ..." \
#     bash envs/verify_node_env.sh             # 检查全部节点，与参考节点做 diff
#   REF_IP=... HEAD_IP=... bash envs/verify_node_env.sh <node-ip>   # 只检查一个节点
#
# 退出码 0 = 全部一致且无缺失；非 0 = 有节点不一致（详情见输出）。
#
# 为什么需要它：/usr/local/lib/python3.12/dist-packages 在 overlayfs（节点本地）上，
# 不是共享 fs，所以每个节点都要独立装依赖。历史上新节点缺包直接崩过训练。
set -uo pipefail

# 基准用一个【已验证能跑 rollout 的旧 worker】而非 head：head 上装了大量 dev 工具
# (gradio/nemo/sklearn...)，拿它当基准会让每个 worker 都误报缺包。
REF_IP="${REF_IP:?请指定参考节点 IP (REF_IP=<proven rollout worker>)}"
HEAD_IP="${HEAD_IP:?请指定 Ray Head 节点 IP (HEAD_IP=<head>)}"
# 全部节点 IP（空格或逗号分隔）；只检查单个节点时可省略，直接把 IP 作为参数传入。
# 平台 hostfile 通常是最可靠的来源，例如: ALL_NODE_IPS="$(awk '{print $1}' /path/to/hostfile)"
read -r -a ALL_IPS <<< "$(echo "${ALL_NODE_IPS:-}" | tr ',' ' ')"
if [ $# -eq 0 ] && [ ${#ALL_IPS[@]} -eq 0 ]; then
  echo "请通过 ALL_NODE_IPS 或命令行参数指定待检查的节点 IP" >&2
  exit 2
fi

# ---- 探针：在目标节点上采集环境指纹 ----------------------------------------
PROBE=$(cat <<'PROBE_EOF'
python3 - <<'PY'
import importlib, os, subprocess, sys
r = {}
def rec(name, fn):
    try: r[name] = str(fn())
    except Exception as e: r[name] = f"ERR:{type(e).__name__}"

import torch
rec('torch',   lambda: torch.__version__)
rec('cuda',    lambda: torch.version.cuda)
rec('gpus',    lambda: torch.cuda.device_count())
rec('gpu',     lambda: torch.cuda.get_device_name(0))
rec('nccl',    lambda: '.'.join(map(str, torch.cuda.nccl.version())))
import triton; rec('triton', lambda: triton.__version__)
import fla;    rec('fla',    lambda: fla.__version__)
rec('fla_ops_cp', lambda: 'OK' if importlib.import_module('fla.ops.cp') else '?')

def pkg(mod, pipname=None):
    def g():
        try:
            m = importlib.import_module(mod)
            return getattr(m, '__version__', 'present')
        except Exception:
            out = subprocess.run([sys.executable, '-m', 'pip', 'show',
                                  pipname or mod.replace('.', '-').replace('_', '-')],
                                 capture_output=True, text=True).stdout
            for line in out.splitlines():
                if line.startswith('Version:'):
                    return line.split()[1]
            return 'MISSING'
    return g

for mod, pipname in [('fla_core','fla-core'), ('flash_linear_attention','flash-linear-attention'),
                     ('transformers',None), ('sglang',None), ('ray',None), ('wandb',None),
                     ('tenacity',None), ('daytona',None), ('megatron.core','megatron-core'),
                     ('transformer_engine',None), ('flash_attn',None),
                     ('torch_memory_saver',None), ('numpy',None), ('pybase64',None)]:
    rec(mod, pkg(mod, pipname))

rec('py',       lambda: sys.version.split()[0])
rec('driver',   lambda: subprocess.run(['nvidia-smi','--query-gpu=driver_version',
                                        '--format=csv,noheader'],
                                       capture_output=True, text=True).stdout.strip().split('\n')[0])
rec('ram_GB',   lambda: int(os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/1024**3))

# Megatron 的 routing-replay patch 是否在位（routing replay 的硬前提）
def patched():
    hits = 0
    for f, needle in [('/root/Megatron-LM/megatron/core/transformer/moe/moe_utils.py',
                       'get_routing_replay_compute_topk'),
                      ('/root/Megatron-LM/megatron/core/transformer/moe/router.py',
                       'register_routing_replay')]:
        try:
            hits += needle in open(f).read()
        except Exception:
            pass
    return {2: 'OK', 1: 'PARTIAL', 0: 'MISSING'}[hits]
rec('mcore_routing_patch', patched)

# 共享 fs 可达性
paths = {
  'hf':        '/path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B/config.json',
  'torchdist': '/path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B_torch_dist/release/.metadata',
  'traindata': '/path/to/workspace/slime-terminal-RL_data/tmax15k_terminal_bench',
  'evaldata':  '/path/to/workspace/terminal-bench-2-main',
  'slime':     '/path/to/workspace/slime-terminal-RL/train_async.py',
  'harbor':    '/path/to/workspace/harbor-code-synthesis/src',
}
rec('sharedfs', lambda: 'OK' if all(os.path.exists(p) for p in paths.values())
                        else 'MISSING:' + ','.join(k for k, p in paths.items() if not os.path.exists(p)))

# Triton 持久化缓存目录（防冷编译风暴，必须节点本地）
rec('triton_cache', lambda: 'OK' if all(os.path.isdir(f'/root/.triton_cache_122b_cp{c}')
                                        for c in (2, 4)) else 'MISSING')

# END-TO-END IMPORT PROBE. Enumerating package names by hand misses TRANSITIVE deps:
# on 2026-08-12 both new nodes passed a 25-package check and still died in rollout with
# `ModuleNotFoundError: shortuuid` (a dep of slime_plugins.harbor.agent). Importing the
# real entrypoints is the only check that actually covers the dependency closure.
def harbor_imports():
    import sys
    for extra in ('/root/Megatron-LM',
                  '/path/to/workspace/harbor-code-synthesis/src',
                  '/path/to/workspace/slime-terminal-RL'):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    bad = []
    for mod in ('slime_plugins.harbor.trial_worker', 'slime_plugins.harbor.rollout',
                'slime_plugins.harbor.agent', 'slime_plugins.harbor.atif_adapter',
                'slime_plugins.harbor.llm', 'slime_plugins.models.cp_utils',
                'slime_plugins.harbor.length_adaptive_gae'):
        try:
            importlib.import_module(mod)
        except Exception as e:
            bad.append(f'{mod.split(".")[-1]}:{getattr(e, "name", None) or type(e).__name__}')
    return 'OK' if not bad else 'FAIL(' + ','.join(bad) + ')'
rec('harbor_imports', harbor_imports)

# FULL PIP-SET FINGERPRINT. This is the check that actually works.
# The import probe above still missed `dirhash`, because harbor imports it LAZILY
# inside a function body (harbor/models/task/task.py:168) -- no module-level import
# can ever see it. Comparing the whole installed package set against the head node
# is the only method that catches lazy imports, optional extras and plugin deps.
# On 2026-08-12 the two new nodes were short 98 packages while passing both the
# version table and the import probe.
def pip_names():
    out = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=freeze'],
                         capture_output=True, text=True).stdout
    return ','.join(sorted({l.split('==')[0].strip().lower().replace('_', '-')
                            for l in out.splitlines() if '==' in l}))
rec('pip_names', pip_names)

print('|'.join(f'{k}={r[k]}' for k in sorted(r)))
PY
PROBE_EOF
)

probe_node() {  # $1 = ip 或 "local"
  if [ "$1" = "local" ]; then
    bash -c "$PROBE" 2>/dev/null | tail -1
  else
    timeout 150 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$1" \
      "bash -s" <<< "$PROBE" 2>/dev/null | tail -1
  fi
}

echo "=== 采集基准节点 ($REF_IP, 已验证可跑 rollout) ==="
REF=$(probe_node "$REF_IP")
if [ -z "$REF" ]; then echo "ERROR: 基准节点探针失败"; exit 2; fi
REFFILE=$(mktemp); echo "$REF" > "$REFFILE"
echo "$REF" | tr '|' '\n' | sed 's/^/  /'
echo

# 单节点模式
if [ $# -ge 1 ]; then TARGETS=("$@"); else TARGETS=("${ALL_IPS[@]}"); fi

FAIL=0
echo "=== 与基准对比（nvcc 的 PATH 差异忽略，训练不调用它）==="
printf "%-17s %s\n" "$REF_IP (基准)" "REFERENCE"
for ip in "${TARGETS[@]}"; do
  OUT=$(probe_node "$ip")
  if [ -z "$OUT" ]; then
    printf "%-17s %s\n" "$ip" "PROBE_FAILED (ssh 不通 / python 报错)"; FAIL=1; continue
  fi
  D=$(REF_FILE="$REFFILE" CUR="$OUT" python3 - <<'PY'
import os
ref = dict(kv.split('=', 1) for kv in open(os.environ['REF_FILE']).read().strip().split('|'))
cur = dict(kv.split('=', 1) for kv in os.environ['CUR'].strip().split('|'))
IGNORE = {'nvcc'}
diff = []
for k in sorted(set(ref) | set(cur)):
    if k in IGNORE:
        continue
    if k == 'pip_names':
        # Only MISSING packages matter; EXTRA ones are harmless. An exact match would
        # false-alarm forever because the head node carries dev tooling workers lack.
        # nvitop is an interactive GPU-monitoring TUI; training never imports it.
        BENIGN = {'nvitop'}
        gone = sorted((set(ref.get(k, '').split(',')) - set(cur.get(k, '').split(','))) - BENIGN)
        if gone:
            diff.append(f'pip_missing({len(gone)}):' + ','.join(gone[:12])
                        + (' ...' if len(gone) > 12 else ''))
        continue
    if ref.get(k) != cur.get(k):
        diff.append(f'{k}:{ref.get(k, "-")}->{cur.get(k, "MISSING")}')
print('IDENTICAL' if not diff else 'DIFF: ' + ' ; '.join(diff))
PY
)
  printf "%-17s %s\n" "$ip" "$D"
  [[ "$D" == IDENTICAL ]] || FAIL=1
done
rm -f "$REFFILE"

echo
if [ "$FAIL" -eq 0 ]; then
  echo "[PASS] 所有被检节点与 head 环境一致。"
else
  echo "[FAIL] 有节点不一致 —— 按 envs/README_cluster_env_setup_zh.md §3 修复后重跑。"
fi
exit "$FAIL"
