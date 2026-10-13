# 新服务器配置踩坑记录

本文记录 Harbor Terminus2 + slime 在新申请 Jizhi GPU 节点上拉起时遇到的服务器/运行环境问题。重点是避免把问题误判成 agent、模型或 RL 代码问题。

## 2026-05-22 32GPU 节点

任务：

```text
self_improving_terminal_agent_RL_32GPU_slime_v4
instance: 8B1D81589E49C0BC019E4B7BE2380265
```

### 1. 共享路径名不一致

远程节点上代码路径使用：

```text
/path/to/workspace/code/slime
```

不要在远程命令里默认使用本地看到的：

```text
/path/to/workspace/code/slime
```

当前这批节点两个路径都可能存在，但脚本和 remote exec 统一用 `/cq_1/...` 更稳。模型和大缓存不要放 `/cq_1`，Qwen3.5-9B 用：

```text
/path/to/models/Qwen3.5-9B
/path/to/models/Qwen3.5-9B_torch_dist
```

### 2. Ray IP 要选同一张网

launcher `hostname -I` 会返回多张网卡地址。这次能连通所有节点的是：

```bash
MASTER_ADDR=29.81.244.47
RAY_NODE_IP_PREFIX=29.81
```

如果误用 launcher 的 `29.97.*` 地址，worker 无法加入 head，表现为 launcher 一直等不到 `EXPECTED_NODES`。

启动成功时应看到：

```text
ray cluster has 4/4 nodes
Total resources: 32 GPU
```

### 3. `singularity-container` 不等价于成功环境

直接 apt 安装 Ubuntu universe 的 `singularity-container` 后版本是：

```text
singularity-ce version 4.1.1
```

这个版本能启动 Ray/SGLang，但 Harbor task SIF 容器启动会失败：

```text
WARNING: squashfuse mount failed, falling back to extraction:
exec: "fusermount": executable file not found in $PATH
FATAL: while preparing image: extraction failed
ERROR: Requesting capability set ... while permitted capability set is ...
```

原因是缺 FUSE 用户态命令，SIF 不能正常通过 squashfuse 挂载，fallback 到 sandbox extraction 后又受当前 pod capability 限制。

修复方式写入：

```text
examples/harbor_terminal_rl/install_apptainer_ubuntu.sh
```

当前脚本会优先安装 PPA 的 Apptainer：

```text
apptainer version 1.5.0
```

并显式安装：

```text
fuse
squashfuse
squashfs-tools
```

节点验证命令：

```bash
command -v singularity
singularity --version
command -v fusermount
command -v squashfuse
```

### 4. worker 上的 `/etc/ld.so.preload` 会污染 Singularity bind

worker-0 曾有：

```text
/etc/ld.so.preload: /$LIB/libtdsp_block.so
```

但对应源文件不存在。Singularity 会把它当作需要带进容器的 preload/bind，失败形式是：

```text
FATAL: container creation failed:
mount /$LIB/libtdsp_block.so->/$LIB/libtdsp_block.so error:
mount source /$LIB/libtdsp_block.so doesn't exist
```

`install_apptainer_ubuntu.sh` 已加入清理逻辑：发现这一行时会先备份 `/etc/ld.so.preload`，再删除缺失的 `/$LIB/libtdsp_block.so` 行。

### 5. Apptainer smoke test 必须从 `/tmp` 启动

在 contain/fakeroot 模式下，如果从共享代码目录直接执行：

```bash
cd /path/to/workspace/code/slime
singularity exec ...
```

会失败：

```text
container creation failed: can't change directory to /cq_1/.../slime:
permission denied
```

Harbor 自定义环境 `slime_plugins/harbor/singularity_env.py` 已经把 `create_subprocess_exec(..., cwd="/tmp")` 固定住。安装脚本里的 SIF smoke test 也需要 `cd /tmp` 后执行。

4 个节点当前通过的最小验证：

```bash
cd /tmp
singularity exec \
  --no-mount home \
  --no-mount tmp \
  --no-mount bind-paths \
  --no-mount proc \
  --writable-tmpfs \
  --fakeroot \
  --containall \
  --pid \
  /path/to/workspace/harbor_sif_cache_tb2_full_harbor_runtime/alexgshaw_regex-log_20251031.sif \
  echo harbor_exact_no_mount_ok
```

预期输出包含：

```text
harbor_exact_no_mount_ok
```

### 6. worker Ray 进程也必须带 `PYTHONPATH`

32GPU smoke 在 Apptainer 修复后又出现：

```text
RuntimeError: The actor with name SGLangEngine failed to import on the worker.
ModuleNotFoundError: No module named 'slime.backends.sglang_utils'
```

直接在 worker shell 中执行下面命令是成功的：

```bash
cd /path/to/workspace/code/slime
python -c 'import slime.backends.sglang_utils; print("ok")'
```

问题出在 `run_qwen35_9b_terminal_bench2_64h20.sh` 原来只在 launcher 分支后面 export `PYTHONPATH`。`RUN_ROLE=worker` 分支在此之前就 `ray start` 并 `exit 0`，导致 worker raylet 本身没有 slime/harbor 路径。

修复：把下面两行提前到 worker 分支之前：

```bash
export PYTHONPATH="${HARBOR_SRC}:${SLIME_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF
```

这个问题容易被误判成代码没有同步到 worker，但实际共享目录和直接 import 都是正常的。

### 7. 依赖安装会影响 SGLang 的 OpenAI 版本

为了让 Harbor import 通过，远程节点上跑过：

```bash
python -m pip install -e /path/to/workspace/code/harbor
```

它会把 `openai` 升到较新版本，而当前 SGLang 依赖需要：

```text
openai==2.6.1
```

所以安装 Harbor 后需要恢复：

```bash
python -m pip install openai==2.6.1
```

当前验证过：

```bash
python -c 'import harbor.trial.trial, tenacity, toml, shortuuid; import openai; print(openai.__version__)'
```

应输出：

```text
2.6.1
```

`litellm` 可能 warning 说需要更高版本 `openai`，但当前 rollout 走 `slime_sglang`，先以 SGLang 兼容性为准。

### 8. 32GPU smoke 验证状态

已验证通过：

- Ray 4 节点可连通，32 GPU placement group 可创建。
- `TrialWorker(num_cpus=0.05)` 可分散到 4 个节点。
- 4 个节点直接 `singularity exec` runtime SIF 可用。
- Harbor Python 依赖在 4 个节点可 import。
- `SGLangEngine` 在 worker 节点可 import，`PYTHONPATH` 修复有效。
- Harbor task container 可启动，FastAPI server 可 ready。
- slime 可从 Harbor/Terminus2 轨迹转换出 trainable samples。

已暴露并修复：

- 缺 `fusermount` 导致 SIF 挂载 fallback 到 extraction。
- worker-0 `/etc/ld.so.preload` 中缺失 `/$LIB/libtdsp_block.so`。
- Apptainer 从共享代码目录启动时 cwd permission denied。
- worker Ray 分支未提前设置 `PYTHONPATH`。

成功 smoke：

```text
Ray job: raysubmit_6BMTX5GpubPDgh7m
status: succeeded
```

输出文件：

```text
/path/to/workspace/slime_harbor_qwen35_9b_tb2_multinode_smoke/rollout_dumps/0.pt
/path/to/workspace/slime_harbor_qwen35_9b_tb2_multinode_smoke/debug_rollout_only_32gpu_multinode_smoke_py_path_fix/rollout/harbor_data_source_state_0.pt
```

4 条 trajectory：

```text
/path/to/workspace/slime_harbor_trials/tb2_full89_multinode_smoke_apptainer15/slime-r0-g0-s0-i0/agent/trajectory.json
/path/to/workspace/slime_harbor_trials/tb2_full89_multinode_smoke_apptainer15/slime-r0-g0-s1-i1/agent/trajectory.json
/path/to/workspace/slime_harbor_trials/tb2_full89_multinode_smoke_apptainer15/slime-r0-g0-s2-i2/agent/trajectory.json
/path/to/workspace/slime_harbor_trials/tb2_full89_multinode_smoke_apptainer15/slime-r0-g0-s3-i3/agent/trajectory.json
```

关键指标：

```text
harbor/num_groups: 1
harbor/num_samples: 4
harbor/num_converted_samples: 4
harbor/num_failed_trials: 0
harbor/num_missing_rollout_details: 0
harbor/num_incomplete_groups_dropped: 0
harbor/reward_mean: 0.0
rollout/response_len/mean: 172.5
perf/rollout_time: 71.99s
```

注意：这个 smoke 只使用 `max_turns=1`、`n_samples_per_prompt=4`、`debug-rollout-only`，证明多节点 rollout/env 路径可用；它不是 full 89 task / 2 epoch GRPO 训练验证。

### 9. 32GPU true train smoke 验证状态

debug rollout smoke 只能证明 Harbor/Terminus2 env 和 slime rollout 转换可用，不能证明 Megatron actor train、权重同步和 checkpoint 保存可用。因此又加了一个 non-debug train smoke：

```text
config: examples/harbor_terminal_rl/harbor_terminal_bench2_full89_multinode_train_smoke.yaml
script: examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_32gpu_train_smoke.sh
Ray job: raysubmit_iV8z4EuediU9iVjA
status: succeeded
```

这次资源切分是：

```text
total GPUs: 32
actor GPUs: 8
rollout GPUs: 24
actor nodes: 1
rollout nodes: 3
```

原因是 `train_async.py` 当前这条路径没有使用 colocate；non-debug 训练需要同时给 actor 和 rollout placement group 分配 GPU。32 卡节点组不能配置成 `ACTOR_NUM_GPUS=8` 加 `ROLLOUT_NUM_GPUS=32`，否则总需求会超过 32。

rollout placement group 只覆盖 3 个节点，所以这次 smoke 设置：

```text
harbor_trials_per_node: 2
harbor_num_concurrent_trials: 4
n_samples_per_prompt: 4
```

这样即使只有 3 个 rollout 节点，也有足够 TrialWorker 并发跑完 `n=4`。

### 10. `NUM_ROLLOUT=1` 可能只是 rollout-only 假阳性

第一次 non-debug smoke 使用了：

```text
NUM_ROLLOUT=1
Ray job: raysubmit_bdbaR6ChMAqihspY
status: succeeded
```

但它只产出了 rollout dump，没有进入 actor train。原因是从 fresh torch_dist checkpoint 启动时，Megatron 初始化得到的 `start_rollout_id` 是 1，而 `train_async.py` 的主循环使用近似下面的语义：

```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    ...
```

所以 `start_rollout_id=1` 且 `num_rollout=1` 时循环为空。这个 job 成功只能说明启动和初始 rollout 没坏，不能算训练验证。

修复是让 32GPU train smoke 默认：

```bash
NUM_ROLLOUT=2
```

这会在 fresh checkpoint 上执行 1 个真实训练 step，并在 iteration 1 保存 checkpoint。

### 11. true train smoke 的关键指标

成功的 true train smoke 输出：

```text
rollout dump:
/path/to/workspace/slime_harbor_qwen35_9b_tb2_train_smoke_step/rollout_dumps/1.pt

checkpoint:
/path/to/workspace/slime_harbor_qwen35_9b_tb2_train_smoke_step/train_smoke_32gpu_actor8_rollout24/iter_0000001/
```

checkpoint 目录确认有：

```text
common.pt
.metadata
metadata.json
__0_0.distcp ... __7_1.distcp
latest_checkpointed_iteration.txt
rollout/harbor_data_source_state_1.pt
```

rollout 指标：

```text
harbor/num_groups: 1
harbor/num_samples: 4
harbor/num_converted_samples: 4
harbor/num_failed_trials: 0
harbor/num_missing_rollout_details: 0
harbor/reward_mean: 0.0
rollout/response_len/mean: 165.0
rollout/response_len/max: 199
perf/rollout_time: 58.196s
```

训练指标：

```text
train/step: 1
train/loss: 0.0
train/pg_loss: 0.0
train/entropy_loss: 0.2057
train/grad_norm: 0.0
perf/update_weights_time: 16.3006s
perf/train_wait_time: 74.9649s
perf/actor_train_time: 499.9184s
perf/train_time: 500.0636s
perf/actor_train_tok_per_s: 11.5859
perf/step_time: 575.0286s
save_model elapsed: 25.3s
```

这次 `train/loss=0.0` 不是训练链路失败，而是 smoke 采样的 4 条样本 reward 全为 0，GRPO advantage 为 0，导致 pg loss/grad norm 都为 0。真正 full 89 task 训练要重点看 reward 分布和 pass/fail 任务覆盖。

### 12. true train smoke 暴露的新瓶颈

在 `max_turns=1`、`n=4` 的极小 smoke 下：

```text
rollout/env: ~58s
actor train: ~500s
full step: ~575s
```

因此这次端到端训练的主要瓶颈不是 Harbor env，而是 Megatron actor train。之前 full rollout 慢主要卡在 agent/env 执行；但在这个极小样本 true train smoke 中，actor 8 卡训练 4 个 microbatch 就用了约 500s。

当前需要后续继续查的方向：

```text
1. actor 并行策略是否过重：当前 TP=1, PP=8, SEQUENCE_PARALLEL=0, CP=1。
2. microbatch/global batch 是否导致 pipeline bubble 太大。
3. qwen3.5.py、train_actor.py、placement_group.py 中已有本地改动是否影响训练性能。
4. full 89 task 时 rollout 是否重新成为主瓶颈，需要用 wandb 同时对比 perf/rollout_time 和 perf/actor_train_time。
```

### 13. checkpoint 保存 warning

true train smoke 保存 checkpoint 时出现多条类似 warning：

```text
validation.py:397 - Rank X common state dict differs from rank 0 common state dict ...
optimizer param_state ... args
```

但日志随后明确显示：

```text
successfully saved checkpoint from iteration 1
```

并且 checkpoint 文件完整落盘。因此这个 warning 目前不是 blocker，但 full run 前后要重点验证：

```text
1. 能否从 iter_0000001 resume。
2. resume 后 start_rollout_id 是否正确。
3. 保存出的 HuggingFace/torch_dist 转换是否能正常加载推理。
```

### 14. Ray temp dir 不能太长

32GPU full89/max_turns=100 第一次启动时 Ray head 和 4 个节点都起来了，但 dashboard 起不来，后续 `ray job submit` 无法工作。launcher 日志里看到：

```text
Failed to start the dashboard
RuntimeError: Module MetricsHead failed to start. Received EOF from pipe.
```

真正原因在 `dashboard_MetricsHead.err`：

```text
OSError: validate_socket_filename failed:
AF_UNIX path length cannot exceed 107 bytes:
/tmp/slime-harbor-ray-full89-32gpu-max100/session_.../sockets/dash_MetricsHead
```

Ray dashboard 子进程会在 `$RAY_TEMP_DIR/session_*/sockets/` 下创建 Unix socket。`RAY_TEMP_DIR` 太长时，Linux AF_UNIX 107 字节限制会让 MetricsHead 失败，进而 JobHead/API 也不可用。

修复：full89 32GPU 脚本默认改成短路径：

```bash
RAY_TEMP_DIR=/tmp/rayf89m100
```

这个问题发生在 Ray job submit 之前，没有进入训练。重启后 dashboard/job submit 正常。

### 15. 32GPU full89 max_turns=100 启动记录

当前已启动的 full89/max_turns=100 训练：

```text
script: examples/harbor_terminal_rl/run_full89_grpo_32gpu_max100_n16.sh
config: examples/harbor_terminal_rl/harbor_terminal_bench2_full89_32gpu_max100_last_turn.yaml
Ray job: raysubmit_nXrjk3a6QcF4UBxB
status at launch check: RUNNING
```

资源切分：

```text
total GPUs: 32
actor GPUs: 8
rollout GPUs: 24
actor nodes: 1
rollout nodes: 3
```

训练/rollout 参数：

```text
max_turns: 100
n_samples_per_prompt: 16
rollout_batch_size: 4
global_batch_size: 64
num_rollout: 24
rollout_max_response_len: 8096
rollout_max_context_len: 16384
seq_length: 16384
sample_mode: last_turn
```

`NUM_ROLLOUT=24` 在 fresh torch_dist checkpoint 上会训练 rollout id 1..23。配合 `rollout_batch_size=4`，总共覆盖 92 个 task groups，也就是 89 个 terminal-bench-2 task 完整一遍再 rollover 3 个 task。

为了控制失败面和训练耗时，没有沿用 64GPU full run 的 `rollout_batch_size=16`。在 32GPU/3 rollout 节点下，`rollout_batch_size=4, n=16` 每轮是 64 条 trial；每个 rollout 节点 4 个 TrialWorker，总并发 12 条 trial。

本次保守开启的省时/稳定设置：

```text
enable_thinking: false
record_terminal_session: false
enable_summarize: false
last_turn sample mode
LOG_PROBS_CHUNK_SIZE=512
HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768
USE_CUSTOM_NO_ENTROPY_LOSS=1
harbor_require_full_rollout_groups=true
harbor_rollout_group_attempt_multiplier=8
```

W&B 当前是 offline：

```text
WANDB_MODE=offline
```

原因是远端节点环境里没有 `WANDB_API_KEY`。这避免把 token 写进共享脚本或命令行。offline run 落盘路径在：

```text
/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_max100_n16_1epoch/wandb/
```

启动后已确认：

```text
Ray cluster: 4/4 active nodes
Ray job: RUNNING
SGLang engines: health/model_info OK
TrialWorkers: 3 actors, one per rollout node, max_concurrency=4
Harbor env: cached SIF hit, FastAPI server ready
Terminus2: max_turns limited to 100
trajectory.json count at early check: 10
rollout_dumps at early check: 0
```

周期监控：

```text
tmux session: slime_full89_32gpu_max100_monitor
log: /tmp/slime_full89_32gpu_max100_monitor.log
interval: 1800s
launcher log: /tmp/slime_full89_32gpu_max100_launcher.log
```

## 2026-05-23 新 32GPU 节点重配

任务：

```text
self_improving_terminal_agent_RL_32GPU_slime_v4
instance: 8B1D81049E49C12E019E506B3616084F
```

### 1. 本批节点 IP 不在同一个二级网段

launcher:

```text
29.72.156.230 ... 29.81.242.251
```

worker-0:

```text
29.72.157.193 ... 29.81.226.140
```

worker-1:

```text
29.73.142.101 ... 29.79.8.135
```

worker-2:

```text
29.72.156.231 ... 29.81.242.196
```

和上一批不同，worker-1 没有 `29.81.*`，一开始尝试：

```bash
MASTER_ADDR=29.72.156.230
RAY_NODE_IP_PREFIX=29
```

结果只有 launcher 和 worker-2 加入 Ray，worker-0/1 连不上 `29.72.156.230:6379`。用临时 HTTP probe 反复测试后确认：

```text
worker-0/1 -> launcher: 只有 29.81.242.251 可达
launcher -> worker-0: 只有 29.81.226.140 可达
launcher -> worker-1: 只有 29.79.8.135 可达
launcher -> worker-2: 多个地址可达，选 29.81.242.196
```

最终 Ray 拓扑应使用：

```text
MASTER_ADDR=29.81.242.251
worker-0 WORKER_ADDR=29.81.226.140
worker-1 WORKER_ADDR=29.79.8.135
worker-2 WORKER_ADDR=29.81.242.196
```

### 2. Apptainer 需要重新安装

新节点最初没有：

```text
apptainer
singularity
fusermount
squashfuse
```

按脚本重装：

```bash
APT_PROXY=http://star-proxy.oa.com:3128 \
SIF_SMOKE_TEST=/path/to/workspace/harbor_sif_cache_tb2_full_harbor_runtime/alexgshaw_regex-log_20251031.sif \
bash examples/harbor_terminal_rl/install_apptainer_ubuntu.sh
```

四个节点最终均验证通过：

```text
apptainer version 1.5.0
/usr/bin/singularity
/usr/bin/fusermount
/usr/bin/squashfuse
harbor_exact_no_mount_ok
```

### 3. 旧的 slime editable 会污染 pip 和 import

新节点内置了旧的 editable slime metadata：

```text
/usr/local/lib/python3.12/dist-packages/__editable__.slime-0.2.4.pth
/usr/local/lib/python3.12/dist-packages/__editable___slime_0_2_4_finder.py
/usr/local/lib/python3.12/dist-packages/slime-0.2.4.dist-info
```

其中 finder 指向：

```python
{'slime': '/root/slime/slime', 'slime_plugins': '/root/slime/slime_plugins'}
```

这和实际共享代码路径 `/path/to/workspace/code/slime` 不一致，会导致两类问题：

```text
ModuleNotFoundError: No module named 'slime_plugins'
ModuleNotFoundError: No module named 'slime_plugins.harbor.import_utils'
```

pip 也会在扫描旧 editable 时异常：

```text
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/workspace/code/slime'
```

处理方式是把上述 metadata 备份到 `/tmp/slime_editable_backup_32gpu_*`，不动共享源码。训练脚本通过显式 `PYTHONPATH` 使用当前代码：

```bash
PYTHONPATH=/path/to/workspace/code/harbor/src:/path/to/workspace/code/slime:$PYTHONPATH
```

### 4. Harbor 依赖安装后必须恢复 OpenAI 版本

Harbor import 需要补齐：

```text
tenacity
toml
shortuuid
litellm
...
```

`litellm` 会拉高 `openai`，但当前 SGLang 需要：

```text
openai==2.6.1
```

因此安装依赖后必须再次执行：

```bash
python -m pip install --break-system-packages openai==2.6.1
```

最终四节点验证：

```text
import slime.backends.sglang_utils
import slime_plugins.harbor.rollout
import harbor.trial.trial
import harbor.agents.terminus_2.terminus_2
import tenacity, toml, shortuuid, openai, litellm
openai.__version__ == 2.6.1
```

LiteLLM 会尝试联网拉 model cost map，当前环境可能打印 network unreachable warning；这不是 blocking error。

### 5. 本次启动命令

worker 节点：

```bash
MASTER_ADDR=29.72.156.230 \
WORKER_ADDR=<per-node verified addr> \
RUN_ROLE=worker \
WORKER_KEEP_ALIVE=1 \
CLEAN_START=1 \
examples/harbor_terminal_rl/run_full89_grpo_32gpu_max100_n16_5epoch.sh
```

具体：

```text
worker-0: MASTER_ADDR=29.81.242.251 WORKER_ADDR=29.81.226.140 RAY_NODE_IP_PREFIX=29.81
worker-1: MASTER_ADDR=29.81.242.251 WORKER_ADDR=29.79.8.135 RAY_NODE_IP_PREFIX=29.79
worker-2: MASTER_ADDR=29.81.242.251 WORKER_ADDR=29.81.242.196 RAY_NODE_IP_PREFIX=29.81
```

launcher 节点：

```bash
MASTER_ADDR=29.81.242.251 \
RAY_NODE_IP_PREFIX=29.81 \
RUN_ROLE=launcher \
CLEAN_START=1 \
examples/harbor_terminal_rl/run_full89_grpo_32gpu_max100_n16_5epoch.sh
```

tmux/log：

```text
worker tmux: slime_full89_32gpu_max100_5epoch_worker
launcher tmux: slime_full89_32gpu_max100_5epoch_launcher
monitor tmux: slime_full89_32gpu_max100_5epoch_monitor
worker log: /tmp/slime_full89_32gpu_max100_5epoch_worker.log
launcher log: /tmp/slime_full89_32gpu_max100_5epoch_launcher.log
monitor log: /tmp/slime_full89_32gpu_max100_5epoch_monitor.log
```

训练配置：

```text
model: /path/to/models/Qwen3.5-9B
torch_dist: /path/to/models/Qwen3.5-9B_torch_dist
tasks: terminal-bench-2 full 89
max_turns: 100
n_samples_per_prompt: 16
rollout_batch_size: 4
global_batch_size: 64
num_rollout: 113
rollout_max_response_len: 2048
rollout_max_context_len: 131072
seq_length: 32768
qkv_format: thd
actor GPUs: 8
rollout GPUs: 24
```

启动后确认：

```text
Ray cluster: 4/4 active nodes, 32 GPU
Ray job: raysubmit_bMwSxNP5zy8CR8UR
W&B run: https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/3065fgs8
SGLang rollout engines: launched on 29.81.226.140, 29.81.242.196, 29.81.242.251
```

## 2026-05-23 32GPU high-concurrency restart

这次重启的目标是验证 rollout/env 段是否可以通过更高 Apptainer 并发提速。上一版
32GPU run 的 GPU 显存已占用，但大部分时间 SM util 为 0，日志显示 SGLang 是短 burst，
主要等待 Terminus/env 执行。上一版配置只有：

```text
harbor_num_concurrent_trials: 12
harbor_trials_per_node: 4
rollout_batch_size: 4
n_samples_per_prompt: 16
```

也就是 3 个 rollout 节点最多 12 条 active Apptainer-backed trial。用户说明此前单节点
32 个 Apptainer 已测试可行，因此本轮直接放大并发。

### stop 旧 run

32GPU 9B run:

```text
Ray job: raysubmit_bMwSxNP5zy8CR8UR
launcher tmux: slime_full89_32gpu_max100_5epoch_launcher
monitor tmux: slime_full89_32gpu_max100_5epoch_monitor
worker tmux: slime_full89_32gpu_max100_5epoch_worker
```

64GPU 27B run:

```text
Ray job: raysubmit_mg9q1rm4MK5GRZZL
launcher tmux: slime_full89_64gpu_27b_max100_launcher
monitor tmux: slime_full89_64gpu_27b_max100_monitor
worker tmux: slime_full89_64gpu_27b_max100_worker
```

两边均执行了 tmux kill、`ray stop --force`、训练/SGLang/TrialWorker/Apptainer runtime
进程 TERM。64GPU worker-1 曾出现一个 `ray::MegatronTr` disk-sleep 残留，这是之前
27B 大 checkpoint I/O wait 的同类表现。32GPU rollout 节点停止后留下若干
`fuse-overlayfs` zombie 形态 helper，已没有 active `starter`，不会继续跑 trial；
如果后续累计影响 mount 资源，需要直接重建节点或进一步清理 namespace。

### 新 high-concurrency 参数

新增文件：

```text
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_32gpu_max100_5epoch_high_concurrency_last_turn.yaml
examples/harbor_terminal_rl/run_full89_grpo_32gpu_max100_n16_5epoch_high_concurrency.sh
examples/harbor_terminal_rl/monitor_full89_32gpu_max100_5epoch_high_concurrency.sh
```

关键参数：

```text
harbor_trials_per_node: 64
harbor_num_concurrent_trials: 192
rollout_batch_size: 16
n_samples_per_prompt: 16
global_batch_size: 256
num_rollout: 29
```

这里没有直接把 `rollout_batch_size` 设成 64。原因是 `n=16` 时 `rollout_batch_size=64`
会让单轮等待 1024 条 trajectory 才进入训练，first update latency 和 rollout dump 体积都会很大。
本轮先用 `rollout_batch_size=16`，单轮 256 条 trajectory，能覆盖 3 个 rollout 节点的
192 并发位，同时保留较快的 first update 反馈。

输出目录：

```text
BASE_FOLDER=/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_max100_n16_5epoch_resp2048_ctx128k_rb16_conc192_pn64
TRIALS_DIR=/path/to/workspace/slime_harbor_trials/tb2_full89_32gpu_max100_5epoch_resp2048_ctx128k_rb16_conc192_pn64
```

启动拓扑沿用上一版可达 IP：

```text
launcher: MASTER_ADDR=29.81.242.251 RAY_NODE_IP_PREFIX=29.81
worker-0: MASTER_ADDR=29.81.242.251 WORKER_ADDR=29.81.226.140 RAY_NODE_IP_PREFIX=29.81
worker-1: MASTER_ADDR=29.81.242.251 WORKER_ADDR=29.79.8.135 RAY_NODE_IP_PREFIX=29.79
worker-2: MASTER_ADDR=29.81.242.251 WORKER_ADDR=29.81.242.196 RAY_NODE_IP_PREFIX=29.81
```

tmux/log：

```text
launcher tmux: slime_full89_32gpu_high_concurrency_launcher
worker tmux: slime_full89_32gpu_high_concurrency_worker
monitor tmux: slime_full89_32gpu_high_concurrency_monitor
launcher log: /tmp/slime_full89_32gpu_max100_5epoch_high_concurrency_launcher.log
worker log: /tmp/slime_full89_32gpu_max100_5epoch_high_concurrency_worker.log
monitor log: /tmp/slime_full89_32gpu_max100_5epoch_high_concurrency_monitor.log
```

启动后当前信息：

```text
Ray job: raysubmit_4UGuedaispGd24Dx
W&B run: https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/acr7vi57
Ray cluster: 4/4 active nodes
```

启动后 02:51 左右确认：

```text
TrialWorker actors: 3
per TrialWorker max_concurrency: 64
first update_weights: completed, elapsed about 14.5s
trajectory_json: 192
rollout_dumps: 0
```

02:58 左右 active Apptainer runtime 已达到目标并发：

```text
launcher rollout node: active_starter=62
worker-0 rollout node: active_starter=64
actor node: active_starter=0
worker-2 rollout node: active_starter=64
```

因此本轮瓶颈不再是代码层的 `harbor_trials_per_node=4` 限流；高并发已经推到约
190 active env。后续需要观察的是 256 条样本是否能顺利凑齐第一个 rollout dump，以及
更高 env 并发是否会带来 apt/pip/network/file-system 侧长尾。

03:12-03:13 复查：

```text
32GPU Ray job raysubmit_4UGuedaispGd24Dx: RUNNING
Ray cluster: 4/4 active nodes
trajectory_json: 256
rollout_dumps: 0
checkpoint_files: 0
```

这次复查说明新 32GPU 高并发 run 没有停止。它已经凑到第一轮配置上的
256 条 trajectory，但当时还没有看到 `rollout_dumps/*.pt` 或 checkpoint 落盘；
状态更像是刚到首轮 rollout 边界，正在做后处理/等待完整 group，而不是 Ray 失败。

同一时间 active Apptainer 和 GPU 利用率：

```text
node0 launcher/rollout: active Apptainer ~=55, TrialWorker ~=2, SGLang ~=9
node1 rollout:          active Apptainer ~=56, TrialWorker ~=2, SGLang ~=9
node2 actor:            active Apptainer ~=1,  TrialWorker ~=1, SGLang ~=1
node3 rollout:          active Apptainer ~=51, TrialWorker ~=2, SGLang ~=9
```

rollout 节点上多张 GPU 在 90% 左右，说明 SGLang/rollout 仍在吃算力。
actor 节点当时 GPU 利用率接近 0，符合“仍在等首轮 rollout dump/训练 batch”的阶段。

64GPU 旧 run 复查时没有看到旧 Ray job 继续运行，但 launcher 节点上还有大量
`squashfuse_ll` / `fuse-overlayfs` 辅助进程。这里要区分两件事：训练/Ray job 已停，
但 Apptainer/FUSE helper 可能残留。后续清理 64GPU 节点时应优先确认没有新的
`starter` / `TrialWorker` / `SGLangEngine` / `Megatron` 进程，再清理残留 helper。

## 2026-05-23 64GPU 27B high-concurrency restart

本轮给 64GPU/27B 单独新增高并发 wrapper/config，不覆盖旧 64GPU 脚本：

```text
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_64gpu_27b_max100_high_concurrency_last_turn.yaml
examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency.sh
examples/harbor_terminal_rl/monitor_full89_64gpu_27b_max100_high_concurrency.sh
```

关键参数：

```text
ACTOR_NUM_NODES=2
ROLLOUT_NUM_GPUS=48
harbor_trials_per_node: 64
harbor_num_concurrent_trials: 384
rollout_batch_size: 24
n_samples_per_prompt: 16
global_batch_size: 384
num_rollout: 20
```

这里把 `rollout_batch_size` 从 16 提到 24。原因是 64GPU/27B 当前是 6 个 rollout
节点，每个节点 64 个 Apptainer 并发位，`24 * n16 = 384` 正好能填满首轮 rollout
的全局并发上限。89 tasks * 5 epochs = 445 task groups，`NUM_ROLLOUT=20`
覆盖 19 个 rollout，即 456 groups。

输出目录：

```text
BASE_FOLDER=/path/to/workspace/slime_harbor_qwen35_27b_tb2_full89_64gpu_max100_n16_5epoch_resp2048_ctx128k_actorpp8_thd_rb24_conc384_pn64
TRIALS_DIR=/path/to/workspace/slime_harbor_trials/tb2_full89_64gpu_27b_max100_resp2048_ctx128k_actorpp8_thd_rb24_conc384_pn64
```

tmux/log：

```text
launcher tmux: slime_full89_64gpu_27b_high_concurrency_launcher
worker tmux: slime_full89_64gpu_27b_high_concurrency_worker
monitor tmux: slime_full89_64gpu_27b_high_concurrency_monitor
launcher log: /tmp/slime_full89_64gpu_27b_max100_high_concurrency_launcher.log
worker log: /tmp/slime_full89_64gpu_27b_max100_high_concurrency_worker.log
monitor log: /tmp/slime_full89_64gpu_27b_max100_high_concurrency_monitor.log
```

启动策略：

```text
MASTER_ADDR=10.0.0.4
RAY_NODE_IP_PREFIX=29.68
```

worker 先启动，launcher 后启动。为了避免重复复制 51G torch_dist：

```text
node1: STAGE_HF_CKPT_TO_TMP=0, STAGE_TORCH_DIST_CKPT_TO_TMP=1
node2-node7: STAGE_HF_CKPT_TO_TMP=1, STAGE_TORCH_DIST_CKPT_TO_TMP=0
launcher: defaults, HF 和 torch_dist staged marker 都存在
```

实际 Ray placement 里 actor 使用：

```text
10.0.0.8
10.0.0.22
```

这两个节点都有 `/tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60`
的 `.slime_stage_complete`，因此没有回到“从共享盘直接读 27B torch_dist 导致
Megatron D 状态/I/O wait”的老问题。

启动后信息：

```text
Ray job: raysubmit_Yx6tHC8Me24x78cm
W&B run: https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/zag9aboi
Ray cluster: 8/8 active nodes
```

03:43 左右 6 个 rollout TrialWorker 全部 spawn：

```text
Total TrialWorker actors spawned: 6
per TrialWorker max_concurrency: 64
```

03:45 左右 Megatron actor 从本机 staged torch_dist 成功加载并完成第一次权重广播：

```text
loading release distributed checkpoint from /tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60
successfully loaded checkpoint ...
Timer update_weights end (elapsed: 13.6s)
```

03:47 左右 Apptainer trials 已经开始铺开：

```text
rollout node active Apptainer ~= 30 / 39 / 37 / 27 / 34 / 34
```

03:48 复查时：

```text
trajectory_json=0
rollout_dumps=0
```

这不表示失败；当时刚完成 actor -> SGLang 的 `update_weights`，并发 env 正在启动，
还没有任务跑到 trajectory dump。下一步观察点是 active Apptainer 是否继续接近
384，以及第一批 trajectory 是否开始落盘。

## 2026-05-23 32GPU resp4096/os32 restart: Ray worker IP selection pitfall

触发背景：

- TaiJi 查杀告警第二次触发，32GPU 旧 run 最近 3 小时 GPU 平均利用率 29.15%，低于
  30% 阈值。
- 旧 run 使用 `--rollout-max-response-len 2048`，且没有启用 Harbor over-sampling。
- 处理动作：停止旧 Ray job `raysubmit_4UGuedaispGd24Dx`，清理 Ray/Apptainer 后重启新版本。

新 run 参数：

```text
Ray job: raysubmit_p1J3bUUphPvsYxpX
W&B run: https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/4vyw3xwk
rollout_max_response_len: 4096
over_sampling_batch_size: 32
rollout_batch_size: 16
n_samples_per_prompt: 16
global_batch_size: 256
harbor_num_concurrent_trials: 192
```

目录：

```text
BASE_FOLDER=/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_max100_n16_5epoch_resp4096_ctx128k_rb16_os32_conc192_pn64
SAVE_SUBDIR=Qwen3.5-9B_terminal_bench2_full89_32gpu_max100_n16_5epoch_resp4096_ctx128k_rb16_os32_conc192_pn64
TRIALS_DIR=/path/to/workspace/slime_harbor_trials/tb2_full89_32gpu_max100_5epoch_resp4096_ctx128k_rb16_os32_conc192_pn64
```

踩坑：

- launcher 可用 Ray head IP 仍是 `29.81.242.251`。
- worker-1 的 `hostname -I` 没有 `29.81.*` 地址，包含：

```text
29.73.142.101 ... 29.79.8.135 ...
```

- 首次启动 worker-1 时脚本默认选了第一个 `29.73.142.101`，Ray 里该节点随后变成
  `Alive=False`。
- 结果 Ray 只有 24 张 GPU 可用，placement group 卡在：

```text
{'CPU': 1.0, 'GPU': 1.0} * 32 (PACK): pending placement groups
```

修复：

```text
MASTER_ADDR=29.81.242.251
WORKER_ADDR=29.79.8.135
RUN_ROLE=worker
WORKER_KEEP_ALIVE=1
bash examples/harbor_terminal_rl/run_full89_grpo_32gpu_max100_n16_5epoch_high_concurrency.sh
```

修复后 Ray 状态：

```text
Ray cluster: 32 GPUs total
Pending Demands: no resource demands
Job status: RUNNING
```

启动后进度：

- Megatron actor 成功加载 `/path/to/shared_storage/.../Qwen3.5-9B_torch_dist`。
- `update_weights` 完成，随后 TrialWorker 开始执行 `r1` 的 Harbor rollout。
- 日志中已看到 cached SIF 被使用并开始启动 Apptainer container。

## 2026-05-23 64GPU 27B resp4096 restart after actor OOM

64GPU 27B 高并发 run `raysubmit_Yx6tHC8Me24x78cm` 崩溃点在 actor 训练，不在
Terminus env/rollout：

```text
ray::MegatronTrainRayActor.train()
slime_plugins/harbor/train_loss.py -> get_log_probs_and_entropy
Megatron fused_vocab_parallel_cross_entropy
torch.OutOfMemoryError: Tried to allocate 244.00 MiB
GPU free: 231.06 MiB
```

当时参数是：

```text
rollout_max_response_len=2048
seq_length=32768
harbor_max_train_total_tokens=32768
log_probs_chunk_size=256
```

处理策略：

- 用户确认需要调高的是每一步生成上限，即 `rollout_max_response_len`，不是训练侧
  `HARBOR_MAX_TRAIN_TOTAL_TOKENS`。
- 新 run 改为 `rollout_max_response_len=4096`，减少单步回复被截断。
- 训练侧仍保留 `seq_length=32768` / `HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768`，避免
  16k train cap 严重裁掉 prompt。
- 为控制 27B actor logprob/cross-entropy 峰值，把 `LOG_PROBS_CHUNK_SIZE` 从
  256 降到 32。
- 同时启用 Harbor over-sampling：`OVER_SAMPLING_BATCH_SIZE=48`，最终训练 batch
  仍是 `rollout_batch_size=24`。

新目录后缀：

```text
resp4096_ctx128k_actorpp8_thd_rb24_os48_conc384_pn64_traincap32k_lpc32
```

启动踩坑：

- 第一版 `RAY_TEMP_DIR=/tmp/ray27bf89pp8rb24os48c384r4096tc32klpc32` 太长，
  Ray head 在创建 plasma store socket 时失败：

```text
AF_UNIX path length cannot exceed 107 bytes
.../session_.../sockets/plasma_store
```

- 已改短为：

```text
RAY_TEMP_DIR=/tmp/ray27b4096
```

更新文件：

```text
examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency.sh
examples/harbor_terminal_rl/monitor_full89_64gpu_27b_max100_high_concurrency.sh
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_64gpu_27b_max100_high_concurrency_last_turn.yaml
examples/harbor_terminal_rl/HARBOR_ROLLOUT_LONG_TAIL_ASYNC_NOTES.md
```

重启前检查：

- old Ray job 已 failed，没有保存 checkpoint；只看到旧目录里 `rollout_dumps/1.pt`。
- launcher 无 GPU 残留。
- worker-0 和 worker-1 都有
  `/tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60/.slime_stage_complete`。
  这两个节点优先启动 worker tmux，降低 actor 被放到未 staging torch_dist 节点上的风险。

本轮启动策略：

```text
MASTER_ADDR=10.0.0.4
RAY_NODE_IP_PREFIX=29.68
worker-0: STAGE_HF_CKPT_TO_TMP=0, STAGE_TORCH_DIST_CKPT_TO_TMP=1
worker-1: STAGE_HF_CKPT_TO_TMP=1, STAGE_TORCH_DIST_CKPT_TO_TMP=1
worker-2..6: STAGE_HF_CKPT_TO_TMP=1, STAGE_TORCH_DIST_CKPT_TO_TMP=0
```

tmux/log 沿用：

```text
launcher tmux: slime_full89_64gpu_27b_high_concurrency_launcher
worker tmux: slime_full89_64gpu_27b_high_concurrency_worker
monitor tmux: slime_full89_64gpu_27b_high_concurrency_monitor
launcher log: /tmp/slime_full89_64gpu_27b_max100_high_concurrency_launcher.log
worker log: /tmp/slime_full89_64gpu_27b_max100_high_concurrency_worker.log
monitor log: /tmp/slime_full89_64gpu_27b_max100_high_concurrency_monitor.log
```

## 2026-05-23 64GPU 27B resp4096 second actor OOM and TP2 restart

`resp4096_ctx128k_actorpp8_thd_rb24_os48_conc384_pn64_traincap32k_lpc32`
这轮已经能正常完成 SGLang engine 启动、TrialWorker rollout、`update_weights` 和
第一轮 rollout dump，但在 actor train 早期再次 OOM：

```text
actor train micro-batch: 20/192
torch.OutOfMemoryError: Tried to allocate 32.00 MiB
GPU 7 total: 95 GiB
GPU 7 free: 11.06 MiB
PyTorch allocated: 90.60 GiB
```

这次已经把 `LOG_PROBS_CHUNK_SIZE` 从 256 降到 32，所以继续单纯调小 chunk 的收益很小。
关键判断是：保持 `TP=1, PP=8` 时，即使把 `ACTOR_NUM_NODES` 从 2 增大，也主要增加
data-parallel actor 副本，不会降低每张 GPU 上的 vocab-parallel logprob / cross-entropy
峰值。真正能分摊这条 OOM 路径的是 tensor parallel。

新的 64 卡重启配置：

```text
ACTOR_NUM_NODES=4
ACTOR_NUM_GPUS_PER_NODE=8
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=8
ROLLOUT_NUM_GPUS=32
ROLLOUT_BATCH_SIZE=16
OVER_SAMPLING_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=256
NUM_ROLLOUT=29
ROLLOUT_MAX_RESPONSE_LEN=4096
SEQ_LENGTH=32768
HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768
LOG_PROBS_CHUNK_SIZE=32
```

新目录后缀：

```text
resp4096_ctx128k_actor4_tp2pp8_thd_rb16_os32_conc256_pn64_traincap32k_lpc32
```

这个配置牺牲两个 rollout 节点给 actor，rollout 侧从 6 节点 384 并发降到 4 节点
256 并发，但目标是先把 27B actor train 的单卡显存压下来，避免反复在训练阶段崩溃。

仍需注意：

- 当前离线转换目录名仍是 `Qwen3.5-27B_torch_dist_pp8_timeout60`，原始转换 layout 是
  `TP=1, PP=8`。Megatron dist checkpoint 理论上可以按新的训练并行 layout 加载/reshard，
  但如果 TP2/PP8 启动时报 checkpoint shard 不匹配，下一步需要单独转换或保存一个
  `TP=2, PP=8` 的 torch_dist checkpoint。
- rollout 节点减少后，GPU 利用率可能不如 6 rollout 节点版本高；这轮优先解决 27B
  actor OOM，再根据 W&B 的 `actor_train` 时间和 rollout idle 时间决定是否继续调大
  `OVER_SAMPLING_BATCH_SIZE` 或 TrialWorker 并发。

实际重启记录：

- Ray job: `raysubmit_Yz3pEKbpYQJEvB13`
- W&B run: `mwtugsqw`
- 由于 node7 之前没有本地 `/tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60`
  marker，首次启动先 rsync 了 51G torch_dist，导致 Ray job 一度 pending 在
  `{'GPU': 1.0, 'CPU': 1.0} * 64 (PACK)` placement group。marker 写完后 node7
  加入 Ray，集群变成 8 节点 / 64 GPU，pending demand 清空。
- TP2/PP8 从已有 TP1/PP8 checkpoint 加载成功。日志里有预期提示：

```text
(TP, PP) mismatch after resume ((2, 8) vs (1, 8) from checkpoint): RNG state will be ignored
successfully loaded checkpoint from /tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60
```

- 第一次 `update_weights` 成功，耗时约 12.8s；actor 更新前后单卡显存大约从
  `free_GB=82.31` 变成 `free_GB=80.73`，没有再在启动阶段 OOM。
- 4 个 rollout 节点启动了 16 个 SGLang engine 和 4 个 TrialWorker：

```text
rollout nodes: 10.0.0.23, 10.0.0.14, 10.0.0.9, 10.0.0.16
TrialWorker max_concurrency: 64 each
harbor_num_concurrent_trials: 256
```

- 截至 2026-05-23 14:23 左右，`r1` 的 Harbor rollout 已开始启动 Apptainer trials：

```text
Timer update_weights end (elapsed: 12.8s)
[TrialWorker ...] running r1 g0 s0
Starting Slime Singularity container ... (attempt 1/3)
```

## 2026-05-23 replacement 32GPU 9B high-concurrency retry

旧 32GPU 任务被集群回收后，新实例：

```text
self_improving_terminal_agent_RL_32GPU_slime_v4
instance: 8B1D80049E49C044019E5530D1C20CB1
```

这轮继续用 9B / 89 tasks / 5 epochs / `max_turns=100` / `rollout_max_response_len=4096`，
但把 env 并发从上一轮的 3 rollout nodes * 64 = 192 提高到：

```text
harbor_trials_per_node: 128
harbor_num_concurrent_trials: 384
```

保持训练 batch 形状不变，避免把并发压测和训练侧显存变量混在一起：

```text
ROLLOUT_BATCH_SIZE=16
OVER_SAMPLING_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=16
GLOBAL_BATCH_SIZE=256
NUM_ROLLOUT=29
```

新目录后缀：

```text
resp4096_ctx128k_rb16_os32_conc384_pn128
```

更新文件：

```text
examples/harbor_terminal_rl/run_full89_grpo_32gpu_max100_n16_5epoch_high_concurrency.sh
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_32gpu_max100_5epoch_high_concurrency_last_turn.yaml
examples/harbor_terminal_rl/monitor_full89_32gpu_max100_5epoch_high_concurrency.sh
examples/harbor_terminal_rl/setup_32gpu_node_runtime.sh
```

风险点：

- 384 env 并发会显著增加 Apptainer 启动、端口、CPU 和 `/tmp` 压力；如果出现端口占用、
  server startup timeout 或 GPU 利用率反而下降，先降到 `harbor_trials_per_node=96`
  / `harbor_num_concurrent_trials=288`。
- `OVER_SAMPLING_BATCH_SIZE=32` 等价于最多 512 sample attempts；384 并发不能一次性全
  部在飞，但已经比上一轮 192 更接近填满 over-sampling window。

新 32GPU 节点安装 runtime 时再次遇到 pip legacy editable 残留问题：

```text
FileNotFoundError: [Errno 2] No such file or directory:
'/path/to/workspace/code/slime'
```

根因不是依赖下载失败，而是节点的 `/usr/local/lib/python3.12/dist-packages` 里还留着旧
`slime` editable metadata，pip 在安装任何包之前会扫描 `.egg-link` / `easy-install.pth`
里的 legacy editable path；如果里面记录的 `/cq_1/.../slime` 当时不可见或已失效，pip
会直接崩。`setup_32gpu_node_runtime.sh` 已补充清理：

```text
__editable__.slime-0.2.4.pth
__editable___slime_0_2_4_finder.py
slime-0.2.4.dist-info
slime.egg-link / *slime*.egg-link
easy-install.pth 中指向旧 slime workspace 的行
```

这个清理只移动/备份旧 metadata，不删除源码；运行时仍然通过 `PYTHONPATH` 指向当前
`slime` 和 `harbor/src`。

补充：如果在 `slime` repo 目录下直接运行 `python3 -m pip install`，pip 自己的
`sys.path` 也可能把当前 `/cq_1/.../slime` 放进 metadata scan path，继续触发同类
`FileNotFoundError`。`install_harbor_runtime_deps.sh` 已改成在 `/tmp` 子 shell 中、
清空 `PYTHONPATH` 后执行 pip；依赖安装完成后再由启动脚本注入源码 `PYTHONPATH`。

再次补充：`SlimeSingularityEnvironment` 是由 Harbor task config 的
`environment.import_path` 动态加载的，文件顶层会 import Harbor 的
`SingularityEnvironment`。源码模式下 Harbor 没有 package metadata，必须先调用
`ensure_harbor_distribution_metadata()`，否则会在真正创建 task environment 时触发
`PackageNotFoundError: No package metadata was found for harbor`。已在
`slime_plugins/harbor/singularity_env.py` 顶层补 metadata fallback，setup 验证也改成先
注入 fallback 再 import Harbor 组件。

## 2026-05-23 replacement 32GPU launch log

新 32GPU 实例：

```text
self_improving_terminal_agent_RL_32GPU_slime_v4
instance: 8B1D80049E49C044019E5530D1C20CB1
```

本次启动参数：

```text
MASTER_ADDR=29.81.244.30
node1 WORKER_ADDR=29.81.242.251
node2 WORKER_ADDR=29.81.242.55
node3 WORKER_ADDR=29.81.224.7

harbor_trials_per_node: 128
harbor_num_concurrent_trials: 384
rollout_batch_size: 16
over_sampling_batch_size: 32
n_samples_per_prompt: 16
rollout_max_response_len: 4096
rollout_max_context_len: 131072
max_turns: 100
```

启动前在 4 个节点完成：

```text
apptainer version 1.5.0
openai 2.6.1
slime / Harbor / SlimeSingularityEnvironment import smoke OK
```

23:35 左右启动：

```text
node1/2/3 tmux: slime_full89_32gpu_high_concurrency_worker
node0 tmux: slime_full89_32gpu_high_concurrency_launcher
node0 tmux: slime_full89_32gpu_high_concurrency_monitor
```

23:38 检查：

```text
Ray job: raysubmit_c7dqe8e6YG1JD5Hz
W&B run: c5qdwmk9
Ray nodes: 4 active / 32 GPUs
SGLang engines: ready/starting across 24 rollout GPUs
TrialWorkers: 3 actors, max_concurrency=128 each
Actual layout: node3 actor/train; node0/node1/node2 rollout/env
```

这次 launcher 节点本身没有做 actor，actor 被 Ray placement group 排到了
`29.81.224.7`；因此三个 TrialWorker 分布在剩余三个节点，符合“1 train node + 3 rollout
nodes”的目标。

## 2026-05-30 32GPU 9B actor2/rollout2 restart blocker

目标配置：

```text
task: self_improving_terminal_agent_RL_32GPU_slime_v4
instance: 8B1D81049E49C12E019E5B4BDF170FCA
model: /path/to/models/Qwen3.5-9B
topology: 2 actor/train nodes + 2 rollout nodes
rollout/train context: 131072
rollout_max_response_len: 4096
agent override_timeout_sec: 3600
progress reward: slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards
```

已新增 wrapper：

```text
examples/harbor_terminal_rl/run_full89_grpo_32gpu_9b_max100_actor2_rollout2_128k_timeout3600_turn_grouped.sh
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_32gpu_9b_max100_actor2_rollout2_128k_timeout3600_turn_grouped.yaml
```

本次没有直接拉起训练的原因不是配置问题，而是 node1/worker-0 控制面异常：

```text
node0 launcher: jizhi exec OK
node2 worker-1: jizhi exec OK
node3 worker-2: jizhi exec OK
node1 worker-0: Taiji 能列出 pod，logs 能读取，但 jizhi exec 选 1 后不执行命令并卡到 timeout
```

复现命令：

```bash
timeout 60 expect /path/to/workspace/code/scripts/exec_multinode.exp \
  self_improving_terminal_agent_RL_32GPU_slime_v4 \
  8B1D81049E49C12E019E5B4BDF170FCA \
  1 \
  "date +%F_%T; hostname -I; echo NODE1_EXEC_OK"
```

结果只停在：

```text
choose one host to execute command (0-3): 1
```

直到本地 `timeout` 杀掉命令，没有输出 `NODE1_EXEC_OK`。因此如果自动启动
`EXPECTED_NODES=4`，launcher 会一直等不到第 4 个 Ray node，不应该强行启动。

可行的手动恢复路径：

1. 在 Taiji 页面打开 worker-0 的 Web terminal，手动执行 worker 启动命令。
2. 或者重新申请/重启 32GPU 实例，让 4 个 pod 的 `jizhi_client exec` 都能正常执行。
3. 如果只想临时验证代码，可以改成 3 节点拓扑，但这已经不是“2 train + 2 rollout / 32GPU”。

后续人工验证发现：从 node0 的物理/高速网 shell 可以直接 SSH 到 worker，例如：

```bash
ssh 28.133.34.78
ssh 28.133.32.83
```

这条链路和 Codex 使用的 `jizhi_client exec ... bash -c ...` 不是同一个控制上下文。
Codex 通过 `jizhi_client exec` 进入 node0 后再 `ssh 28.133.32.83` 会提示密码或卡住，
无法复用人工 shell 中的 SSH 免密状态。因此如果 node1 的 `jizhi_client exec` 持续不可用，
最稳的办法是在 node0 的人工 shell 或各节点 Web terminal 中手动启动 4 个 tmux 会话。

推荐手动启动时统一使用 28.133 网段：

```bash
MASTER_ADDR=28.133.34.78
RAY_NODE_IP_PREFIX=28.133
SOCKET_IFNAME=bond1
```

这样 Ray、Gloo、NCCL 和 worker0 SSH 都在同一高速网段上，避免 28.* / 29.* 混用。

## 2026-05-31 32GPU 9B 64k restart: node3 Ray agent 端口问题

新实例 `8B1D801D9E6E1752019E7980B85A3387` 上，4 个节点都能通过
`jizhi_client exec` 访问，但 node3 的环境安装最慢。安装 apptainer 时先卡在 apt 下载，完成后
setup smoke test 仍会报：

```text
ModuleNotFoundError: No module named 'slime_plugins.harbor'
```

这个 smoke test 报错是 setup 脚本没有带正式训练的 `PYTHONPATH`，不是正式启动 blocker。
正式脚本会设置：

```bash
PYTHONPATH="${HARBOR_SRC}:${SLIME_DIR}:${PYTHONPATH:-}"
```

真正的 node3 blocker 是 Ray worker 第一次加入集群时 dashboard agent 报端口占用：

```text
OSError: [Errno 98] Address already in use
agent_manager.cc:87: The raylet exited immediately because one Ray agent failed, agent_name = dashboard_agent
```

随后 `ray start` 还出现：

```text
GCS cannot find the node with node ID ...
Try increase the RAY_raylet_start_wait_time_s config.
```

最终可用的处理方式：

1. 先让 node3 完成 apptainer 安装，确认 `/usr/bin/apptainer` 和 `/usr/bin/singularity` 都存在。
2. node3 不再用默认清理段启动 worker，显式设置 `CLEAN_START=0`。
3. 给 node3 单独换一组 Ray 本地端口，避开 `18001-18006` 的短暂冲突。
4. 增加 `RAY_raylet_start_wait_time_s=120`，给 GCS 注册留足时间。

本次成功加入集群的 node3 worker 启动核心参数：

```bash
CLEAN_START=0 \
RAY_raylet_start_wait_time_s=120 \
RAY_NODE_MANAGER_PORT=18101 \
RAY_OBJECT_MANAGER_PORT=18102 \
RAY_RUNTIME_ENV_AGENT_PORT=18103 \
RAY_DASHBOARD_AGENT_LISTEN_PORT=18104 \
RAY_DASHBOARD_AGENT_GRPC_PORT=18105 \
RAY_METRICS_EXPORT_PORT=18106 \
RAY_MIN_WORKER_PORT=21000 \
RAY_MAX_WORKER_PORT=21999 \
RUN_ROLE=worker \
WORKER_KEEP_ALIVE=1 \
MASTER_ADDR=29.81.242.16 \
RAY_NODE_IP_PREFIX=29.81 \
SOCKET_IFNAME=bond1 \
bash examples/harbor_terminal_rl/run_full89_grpo_32gpu_9b_max100_actor2_rollout2_64k_timeout3600_turn_grouped_dynbatch.sh
```

成功后 head 侧 `ray status` 显示：

```text
Active: 4 nodes
Total: 32 GPU
```
