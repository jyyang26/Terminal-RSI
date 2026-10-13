# Harbor Terminus 2 and slime Integration Summary

## 2026-05-22 Update: 27B/64GPU 与 32GPU 5-epoch 重启

本节记录本轮按用户要求做的两件事：停掉 32 卡旧 run 后用更新代码重启 9B/32GPU 监控版；同时准备在 64 张 H20 上跑 Qwen3.5-27B、full89、`n=16`、`max_step=100`、约 5 epoch。

### 参数选择

- `rollout_max_response_len=2048`：按本轮要求降低单步生成上限，避免 8096 response 继续放大 env 和训练侧耗时。
- `max_step=100`：写在 Harbor terminal-bench config 的 `agent.max_steps` 里。
- `rollout_max_context_len=131072`、`seq_length=32768`：32k 和 64k 都被真实 `max_step=100` terminal history 打穿；已观察到 prompt 63639 tokens，加 2048 response 后超过 64k。当前使用 128k 作为 rollout/SGLang 生成上下文，actor 训练侧的 `seq_length` 仍保持 32k，并通过 sample postprocess 把超长 train sample cap 到 32k。
- SGLang server 也必须显式同步上下文参数，否则只改 slime 的 `rollout_max_context_len` 不够。本轮已在通用脚本里传入 `--sglang-context-length 131072`、`--sglang-max-prefill-tokens 131072`、`--sglang-chunked-prefill-size 8192`。
- 27B actor 初始按 2 个节点更新：`actor_num_nodes=2`、16 GPU，`TP=4`、`PP=2`、`DP=2`。剩余 48 GPU 给 rollout，`rollout_num_gpus_per_engine=2`，即 24 个 rollout engines。

### 新增/修改文件

- `examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh`
  - 新增 `DECODER_LAST_PIPELINE_NUM_LAYERS` 透传，用于 Qwen3.5-27B 的 PP 不均匀层切分。
  - 新增 SGLang context/prefill/chunked-prefill 参数透传。
- `examples/harbor_terminal_rl/convert_qwen35_27b_torch_dist.sh`
  - Qwen3.5-27B HF -> Megatron torch_dist 离线转换脚本。
  - 输入：`/path/to/models/Qwen3.5-27B`。
  - 当前输出：`/path/to/models/Qwen3.5-27B_torch_dist_pp8_timeout60`。
  - 使用 `TP=1`、`PP=8`、`distributed_timeout_minutes=60`。这是仓库已有 27B 转换脚本采用的 PP-only 方案，用来规避 TP conversion 在 Qwen3.5 GatedDeltaNet/mbridge scatter 上卡住的问题。
- `examples/harbor_terminal_rl/harbor_terminal_bench2_full89_64gpu_27b_max100_last_turn.yaml`
  - full89 terminal-bench-2，`max_steps=100`，`harbor_sample_mode=last_turn`，6 个 rollout trial nodes，24 个 concurrent trials。
- `examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch.sh`
  - 64GPU/27B 训练入口，约 5 epoch。
  - `rollout_batch_size=16`、`n_samples_per_prompt=16`、`global_batch_size=256`、`num_rollout=29`。
  - rollout/SGLang context 128k，actor train `seq_length` 32k。
  - actor 训练仍使用 `TP=4`、`PP=2`、`decoder_last_pipeline_num_layers=30`，从 PP8 torch_dist checkpoint 加载时由 Megatron dist checkpoint 做重分片。
  - `qkv_format=bshd`，和仓库已有 Qwen3.5-27B 脚本一致；27B 的 GatedDeltaNet 不走 9B 当前使用的 `thd` 路径。
  - `save_interval=10`，避免过频繁保存。
  - W&B 已写入脚本环境变量，但本文档不记录 token。
- `examples/harbor_terminal_rl/harbor_terminal_bench2_full89_32gpu_max100_5epoch_last_turn.yaml`
  - 32GPU/9B 5 epoch 配置，`max_steps=100`。
- `examples/harbor_terminal_rl/run_full89_grpo_32gpu_max100_n16_5epoch.sh`
  - 32GPU/9B 训练入口，1 个 actor 节点、24 rollout GPU。
  - `rollout_batch_size=4`、`n_samples_per_prompt=16`、`global_batch_size=64`、`num_rollout=113`，fresh start 下约覆盖 448 task groups。
  - rollout/SGLang context 128k，actor train `seq_length` 32k。
- `examples/harbor_terminal_rl/monitor_full89_64gpu_27b_max100.sh`
- `examples/harbor_terminal_rl/monitor_full89_32gpu_max100_5epoch.sh`
  - 监控脚本只解析 launcher log 中的 Ray job id，然后用 `ray job status`，避免 `ray job list` 把 runtime env 和 W&B key 打进监控日志。

### 32GPU 当前状态

旧 32GPU run 已停掉。第一版 32k context run 和第二版 64k context run 都已停止，当前已使用 128k rollout/SGLang context 重启：

```text
stopped 32k Ray job: raysubmit_FcjbjHySEQ55d2Pe
stopped/being replaced 64k Ray job: raysubmit_p37R34G9HbekfGyi
current 128k Ray job: raysubmit_YxZ3FR45G5akguiy
current W&B run: https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/28addspp
launcher tmux: slime_full89_32gpu_max100_5epoch_launcher
monitor tmux: slime_full89_32gpu_max100_5epoch_monitor
launcher log: /tmp/slime_full89_32gpu_max100_5epoch_launcher.log
monitor log: /tmp/slime_full89_32gpu_max100_5epoch_monitor.log
```

已确认：

- Ray job 状态为 `RUNNING`。
- 第一版 32GPU 重启使用 32k rollout context，后来发现一个真实 max_step=100 trial 的 prompt 已到 31365 tokens，加 2048 response 后总长 33413，SGLang 以超过 32768 context 拒绝生成。该 run 已停止，停止前已有 28 条 `trajectory.json`，没有产生 rollout dump。
- 第二版 64k rollout context 又遇到 prompt 63639 tokens，加 2048 response 后总长 65687，超过 65536。说明 `max_step=100` 下只提高到 64k 仍不够。
- 当前脚本已切到 128k rollout/SGLang context，并使用新的 `ctx128k` 输出目录，避免和前两次失败数据混在一起。
- 2026-05-22 14:37 左右，128k run 仍为 `RUNNING`；已看到 SGLang `context_length=131072`、`max_prefill_tokens=131072` 生效，已有 12 条 `trajectory.json`，还没有第一个 rollout dump。当前还在首批 4 个 task group 的 env/agent 执行阶段。
- 2026-05-22 16:42 左右，32GPU/9B run 继续为 `RUNNING`；已有 92 条 `trajectory.json` 和 1 个 rollout dump。日志里已出现 `step 1`：

```text
step 1: train/loss=0.042724818479075465, train/grad_norm=2.0887998451481224, lr=1e-06
Timer train end (elapsed: 2153.7s)
```

这说明 32GPU 监控版已经完成至少一个 rollout -> train 的闭环。`ray status` 里当前只有 8/32 GPU 显示为 actively used，是因为 actor train step/rollout 阶段的 Ray resource accounting 不等同于所有 SGLang/TrialWorker 的瞬时 GPU 利用；判断是否健康仍以 trajectory 增长、rollout dump、train step、错误日志为准。

### 27B/64GPU 当前状态

27B 训练必须先有 torch_dist checkpoint。直接使用 `/path/to/models/Qwen3.5-27B` 的 HF 权重可用于 SGLang rollout，但 Megatron actor 仍需要能加载的 checkpoint。

第一次离线转换失败点不是 OOM，而是 NCCL collective timeout：

```text
Watchdog caught collective operation timeout: OpType=SCATTER Timeout(ms)=600000
```

当时部分 rank 还在从共享文件系统加载/映射 HF safetensors，部分 rank 已经进入 tensor-parallel scatter，同步等待超过默认 10 分钟。判断是共享存储加载进度 skew 触发超时，而不是显存不足。

本轮处理：

- 最初尝试把转换输出切到 `Qwen3.5-27B_torch_dist_tp4pp2_timeout60`，希望转换产物直接匹配训练的 `TP=4,PP=2`。
- 把 Megatron distributed timeout 提到 60 分钟。
- TP4/PP2 retry 进入 `torchrun` 并完成部分 rank 的 HF model load，但 15 分钟左右日志无新进展，GPU 呈现部分 rank 长时间 100% 利用率、部分 rank 等待，形态接近 TP scatter/weight mapping 卡住。
- 已切换到仓库已有 27B 转换脚本同类的 PP-only 方案：`TP=1,PP=8`，输出目录 `Qwen3.5-27B_torch_dist_pp8_timeout60`。训练时仍按 `TP=4,PP=2` 启动，由 Megatron dist checkpoint 加载阶段重分片。
- PP-only 离线转换已完成，输出目录约 51G，并写入 `latest_checkpointed_iteration.txt`。训练脚本现在使用：

```text
PRETRAINED_CKPT=/path/to/models/Qwen3.5-27B_torch_dist_pp8_timeout60
HF_CKPT=/path/to/models/Qwen3.5-27B
```

转换完成后再启动 64GPU/27B 训练：

```text
worker session: slime_full89_64gpu_27b_max100_worker
launcher session: slime_full89_64gpu_27b_max100_launcher
monitor session: slime_full89_64gpu_27b_max100_monitor
launcher log: /tmp/slime_full89_64gpu_27b_max100_launcher.log
monitor log: /tmp/slime_full89_64gpu_27b_max100_monitor.log
```

2026-05-22 14:37 左右，64GPU/27B 尚未提交 Ray job，当前状态是所有节点在把 52G HF 权重 staging 到本机 `/tmp/slime-harbor-model-cache/Qwen3.5-27B`。观察到 launcher 节点约 21G，worker-0 约 9G，`rsync` 仍在运行；这是启动路径里的预期步骤，不是训练失败。由于 `STAGE_MARKER=/tmp/slime-harbor-model-cache/Qwen3.5-27B/.slime_stage_complete`，首次 staging 完成后后续重启会跳过全量同步，除非显式设置 `STAGE_HF_CKPT_FORCE=1`。

2026-05-22 14:55 左右复查，launcher 本机缓存已到约 52G，但 `rsync` 仍在收尾；worker 节点之间差异较大，抽查到 worker-0 约 23G、worker-1 约 27G、worker-3/worker-6 约 16G。这个阶段的瓶颈是 8 节点同时从 gy2 读 27B HF 权重，不是 GPU/训练侧。Ray head 和 Ray job 仍未启动。

2026-05-22 15:07 左右做了一次运行时优化：worker-0 是第二个 actor 节点，按当前 `actor_num_nodes=2`、`rollout_num_gpus=48` 的布局，它不承载 SGLang rollout engine，因此不需要本机 HF rollout 权重。已只重启 worker-0 的 worker tmux，并临时传入 `STAGE_HF_CKPT_TO_TMP=0`，让它跳过 HF staging 并先加入 Ray。该处理没有改脚本默认值；如果要固化，后续可以按 node rank 判断 actor-only 节点跳过 HF staging。

2026-05-22 15:12 左右，64GPU/27B Ray job 已提交：

```text
Ray job id: raysubmit_pRYwBBSwb19EA3Yv
status: RUNNING
Ray active nodes: 3 / 8
active GPU: 24 / 64
placement group: pending, waiting for 64 GPUs
```

此时 launcher、worker-0、worker-1 已进入 Ray；其余 rollout worker 仍在同步 27B HF 权重，抽查到 worker-2 约 35G、worker-3 约 39G、worker-6 约 36G。训练 driver 已开始创建 64-GPU placement group，但 actor checkpoint load 和 SGLang rollout engine 尚未启动，仍在等待剩余节点加入。

2026-05-22 15:31 左右，64GPU/27B 已进入实际运行阶段：

```text
Ray job id: raysubmit_pRYwBBSwb19EA3Yv
status: RUNNING
Ray active nodes: 8 / 8
active GPU: 64 / 64
W&B run: https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/9mnmhp72
```

确认事项：

- 64-GPU placement group 已满足，不再 pending。
- SGLang 27B rollout engines 已在 6 个 rollout 节点启动，HF weights load 成功。
- SGLang server args 生效：`context_len=131072`、`max_prefill_tokens=131072`、`chunked_prefill_size=8192`，每个 2-GPU engine `max_total_num_tokens` 约 773k。
- 每个 SGLang TP rank load weight 后剩余显存约 68G，分配 KV/Mamba cache 后仍有约 22G available，未见 OOM。
- 6 个 `TrialWorker` 已 spawn；下一步需要继续观察 Megatron actor 从 PP8 torch_dist checkpoint 按 `TP=4,PP=2` 加载是否完成。

64GPU monitor 已先启动。第一条快照里 `ray status` 报 “Could not find any running Ray instance”，原因是 Ray head 还没启动；等 staging 结束、launcher 进入 `ray start` 和 `ray job submit` 后，监控脚本会从 launcher log 解析 Ray job id。

2026-05-22 15:38 左右继续复查，64GPU/27B 的 TP4/PP2 actor 没有越过 checkpoint load：

```text
Ray job id: raysubmit_pRYwBBSwb19EA3Yv
status: RUNNING
trajectory_json: 0
rollout_dumps: 0
last actor log: loading release distributed checkpoint from ...Qwen3.5-27B_torch_dist_pp8_timeout60
```

进一步看 actor 节点进程状态，worker-0 上 8 个 `MegatronTrainRayActor.init` 都处于 `D` 状态，GPU 显存约 31G/卡但 GPU 利用率为 0。这说明卡点不是 SGLang、Ray placement group 或显存 OOM，而是 Megatron actor 从 `TP=1,PP=8` checkpoint 在线重分片到 `TP=4,PP=2` 的 checkpoint load/reshard 阶段，表现为共享存储 I/O 或分布式 checkpoint 重分片长时间不返回。

因此本轮把 64GPU/27B 默认训练并行策略改成和已完成转换产物一致：

```text
actor_num_nodes=2
actor_num_gpus_per_node=8
tensor_model_parallel_size=1
pipeline_model_parallel_size=8
data_parallel_size=2
sequence_parallel=0
```

这样仍然用 2 个节点更新 actor，但每个 DP replica 的模型切分和 `/path/to/models/Qwen3.5-27B_torch_dist_pp8_timeout60` 完全一致，避免在线 TP/PP 重分片。为了避免和 TP4/PP2 卡住那次混目录，新 run 使用新的输出后缀：

```text
BASE_FOLDER=.../slime_harbor_qwen35_27b_tb2_full89_64gpu_max100_n16_5epoch_resp2048_ctx128k_actorpp8
SAVE_SUBDIR=Qwen3.5-27B_terminal_bench2_full89_64gpu_max100_n16_5epoch_resp2048_ctx128k_actorpp8
trials_dir=.../slime_harbor_trials/tb2_full89_64gpu_27b_max100_resp2048_ctx128k_actorpp8
W&B group suffix: actorpp8
```

对应修改已写入：

- `examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch.sh`
- `examples/harbor_terminal_rl/harbor_terminal_bench2_full89_64gpu_27b_max100_last_turn.yaml`
- `examples/harbor_terminal_rl/monitor_full89_64gpu_27b_max100.sh`

脚本语法检查已通过：`bash -n` 对 64GPU/27B launcher 脚本和 monitor 脚本均无报错。

随后已停止 `raysubmit_pRYwBBSwb19EA3Yv`，并用 `TP=1,PP=8,DP=2` 版本重启 64GPU/27B。第一版 actorpp8 run 仍然长时间停在 checkpoint load；进一步确认不是 TP/PP 重分片，而是从 gy2 共享文件系统直接加载 51G torch_dist checkpoint 时 actor 进程进入 `D` 状态，表现为 I/O wait/不可中断等待。因此本轮又在通用 launcher 里新增了 torch_dist checkpoint 本机 staging：

```text
STAGE_TORCH_DIST_CKPT_TO_TMP=1
STAGED_TORCH_DIST_CKPT=/tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60
```

27B 的 HF 权重仍用于 SGLang rollout，本机 staged 到 `/tmp/slime-harbor-model-cache/Qwen3.5-27B`；Megatron actor 的 torch_dist 权重也 staged 到本机 `/tmp` 后再作为 `--load/--ref-load` 输入。为了避免 8 个节点同时复制 51G torch_dist，把已经进入 rollout-only 角色的节点用 `STAGE_TORCH_DIST_CKPT_TO_TMP=0` 重启过 worker tmux；实际需要 actor checkpoint 的节点保留本机 torch_dist staging。

当前 64GPU/27B run：

```text
Ray job id: raysubmit_mg9q1rm4MK5GRZZL
W&B run: https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/5p1ng686
launcher tmux: slime_full89_64gpu_27b_max100_launcher
monitor tmux: slime_full89_64gpu_27b_max100_monitor
launcher log: /tmp/slime_full89_64gpu_27b_max100_launcher.log
monitor log: /tmp/slime_full89_64gpu_27b_max100_monitor.log
```

2026-05-22 16:42 复查结果：

- Ray job 为 `RUNNING`，8/8 节点 active，无 pending demand。
- CLI 参数确认 actor 使用本机 staged torch_dist：`--load /tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60`、`--ref-load /tmp/slime-harbor-model-cache/Qwen3.5-27B_torch_dist_pp8_timeout60`。
- Megatron actor 已成功从本机 torch_dist checkpoint load：日志出现 `successfully loaded checkpoint ... [ t 1/1, p 1/8 ] at iteration 0`。
- 首次 `update_weights` 已完成，耗时约 12.5s。
- SGLang rollout engine 使用本机 staged HF 权重，`context_length=131072`、`max_prefill_tokens=131072`、`chunked_prefill_size=8192` 生效。
- TrialWorkers 已进入任务执行和 trajectory dump 阶段；当前已有 24 条 `trajectory.json`，rollout dump 还没有生成，说明仍在首轮 rollout/env 收集阶段，尚未凑齐第一个训练 batch。

这次 27B 的关键修复不是再调显存参数，而是避免 actor 从共享文件系统直接读大 checkpoint。当前 run 已经越过之前所有失败点：HF staging、Ray placement group、SGLang 27B load、Megatron 27B checkpoint load、`update_weights`。后续主要观察 first rollout dump、train step 和保存 checkpoint。

## 2026-05-20 Update: full89 GRPO restart with W&B proxy

本节记录 full terminal-bench-2 89-task GRPO 重启状态。目标仍是 Qwen3.5-9B、`n_samples_per_prompt=16`、约 2 epoch、64 H20，使用 gy2 上的 runtime-augmented SIF cache。

### 这次重启前的额外修改

- `examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh`
  - 保留 `set -e`，避免 `set -x` 打印 Ray runtime env 时泄露 W&B key。
  - 新增 `ENABLE_PROXY=1` 和 `PROXY_ENV_FILE`，默认 source `examples/harbor_terminal_rl/registry_proxy_env.sh`。
  - 把 `http_proxy/https_proxy/ftp_proxy` 及大写版本传入 Ray job 的 `runtime-env-json`，用于 W&B online logging。
  - Ray cluster ready 后用 `ray.nodes()` 收集 active node IP，并把 `localhost/127.0.0.1/0.0.0.0/MASTER_ADDR/所有 Ray node IP/registry_proxy_env.sh 里的 no_proxy` 合并成 `JOB_NO_PROXY`，同时传入 `no_proxy/NO_PROXY`。这样 W&B 可以走代理，Ray/SGLang/Harbor 的内网 HTTP 尽量直连。
  - 仍然支持 `ENABLE_PROXY=0` 关闭代理注入。
- `examples/harbor_terminal_rl/monitor_full89_grpo.sh`
  - 新增轻量监控脚本，默认每 1800 秒记录一次 `ray job status`、`ray status`、SIF/FastAPI/parser warning 计数、rollout dump 文件、checkpoint 文件数和最近关键日志。
  - 当前 launcher 上以 tmux session `slime_full89_monitor` 运行，输出文件为 `/tmp/slime_full89_monitor.log`。

### 重启原因

第一次 full89 重启的 Ray job：

```text
raysubmit_NhdCiAxM67Sawzzn
```

状态已经进入 `RUNNING`，64 GPU placement group 已创建，W&B 参数也正确，但日志停在：

```text
wandb: Network error (ConnectTimeout), entering retry loop.
```

原因是旧脚本开头会 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`，Ray runtime env 也没有把代理传给 W&B driver。该 job 已用 `ray job stop raysubmit_NhdCiAxM67Sawzzn` 停掉。

### 当前重启命令形态

worker 节点仍使用 tmux session `slime_full89_worker`：

```bash
cd /path/to/workspace/code/slime
MASTER_ADDR=10.0.0.4 \
RAY_NODE_IP_PREFIX=29.68 \
RUN_ROLE=worker \
WORKER_KEEP_ALIVE=1 \
CLEAN_START=1 \
examples/harbor_terminal_rl/run_full89_grpo_2epoch_n16.sh
```

launcher 节点使用 tmux session `slime_full89_launcher`，W&B key 只放在远端临时 env 文件 `/tmp/slime_full89_wandb_env.sh`，不写入本文档：

```bash
cd /path/to/workspace/code/slime
source /tmp/slime_full89_wandb_env.sh
MASTER_ADDR=10.0.0.4 \
RAY_NODE_IP_PREFIX=29.68 \
RUN_ROLE=launcher \
CLEAN_START=1 \
examples/harbor_terminal_rl/run_full89_grpo_2epoch_n16.sh
```

### 当前最新状态

新 Ray job 已提交：

```text
raysubmit_SAGKephuEetLXBVB
```

启动日志确认：

```text
ray cluster has 8/8 nodes after 3s
Total resources: 64 GPU
```

训练入口参数确认仍是 full89 runtime SIF 配置：

```text
--custom-config-path /path/to/workspace/code/slime/examples/harbor_terminal_rl/harbor_terminal_bench2_full89_gy2_runtime_sif_last_turn.yaml
--rollout-batch-size 16
--n-samples-per-prompt 16
--global-batch-size 256
--num-rollout 12
--rollout-max-response-len 8096
--rollout-max-context-len 16384
--use-wandb --wandb-mode online
```

后续检查结果：

- `ray job status raysubmit_SAGKephuEetLXBVB`：`RUNNING`。
- W&B 已越过前一次的 `ConnectTimeout`：

```text
wandb: Currently logged in as: yucheng to https://api.wandb.ai.
wandb: Tracking run with wandb version 0.26.1
wandb: Run data is saved locally in .../wandb/wandb/run-20260520_072915-bui0bqvd
wandb: View project at https://wandb.ai/yucheng/slime-harbor-terminal-rl
wandb: View run at https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/bui0bqvd
```

- SGLang rollout engines 已在各节点启动，健康检查返回 200：

```text
The server is fired up and ready to roll!
GET /health_generate HTTP/1.1" 200 OK
GET /health HTTP/1.1" 200 OK
GET /server_info HTTP/1.1" 200 OK
GET /model_info HTTP/1.1" 200 OK
```

截至本记录写入时，核心启动链路已经过了 Ray cluster、W&B、SGLang engine 三个阶段，下一步继续观察第一批 Terminus task 容器和 rollout 样本生成。

### 07:42-07:43 CST 进展

rollout 已经进入 Terminus task 执行阶段。日志里能看到 `RolloutManager` 向 task 容器的 tmux session 发命令，例如：

```text
R --version
cat > /app/ars.R
python3 /app/solve_bn.py
pip install pandas numpy scipy scikit-learn
```

当前检查结果：

```text
ray job status raysubmit_SAGKephuEetLXBVB: RUNNING
cached_sif=24
fastapi_ready=24
no_valid_json=8
extra_text_json=103
traceback=0
rollout_dumps: empty
checkpoint_file_count=0
```

解释：

- 训练已经越过前面的 blocker：Ray、W&B、SGLang、Terminus/SIF task 容器都实际跑起来了。
- `rollout_dumps` 为空是正常的中间状态，第一批 rollout 还没有结束；dump 通常会在一个 rollout id 完成后写出。
- 仍然有 JSON parser warning，主要是 `Extra text detected ...`；目前也出现了少量 `No valid JSON object found`。这说明模型格式仍不干净，但没有看到 `Traceback`，agent loop 仍在继续。
- monitor 已重置为当前 run，旧的 5 月 19 日失败日志不再混入新的 `/tmp/slime_full89_monitor.log`。

### 09:07 CST 进展

当前 job 仍在正常运行：

```text
ray job status raysubmit_SAGKephuEetLXBVB: RUNNING
```

最新计数：

```text
trajectory_count=162
rollout_dump_count=0
checkpoint_file_count=0
cached_sif=162
fastapi_ready=162
no_valid_json=26
extra_text_json=1142
traceback=0
```

最新 trajectory 写入时间：

```text
2026-05-20 09:06:32
/tmp/slime-harbor-terminal-bench2-full89-runtime-trials/slime-r1-g10-s1-i161/agent/trajectory.json
```

解释：

- 训练还没有到 optimizer step；当前仍在第一批 full89 rollout 的 Terminus task 执行阶段。
- 已经有 162 条 Terminus trajectory 落盘，说明 task 容器和 agent loop 仍在持续推进。
- `rollout_dumps` 为空意味着 slime 尚未收到一个完整 rollout batch 的最终 `Sample/train_data` dump；checkpoint 目录未创建也是预期状态。
- 没有 `Traceback`，所以目前不是进程崩溃。主要风险仍是 task rollout 很慢，以及模型输出 JSON 格式 warning 较多。

## 2026-05-19 Update: Qwen3.5-9B + terminal-bench-2

本节记录 64 H20 job 上围绕 Qwen3.5-9B、terminal-bench-2 和 Apptainer 的最新尝试。下面内容优先级高于本文档后面的旧 Qwen3-4B/stable task 记录。

### 新增约束

- `/cq_1` 不能存模型。现在代码和 terminal-bench-2 任务可以从 `/path/to/shared_storage/...` 访问，但模型权重和 torch_dist checkpoint 都放在 `/path/to/shared_storage`。
- 当前 HF 权重路径：`/path/to/models/Qwen3.5-9B`。
- 当前 torch_dist 转换输出：`/path/to/models/Qwen3.5-9B_torch_dist`。
- 为了让 SGLang 启动快，HF checkpoint 可 staging 到节点本地 `/tmp/slime-harbor-model-cache/Qwen3.5-9B`。这是节点临时盘，不是 `/cq_1`。

### 本轮新增/修改的 glue 文件

- `slime_plugins/harbor/import_utils.py`
  - 新增 Harbor source-tree import 兼容层。
  - 解决 `PYTHONPATH=/.../harbor/src` 能找到源码、但 Harbor 未 `pip install -e` 时 `importlib.metadata.version("harbor")` 抛 `PackageNotFoundError` 的问题。
  - 只对 distribution name `harbor` 返回 fallback version `0+slime-src`，不影响其他包。
- `slime_plugins/harbor/llm.py`
  - 支持 `apply_chat_template_kwargs` / `chat_template_kwargs` / `add_generation_prompt`。
  - Qwen3.5-9B config 里设置 `enable_thinking: false` 后会传入 tokenizer `apply_chat_template`。
- `slime_plugins/harbor/data_source.py`
  - 支持直接发现 native terminal-bench task dir：`harbor_tasks_dir` + `harbor_tasks_glob`。
  - 本轮 smoke 使用 `/path/to/workspace/code/terminal-bench-2-main/regex-log/task.toml`。
- `slime_plugins/harbor/singularity_env.py`
  - 保留继承 Harbor 原始 `SingularityEnvironment` 的方式，不改 Harbor 原仓。
  - 新增 `singularity_registry_proxy` / `singularity_registry_no_proxy`，只在 Docker image -> SIF 转换阶段临时设置 proxy。
  - 新增 `singularity_registry_mirror`，当前用 `docker.1panel.live` 代替 Docker Hub blob 下载点；缓存仍按原始 image 名保存，例如 `alexgshaw_regex-log_20251031.sif`。
- `scripts/models/qwen3.5-9B.sh`
  - 新增 Qwen3.5-9B Megatron/slime 参数。
  - 使用 `slime_plugins.models.qwen3_5 get_qwen3_5_spec`，hidden 4096、32 layers、16 heads、4 KV groups、KV head dim 256、FFN 12288、untied embeddings、attention output gate。
- `examples/harbor_terminal_rl/convert_qwen35_9b_torch_dist.sh`
  - 离线转换 HF -> torch_dist。
  - 默认输入 `/path/to/models/Qwen3.5-9B`，默认输出 `/path/to/models/Qwen3.5-9B_torch_dist`。
- `examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh`
  - Qwen3.5-9B + Harbor terminal-bench-2 训练/smoke 脚本。
  - 支持 `RUN_ROLE=launcher|worker`。
  - 默认模型和 checkpoint 都在 `/path/to/shared_storage`。
  - 新增 `STAGE_HF_CKPT_TO_TMP=1`，可在每个节点启动前把 HF 权重 staging 到 `/tmp/slime-harbor-model-cache/Qwen3.5-9B`。
  - 新增 `STAGE_HF_CKPT_FORCE=0` 和 `.slime_stage_complete` marker，避免每次 worker 启动都重新 rsync 19GB HF 权重。
  - 新增固定 Ray 端口：`RAY_PORT`、`RAY_NODE_MANAGER_PORT`、`RAY_OBJECT_MANAGER_PORT`、`RAY_RUNTIME_ENV_AGENT_PORT`、dashboard agent ports、metrics port、worker port range。
  - 新增 `RAY_NODE_IP_PREFIX`，worker 默认优先选择和 `MASTER_ADDR` 同前缀的 IP。当前 64 H20 job 必须用 `RAY_NODE_IP_PREFIX=29.68`，否则 `hostname -I` 的第一个 `29.63.*` 地址会导致部分节点被 GCS health check 标记 dead。
  - 新增 `RAY_CLUSTER_ONLY=1`，只启动 Ray head/worker 并保持 tmux session，方便排查多节点 Ray 稳定性，不提交训练任务。
  - 新增 `WORKER_KEEP_ALIVE=1`，worker 用 tmux 启动后保持 session，不让 `jizhi_client exec` 返回导致 Ray worker 生命周期不稳定。
  - 默认 `APPTAINER_CACHEDIR=/tmp/slime-harbor-apptainer-cache`。
- `examples/harbor_terminal_rl/cleanup_qwen35_9b_ray.sh`
  - 清理本实验的 `slime_qwen35_launcher` / `slime_qwen35_worker` tmux session。
  - 停掉 Ray、SGLang、dashboard/log monitor 和 `train_async.py` 残留。
  - 只针对本实验 session 名和相关进程模式，不清理 `gpu_keeper`。
- `examples/harbor_terminal_rl/harbor_terminal_bench2_config.yaml`
  - 当前 smoke 配置指向 terminal-bench-2 的 `regex-log` 小任务。
  - `apply_chat_template_kwargs.enable_thinking: false`。
  - `SlimeSingularityEnvironment` 使用 Apptainer/Singularity，image cache 在 `/tmp/slime-harbor-terminal-bench2-sif-cache`。
  - 配置了 `star-proxy.oa.com:3128` 和 `docker.1panel.live` mirror。
- `examples/harbor_terminal_rl/harbor_terminal_bench2_config_gy2_sif.yaml`
  - 和 `harbor_terminal_bench2_config.yaml` 一致，但 `singularity_image_cache_dir` 指向 gy2 上的 full terminal-bench-2 SIF cache：
    `/path/to/workspace/harbor_sif_cache_tb2_full`。
  - 用于避免 debug/训练时重新访问 Docker registry。
- `examples/harbor_terminal_rl/run_regex_log_8096_parse_debug.sh`
  - 专门用于 `regex-log` 的 8096-token parser smoke。
  - 默认 `DEBUG_ROLLOUT_ONLY=1`、`NUM_ROLLOUT=1`、`N_SAMPLES_PER_PROMPT=1`、`ROLLOUT_MAX_RESPONSE_LEN=8096`。
  - 默认使用 gy2 SIF cache config。
- `examples/harbor_terminal_rl/inspect_rollout_dump.py`
  - 读取 slime debug rollout dump，统计 response length/status/reward，并用 HF tokenizer 解码 response suffix。
- `examples/harbor_terminal_rl/install_harbor_runtime_deps.sh`
  - 把 Harbor 源码运行时依赖安装脚本化。
  - 安装 `shortuuid/toml/tenacity/litellm/dirhash/claude-agent-sdk/supabase`。
  - 最后把 `openai` pin 回 `2.6.1`，满足当前 SGLang wheel 的依赖约束。我们走 `slime_sglang`，不依赖 LiteLLM 发模型请求。
- `examples/harbor_terminal_rl/registry_proxy_env.sh`
  - 记录用户给的完整 proxy/no_proxy 组合，供手动拉 SIF 或诊断 registry 使用。
- `examples/harbor_terminal_rl/probe_terminal_bench2_image_sizes.py`
  - 查询 terminal-bench-2 Docker Hub manifest layer size，帮助选择小镜像 smoke。
  - 发现 `regex-log`、`dna-insert`、`adaptive-rejection-sampler` 等小镜像约 28 MiB。

### 远端环境状态

job:

```text
self_improving_terminal_agent_rl_Qwen3_5_slime_fixed_dop-fuse
task id: 8b1d81989e25e915019e3c1bb0361459
launcher IP used: 10.0.0.4
```

已确认：

- 8 个节点都有 Apptainer/Singularity：`apptainer version 1.5.0`。
- 8 个节点都能看到 `/path/to/models/Qwen3.5-9B`。
- 8 个节点都能看到 `/path/to/workspace/code/slime`、`harbor/src`、`terminal-bench-2-main`。

### terminal-bench-2 full SIF cache

同事已在 cq1 上准备好 full SIF cache：

```text
/path/to/workspace/Agent_training/tmp/harbor_sif_cache_tb2_full
/path/to/workspace/Agent_training/tmp/harbor_sif_cache_tb2_full
```

检查结果：

- 顶层 `.sif` 文件数：89。
- 和 `terminal-bench-2-main/*/task.toml` 里的 `docker_image` 映射完全一致。
- 没有缺失、没有额外 SIF。
- 抽查 `apptainer sif list alexgshaw_regex-log_20251031.sif` 正常。

已复制到 gy2：

```text
/path/to/workspace/harbor_sif_cache_tb2_full
```

复制后校验：

- target `.sif` 文件数：89。
- source/target 文件名 diff 为空。
- source/target 文件大小 manifest diff 为 0。
- 目标目录大小约 44G。

### 8096-token parser smoke

目的：验证把 Terminus2 每步 response token budget 从 128 提到 8096 后，Qwen3.5-9B 是否还能正常被 JSON parser 解析。

运行方式：

```bash
cd /path/to/workspace/code/slime
CLEAN_START=1 bash examples/harbor_terminal_rl/run_regex_log_8096_parse_debug.sh
```

worker 节点使用：

```bash
cd /path/to/workspace/code/slime
RUN_ROLE=worker WORKER_KEEP_ALIVE=1 CLEAN_START=1 bash examples/harbor_terminal_rl/run_regex_log_8096_parse_debug.sh
```

本次使用：

```text
CONFIG=/path/to/workspace/code/slime/examples/harbor_terminal_rl/harbor_terminal_bench2_config_gy2_sif.yaml
BASE_FOLDER=/path/to/workspace/slime_harbor_qwen35_9b_tb2_regex_log_8096_parse_debug
ROLLOUT_MAX_RESPONSE_LEN=8096
DEBUG_ROLLOUT_ONLY=1
NUM_ROLLOUT=1
ROLLOUT_BATCH_SIZE=1
N_SAMPLES_PER_PROMPT=1
GLOBAL_BATCH_SIZE=1
```

结果：

```text
Job 'raysubmit_ZiSEs8rK4B3JR6Yj' succeeded
debug dump: /path/to/workspace/slime_harbor_qwen35_9b_tb2_regex_log_8096_parse_debug/rollout_dumps/0.pt
dump size: 1.1M
harbor/num_groups: 1
harbor/num_samples: 20
harbor/num_failed_trials: 0
harbor/num_missing_rollout_details: 0
harbor/reward_mean: 0.0
rollout/response_len/mean: 958.55
rollout/response_len/median: 862.5
rollout/response_len/max: 1611
rollout/response_len/min: 550
rollout/truncated_ratio: 0.0
perf/rollout_time: 309.20s
```

Parser 相关日志统计：

```text
No valid JSON object found: 0
Parser warnings: 20
Extra text detected before JSON object: 20
```

结论：

- 8096-token 下没有出现 128-token smoke 里的 `No valid JSON object found`。
- 20 个 turn 都能解析出 action 并执行，dump 中 20 个 slime samples 全部 `status=completed`。
- 所有 response 长度都远低于 8096，最大只有 1611，因此本次没有 SGLang token 截断。
- 模型输出仍然包在 fenced block 里，形如 ````json ... ```<|im_end|>`，所以 Terminus2 parser 会给 `Extra text detected before/after JSON object` warning。这是格式洁净度问题，不是 parse failure。
- `regex-log` 奖励仍为 0，说明任务没做对，但 agent/tool loop 和 JSON parser 链路已经可用。
- launcher 已手动安装 Harbor runtime deps。full 64-card run 前要在所有 worker 上也跑 `install_harbor_runtime_deps.sh`。

### Qwen3.5-9B 离线转换

转换命令：

```bash
cd /path/to/workspace/code/slime
HF_CKPT=/path/to/models/Qwen3.5-9B \
SAVE_DIR=/path/to/models/Qwen3.5-9B_torch_dist \
LOG_FILE=/path/to/models/Qwen3.5-9B_torch_dist.convert.log \
NPROC_PER_NODE=8 \
bash examples/harbor_terminal_rl/convert_qwen35_9b_torch_dist.sh
```

结果：

- `CONVERT_RC=0`。
- 输出目录 `/path/to/models/Qwen3.5-9B_torch_dist`。
- 目录大小约 `17G`。
- `latest_checkpointed_iteration.txt` 和 release checkpoint 已生成。

### tokenizer / disable thinking 验证

远端验证过 Qwen3.5-9B tokenizer:

- `apply_chat_template(..., enable_thinking=False)` 可正常返回 token ids。
- decoded prompt 中仍会出现空的 `<think>\n\n</think>\n\n`，这是 Qwen no-thinking 模板，不代表开启了 active thinking。
- `SlimeSGLangLLM` 现在通过 `apply_chat_template_kwargs.enable_thinking=false` 传入该参数。

### SGLang 加载结论

直接让 SGLang 从共享 `/path/to/shared_storage` HF checkpoint 启动时，曾卡在 shard loading；改为先把 HF 权重 staging 到本机 `/tmp` 后启动正常。

launcher staging 命令曾执行：

```bash
mkdir -p /tmp/slime-harbor-model-cache
rsync -a --info=stats2 \
  /path/to/models/Qwen3.5-9B/ \
  /tmp/slime-harbor-model-cache/Qwen3.5-9B/
```

结果：

- 拷贝约 `19.3GB`，耗时约 8 分钟。
- 单节点 SGLang 从 `/tmp/slime-harbor-model-cache/Qwen3.5-9B` 加载成功。
- GDN/hybrid linear attention、Mamba cache、KV cache、CUDA graph capture 都完成，`/health_generate`、`/health`、`/server_info`、`/model_info` 都 OK。

### terminal-bench-2 / Apptainer 结论

原始 Docker Hub pull 问题：

- `apptainer pull docker://alexgshaw/adaptive-rejection-sampler:20251031` 能通过 registry auth/manifest 阶段。
- 真实 blob 下载会卡在 `production.cloudflare.docker.com` 并 `context deadline exceeded`。
- 用户提供的 `star-proxy.oa.com:3128` 可访问 Docker registry `/v2/`，但不能稳定完成 Cloudflare blob 下载。

mirror 结果：

- `docker.m.daocloud.io` 可访问，但 `alexgshaw/...` 不在白名单。
- `docker.1panel.live` 可用。
- 已成功拉取并转换：

```text
/tmp/slime-harbor-terminal-bench2-sif-cache/alexgshaw_regex-log_20251031.sif
size: 30019584 bytes
```

### 成功 smoke

成功命令：

```bash
cd /path/to/workspace/code/slime
MASTER_ADDR=10.0.0.4 \
EXPECTED_NODES=1 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=8 \
ROLLOUT_NUM_GPUS=8 \
ROLLOUT_NUM_GPUS_PER_ENGINE=8 \
SGLANG_MEM_FRACTION_STATIC=0.50 \
DEBUG_ROLLOUT_ONLY=1 \
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
N_SAMPLES_PER_PROMPT=1 \
GLOBAL_BATCH_SIZE=1 \
ROLLOUT_MAX_RESPONSE_LEN=128 \
HF_CKPT=/tmp/slime-harbor-model-cache/Qwen3.5-9B \
PRETRAINED_CKPT=/path/to/models/Qwen3.5-9B_torch_dist \
BASE_FOLDER=/tmp/slime_harbor_qwen35_9b_tb2_regex_log_smoke \
bash examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
```

结果：

```text
Job succeeded
debug dump: /tmp/slime_harbor_qwen35_9b_tb2_regex_log_smoke/rollout_dumps/0.pt

harbor/num_groups: 1
harbor/num_samples: 20
harbor/num_failed_trials: 0
harbor/num_missing_rollout_details: 0
harbor/reward_mean: 0.0
rollout/response_len/mean: 111.45
rollout/response_len/median: 128.0
rollout/response_len/max: 128
rollout/response_len/min: 52
rollout/truncated_ratio: 0.0
perf/rollout_time: 126.73
```

这说明：

- slime custom data source 能读取 terminal-bench-2 task。
- Harbor Trial 能启动 Terminus2。
- Terminus2 能调用 slime SGLang router 上的 Qwen3.5-9B。
- Apptainer/Singularity terminal env 能启动并执行命令。
- Harbor `rollout_details` 能转成 slime `Sample`。
- slime debug rollout dump 能保存。

`reward_mean=0.0` 的原因不是链路失败，而是基座 Qwen3.5-9B 在 Terminus2 JSON parser 下仍有多次 `No valid JSON object found` warning，行为质量不够稳定。日志中也能看到 agent 实际执行了命令，例如 `ls`、`pwd`、写 `/app/regex.txt` 等。

### 2026-05-19 最新验证状态

本地代码验证：

- `python -m py_compile` 覆盖 Harbor glue、9B 脚本和相关测试，返回 0。
- `bash -n` 覆盖 `examples/harbor_terminal_rl/*.sh`，返回 0。
- `pytest tests/test_harbor_import_utils.py tests/test_harbor_llm.py tests/test_harbor_data_source.py tests/test_harbor_atif_adapter.py` 结果为 `11 passed, 1 skipped`。

远端 launcher 清理状态：

- `nvidia-smi --query-compute-apps=...` 没有返回 GPU 计算进程。
- `pgrep -af "raylet|gcs_server|dashboard_agent|runtime_env_agent|log_monitor|train_async.py|SGLangEngine|sglang"` 没有返回残留进程。
- `ray status` 连接 GCS 超时，符合 Ray head 已退出后的状态。

64 H20 job 前置依赖状态：

- 已在 worker 1-7 上执行 `examples/harbor_terminal_rl/install_harbor_runtime_deps.sh`，返回 0。
- 依赖安装过程中 LiteLLM 会临时安装较新的 `openai`，脚本最后已重新 pin 到 `openai==2.6.1`，以匹配当前 SGLang wheel。
- 8 个节点都验证通过真实 glue import 路径：
  `ensure_harbor_distribution_metadata(); from harbor.trial.trial import Trial`。
- 8 个节点的 `openai.__version__` 都是 `2.6.1`。
- 直接裸 `from harbor...` 仍会失败，因为 Harbor 源码没有安装成 distribution；这是预期现象，slime 插件路径会先调用 metadata fallback。

64 H20 Ray 网络诊断：

- 最开始的多节点 run 会在 Ray job supervisor actor 还没运行前失败，GCS 日志显示 worker 注册成功后被 health check 标记 dead。
- 第一轮修复是固定 Ray 端口，避免 node-manager/object-manager/worker port 随机化。固定端口后 cluster 能短暂到 64 GPU，但随后掉到 32-40 GPU。
- 进一步定位发现节点有很多 IP。launcher 使用 `MASTER_ADDR=10.0.0.4`，worker 原来默认 `hostname -I | awk '{print $1}'`，会拿到 `29.63.*` 地址。部分 `29.63.*` 地址 head 无法访问 node-manager port 18001，GCS 日志报：
  `failed to connect to all addresses; last error: UNKNOWN: ipv4:<worker-ip>:18001: Failed to connect to remote host: FD Shutdown`。
- 修复方式是设置 `RAY_NODE_IP_PREFIX=29.68`，worker 选择 `29.68.*` 地址注册 Ray。cluster-only 验证 60 秒后仍是 8 个 ALIVE node、64 GPU、无 recent failures。

64 H20 debug rollout 成功记录：

```bash
cd /path/to/workspace/code/slime
MASTER_ADDR=10.0.0.4 \
RAY_NODE_IP_PREFIX=29.68 \
EXPECTED_NODES=8 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=8 \
ROLLOUT_NUM_GPUS=56 \
ROLLOUT_NUM_GPUS_PER_ENGINE=1 \
SGLANG_MEM_FRACTION_STATIC=0.50 \
DEBUG_ROLLOUT_ONLY=1 \
NUM_ROLLOUT=1 \
ROLLOUT_BATCH_SIZE=1 \
N_SAMPLES_PER_PROMPT=1 \
GLOBAL_BATCH_SIZE=1 \
ROLLOUT_MAX_RESPONSE_LEN=128 \
STAGE_HF_CKPT_TO_TMP=1 \
HF_CKPT=/path/to/models/Qwen3.5-9B \
PRETRAINED_CKPT=/path/to/models/Qwen3.5-9B_torch_dist \
BASE_FOLDER=/path/to/workspace/slime_harbor_qwen35_9b_tb2_regex_log_64h20_debug_v2 \
bash examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
```

worker 用同一个脚本，额外加：

```bash
RUN_ROLE=worker WORKER_KEEP_ALIVE=1
```

结果：

- Ray cluster: 8 active nodes, 64 GPU。
- Ray job id: `raysubmit_fySDfWYZMFKEMjD1`。
- job status: succeeded。
- debug dump: `/path/to/workspace/slime_harbor_qwen35_9b_tb2_regex_log_64h20_debug_v2/rollout_dumps/0.pt`。
- dump 大小约 188 KiB，包含 `rollout_id` 和 `samples`。
- `samples` 数量为 20。
- 第一个 sample 验证字段：
  `tokens=1078`、`response_length=128`、`len(loss_mask)=128`、`reward=0.0`、`status=completed`。
- Harbor metrics：
  `harbor/num_groups=1`、`harbor/num_samples=20`、`harbor/num_failed_trials=0`、`harbor/num_missing_rollout_details=0`、`harbor/reward_mean=0.0`、`rollout/response_len/mean=128.0`、`perf/rollout_time=131.79s`。
- 这轮结束后已运行 `cleanup_qwen35_9b_ray.sh` 清理 8 个节点的 `slime_qwen35_*` tmux session、Ray 和 SGLang 残留。launcher 上只剩原有 `gpu_keeper`。

### 当前剩余 blocker / 下一步

1. 当前这个 64 H20 job 的 worker runtime deps 已安装完成；新申请 job 仍需要在所有 worker 上安装 Harbor runtime deps：

```bash
cd /path/to/workspace/code/slime
bash examples/harbor_terminal_rl/install_harbor_runtime_deps.sh
```

2. full 64-card 前，每个会跑 SGLang engine 的节点最好 staging HF 到本地 `/tmp`：

```bash
STAGE_HF_CKPT_TO_TMP=1 \
HF_CKPT=/path/to/models/Qwen3.5-9B \
bash examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
```

worker role 也支持同样的 staging env。已加 marker，已有 staging 的节点不会重复拷贝。

3. terminal-bench-2 全量任务需要批量预热 SIF。当前只验证了 `regex-log`。Docker Hub 直连 blob 下载不稳定，建议继续使用 `docker.1panel.live` mirror，或者把常用 SIF cache 放到 `/path/to/shared_storage/.../data/...` 这类共享数据目录。
4. Qwen3.5-9B 基座能跑通链路，但 Terminus2 parser 仍有 JSON warning。要提高 reward，需要后续做 prompt/parser/stop token 或使用更适合 terminal-agent 的 instruct/SFT checkpoint。
5. 当前 64 H20 成功的是 `--debug-rollout-only`。下一步可以取消 `DEBUG_ROLLOUT_ONLY`，先用 `regex-log` 跑 1 个 rollout 的 train/update/save smoke，再扩大任务数量。

本文档总结当前 Harbor Terminus 2 接入 slime 做 terminal-agent RL 的修改状态，方便 review。

## 目标和约束

目标是把 Harbor 的 Terminus 2 trial/ATIF 轨迹接到 slime 的训练入口。slime 训练侧最终需要的是 `Sample`/`train_data`，最低字段包括：

- `tokens`: prompt + response 的完整 token ids
- `response_length`: response suffix 长度
- `reward`
- `status`
- `loss_mask`: 长度必须等于 `response_length`

约束：

- 尽量不改 Harbor 原始代码。
- 尽量不改 slime core。
- 新增 glue code，并通过 slime 已有 hook 接入。
- Terminus 环境不能依赖 Docker，后续需要用 Apptainer/Singularity 或其他可用 backend host terminal env。

## 当前实现形态

新增代码放在 slime repo 内的 `slime_plugins/harbor/`，利用 slime 当前 `setup.py` 已经包含的 `slime_plugins*` package pattern，不需要改安装配置。

当前数据流：

```text
slime train.py
  -> --data-source-path slime_plugins.harbor.data_source.HarborTaskDataSource
  -> --rollout-function-path slime_plugins.harbor.rollout.generate_rollout
  -> Harbor Trial.create(...).run()
  -> SlimeTerminus2 agent
  -> SlimeSGLangLLM calls slime SGLang router /generate
  -> Harbor rollout_details / ATIF-like token fields
  -> atif_adapter converts to slime Sample
  -> slime default train_data converter
```

这条路径的核心原则是让 Harbor 继续负责 terminal-agent trial、environment、verifier 和 ATIF 采集，让 slime 继续负责 SGLang rollout server、Megatron actor、weight update 和 RL loss。

## 当前结论

截至 2026-05-18，glue 链路已经验证到 `--debug-rollout-only`：

- Harbor stable task 可以在 Apptainer/Singularity 环境中启动。
- Harbor trial 可以调用 slime 已经启动的 SGLang router。
- SGLang `/generate` 可以返回 token/logprob。
- Harbor `rollout_details`/ATIF-like token 字段可以转换为 slime `Sample`。
- slime 可以保存 debug rollout dump，且 sample 字段满足训练入口最低要求。

最后一次成功 smoke 的关键指标：

```text
HF_CKPT=/tmp/slime-harbor-model-cache/Qwen3-4B
DEBUG_ROLLOUT_ONLY=1
NUM_ROLLOUT=1
N_SAMPLES_PER_PROMPT=1
GLOBAL_BATCH_SIZE=1
ROLLOUT_MAX_RESPONSE_LEN=128

harbor/num_groups: 1
harbor/num_samples: 20
harbor/num_failed_trials: 0
harbor/num_missing_rollout_details: 0
harbor/reward_mean: 0.0
rollout/response_len/mean: 128.0
debug dump: /tmp/slime_harbor_qwen3_4b_rollout_smoke_after_tokenfix/rollout_dumps/0.pt
```

`reward_mean=0.0` 的原因不是链路失败，而是基础 Qwen3-4B 在当前 Terminus JSON prompt 下反复输出非 JSON，parser 给 warning。后续需要调模型、prompt、parser、stop tokens 或先接更适合 terminal-agent 的 checkpoint。

## 新增文件

### `slime_plugins/harbor/atif_adapter.py`

负责把 Harbor `TrialResult` 或 ATIF-like `rollout_details` 转成 slime `Sample`。

支持的 sample mode：

- `turn`: 每个 LLM call 生成一条 slime `Sample`。这是当前默认模式，也是最稳的模式。
- `last_turn`: 只保留每个 rollout detail 的最后一次 LLM call。
- `trajectory`: 把多轮 terminal trajectory 拼成一条 slime `Sample`。模型输出 token 的 `loss_mask=1`，环境 observation delta 的 `loss_mask=0`。

字段映射：

- `tokens = prompt_token_ids[i] + completion_token_ids[i]`
- `response_length = len(completion_token_ids[i])`
- `loss_mask = [1] * response_length`
- `reward` 默认从 `result.verifier_result.rewards["reward"]` 取
- `status` 根据 `exception_info` 映射到 `COMPLETED`、`FAILED`、`TRUNCATED`
- 如果 Harbor 提供每 token logprob，并且长度匹配，会写入 `sample.rollout_log_probs`

`trajectory` 模式额外要求下一轮 prompt 必须以前一轮完整 token 序列为 prefix。否则抛 `RolloutConversionError`，提示改用 `turn` 或 `last_turn`。

### `slime_plugins/harbor/data_source.py`

新增 `HarborTaskDataSource`，把 Harbor task/trial entry 包成 slime `Sample`，供 rollout function 消费。

配置来源：

- `args.harbor_tasks`
- 或 `args.harbor_tasks_path`
- 或 `args.harbor_stable_tasks_path`

每条 entry 会放到：

```python
sample.metadata["harbor_task_entry"]
```

支持 entry 形式：

- 字符串路径：转换为 `{"task": {"path": path}}`
- dict：原样保留，并允许包含 `task`、`agent`、`environment`、`verifier`、`trial_config`
- stable_rl_tb_rl parquet/json：先物化为 Harbor task 目录，再转换为 `{"task": {"path": ...}}`

这个 data source 是 read-only，不支持把样本加回 buffer。

### `slime_plugins/harbor/stable_tasks.py`

负责把 `stable_rl_tb_rl/data/train.parquet` 或 `tasks_info.json` 物化成
Harbor task 目录，避免修改 stable_rl_tb_rl 或 Harbor 原始数据格式。

当前物化内容：

- `instruction.md`
- `task.toml`
- `environment/Dockerfile`
- `environment/files/setup.sh`
- `tests/test_outputs.py`
- `tests/test_weights.json`
- `tests/test.sh`
- `solution/solve.sh`

实现细节：

- 从 parquet 的 `extra_info.instruction` 取任务说明。
- 从 `extra_info.dockerfile_contents` 解析 `FROM`、`WORKDIR`、`ENV`、`RUN`、
  `COPY`、`ADD`，生成容器启动后的 `setup.sh`。
- 把已知 base image 映射到本地 SIF：
  - `ghcr.io/laude-institute/t-bench/python-3-13:20250620`
    -> `python-3-13_20250620.sif`
  - `ghcr.io/laude-institute/t-bench/ubuntu-24-04:latest`
    -> `ubuntu-24-04_latest.sif`
- `tests/test.sh` 用 pytest 执行 stable task 的测试，并写 Harbor 可读取的
  `/logs/verifier/reward.json`。该文件只包含数值，避免 Harbor
  `VerifierResult` 校验失败；测试细节写到 `/logs/verifier/details.json`。

当前 config 使用的 stable 数据：

```yaml
harbor_stable_tasks_path: /path/to/workspace/code/stable_rl_tb_rl/data/train.parquet
harbor_materialized_tasks_dir: /tmp/slime-harbor-stable-tasks
harbor_stable_sif_cache_dir: /path/to/workspace/code/stable_rl_tb_rl/data/sif_cache
harbor_stable_tasks_limit: 1
harbor_stable_tasks_offset: 0
```

### `slime_plugins/harbor/llm.py`

新增 `SlimeSGLangLLM`，作为 Harbor BaseLLM 风格的 adapter，让 Harbor agent 调用 slime 已启动的 SGLang router。

行为：

- 用 `args.hf_checkpoint` 加载 tokenizer。
- 用 tokenizer `apply_chat_template` 把 Harbor message history 编码为 prompt token ids。
- 对 tokenizer 输出做标准化展开，兼容 `list[int]`、batch `input_ids`、
  `BatchEncoding`、tensor-like `tolist()` 等返回形态；这是必要的，因为当前
  Qwen3 tokenizer 在远端环境会返回 `BatchEncoding`，直接 `list(...)` 会得到
  `["input_ids", "attention_mask"]`，导致 SGLang router 报
  `invalid_generate_request: InputIds`。
- POST 到 slime SGLang router 的 `/generate`。
- 请求 `return_logprob=True`。
- 返回 Harbor `LLMResponse`，包含：
  - `content`
  - `prompt_token_ids`
  - `completion_token_ids`
  - `logprobs`
  - `UsageInfo`

注意事项：

- 当前实现假设 SGLang `/generate` response 中有 `meta_info.output_token_logprobs`。
- 如果后续 SGLang response schema 变化，需要在这里适配。

### `slime_plugins/harbor/agent.py`

新增 `SlimeTerminus2Mixin` 和 `SlimeTerminus2`。

行为：

- 继承 Harbor 原始 `Terminus2`。
- 只覆盖 LLM 初始化逻辑。
- 当 `llm_backend == "slime_sglang"` 时使用 `SlimeSGLangLLM`。
- 其他 Terminus2 行为仍走 Harbor 原始实现。

这符合当前原则：不直接修改 Harbor Terminus2 原始代码。

### `slime_plugins/harbor/rollout.py`

新增 slime rollout hook：

```python
slime_plugins.harbor.rollout.generate_rollout
```

行为：

- 从 `data_source.get_samples(args.rollout_batch_size)` 取 Harbor task entries。
- 为每个 slime base sample 构造 Harbor `TrialConfig`。
- 调用 `Trial.create(...).run()`。
- 用 `atif_adapter.samples_from_trial_result(...)` 转成 slime `Sample`。
- 返回 `RolloutFnTrainOutput(samples=..., metrics=...)`。

TrialConfig 合并顺序：

1. `args.harbor_trial_defaults`
2. entry 内的 `trial_config`
3. entry 顶层的 `task`、`agent`、`environment`、`verifier`

默认 agent 配置：

```yaml
agent:
  import_path: slime_plugins.harbor.agent:SlimeTerminus2
  model_name: slime-policy
  kwargs:
    collect_rollout_details: true
    enable_summarize: false
    record_terminal_session: false
    llm_backend: slime_sglang
```

默认 environment：

```yaml
environment:
  import_path: slime_plugins.harbor.singularity_env:SlimeSingularityEnvironment
```

默认 router URL：

```text
http://{args.sglang_router_ip}:{args.sglang_router_port}
```

可配置项：

- `harbor_sample_mode`
- `harbor_reward_key`
- `harbor_num_concurrent_trials`
- `harbor_fail_on_trial_error`
- `harbor_fail_on_conversion_error`
- `harbor_trials_dir`
- `harbor_agent_import_path`
- `harbor_model_name`
- `harbor_llm_backend`
- `harbor_environment_type`
- `harbor_singularity_image_cache_dir`
- `harbor_singularity_no_mount`
- `harbor_trial_name_template`

### `slime_plugins/harbor/singularity_env.py`

新增 `SlimeSingularityEnvironment`，继承 Harbor 原始
`SingularityEnvironment`，只替换容器内 server bootstrap 行为。

当前解决的问题：

- 集群不能使用 Docker，需要走 Apptainer/Singularity。
- 当前 Jizhi pod 在 `--containall --pid` 下无法挂载新的 `/proc`，因此使用
  `singularity_no_mount: home,tmp,bind-paths,proc`。
- `--containall` 下容器 `/etc/resolv.conf` 为空，导致无法解析代理/PyPI 域名。
  自定义环境显式 bind 宿主 `/etc/resolv.conf`。
- Apptainer 启动时继承宿主 cwd 会尝试 chdir 到 `/cq_1/.../slime`，在 contain
  模式下可能 permission denied。自定义环境把 `singularity exec` 子进程 cwd
  固定为 `/tmp`。
- Harbor 原始 server 在容器内 hard-code `127.0.0.1`。在当前 Apptainer
  contain 环境中宿主 health check 连不上容器 loopback。自定义环境只 patch
  staging 副本，把 Uvicorn host 改为 `0.0.0.0`，不修改 Harbor 源码。
- 首次启动 stable task 需要较长 setup，health check 等待窗口从 60 秒扩到
  240 秒。

### `slime_plugins/harbor/bootstrap_singularity.sh`

容器内 bootstrap 脚本，替换 Harbor 默认 bootstrap。

当前行为：

- 设置集群 apt/pip 代理默认值。
- 优先用 `get-pip.py --break-system-packages` 引导 pip，避免
  `apt-get install python3-pip` 拉取大量依赖。
- 安装 Harbor server 所需的 `uvicorn`、`fastapi`。
- 执行物化任务的 `/staging/env_files/setup.sh`。
- 确保 `tmux` 可用。

### `examples/harbor_terminal_rl/smoke_stable_nop.py`

用于验证 Harbor + stable task + Apptainer + verifier 的最小脚本，不启动模型。

远程已验证命令：

```bash
cd /tmp
PYTHONPATH=/path/to/workspace/code/slime:/path/to/workspace/code/harbor/src:$PYTHONPATH \
python /path/to/workspace/code/slime/examples/harbor_terminal_rl/smoke_stable_nop.py \
  --stable-tasks-path /path/to/workspace/code/stable_rl_tb_rl/data/train.parquet \
  --materialized-tasks-dir /tmp/slime-harbor-stable-tasks-smoke4 \
  --sif-cache-dir /path/to/workspace/code/stable_rl_tb_rl/data/sif_cache \
  --trials-dir /tmp/slime-harbor-trials-smoke4
```

验证结果：

- `exception_info: null`
- `verifier_result.rewards.reward: 0.0`
- `pytest_exit_code: 1`

这里 reward 为 0 是符合预期的，因为 smoke 使用 Harbor `nop` agent，没有修改
terminal 环境；这个 smoke 证明环境启动、server 通信、测试执行和 reward 解析
链路已经打通。

### `examples/harbor_terminal_rl/harbor_slime_config.yaml`

示例 custom config。slime 的 `--custom-config-path` 会把 YAML key 挂到
`args` 上，因此不需要修改 argparse。

当前配置默认读取 stable_rl_tb_rl 的 `train.parquet`，物化 1 个任务做 smoke。
也可以改回原生 Harbor task entry：

```yaml
harbor_tasks:
  - task:
      path: /path/to/harbor/task
```

默认使用：

- `SlimeTerminus2`
- `llm_backend: slime_sglang`
- `SlimeSingularityEnvironment`
- `harbor_sample_mode: turn`

### `examples/harbor_terminal_rl/run_qwen35_4b_single_node.sh`

新增单节点 8 卡 Qwen3.5-4B + Harbor rollout 启动脚本。

默认路径：

```bash
CODE_ROOT=/path/to/workspace/code
SLIME_DIR=$CODE_ROOT/slime
HARBOR_SRC=$CODE_ROOT/harbor/src
HF_CKPT=/path/to/models/Qwen3.5-4B
MODEL_SCRIPT=$SLIME_DIR/scripts/models/qwen3.5-4B.sh
BASE_FOLDER=/path/to/workspace/slime_harbor_qwen35_4b
```

默认资源：

- `ACTOR_NUM_NODES=1`
- `ACTOR_NUM_GPUS_PER_NODE=8`
- `--colocate`
- `ROLLOUT_NUM_GPUS_PER_ENGINE=8`，默认单 SGLang engine 占 8 卡，避免
  多个 engine 同时重复读取 Qwen3.5-4B checkpoint
- smoke-sized rollout:
  - `NUM_ROLLOUT=2`
  - `ROLLOUT_BATCH_SIZE=1`
  - `N_SAMPLES_PER_PROMPT=2`
  - `GLOBAL_BATCH_SIZE=2`

脚本会把 Ray runtime env 的 `PYTHONPATH` 设置为：

```text
/root/Megatron-LM/:$HARBOR_SRC:$SLIME_DIR:$PYTHONPATH
```

如果 PATH 中没有 `singularity`，但有 `apptainer`，脚本会创建临时 shim：

```bash
/tmp/slime-harbor-bin/singularity -> $(command -v apptainer)
```

如果二者都不存在，会打印 warning。当前远端 launcher 已安装
`apptainer 1.5.0`，且 `/usr/bin/singularity` 会解析到 Apptainer。

脚本现在支持：

- `CLEAN_START=0`：跳过启动前清理。
- `DEBUG_ROLLOUT_ONLY=1`：加 `--debug-rollout-only`，只验证模型 rollout/数据链路，
  不做 actor train/update/save。
- `ROLLOUT_NUM_GPUS_PER_ENGINE`：控制每个 SGLang engine 使用几张 GPU，默认 8。
- `SGLANG_MEM_FRACTION_STATIC`：控制 SGLang 静态显存比例，默认 0.5。
- `MODEL_SCRIPT`：切换 Megatron 模型结构参数，默认
  `scripts/models/qwen3.5-4B.sh`。
- `MEGATRON_TO_HF_MODE`：默认 `bridge`；测试标准 Qwen3 时可以设为空字符串。

启动前清理已避免 broad `pkill -9 python`，只清理 SGLang、Ray 相关进程和
`python train.py`。

模型 smoke 记录：

- 2026-05-17 03:00 左右用 `ROLLOUT_NUM_GPUS_PER_ENGINE=2` 启动了 4 个
  SGLang engine。
- Ray placement、router、engine 进程、torch distributed 都成功启动。
- 4 个 engine 同时加载 `/path/to/models/Qwen3.5-4B` 时长期停在
  `Multi-thread loading shards: 0/2`，无 traceback/OOM。
- 已停止该 Ray job 并清理 GPU 进程；因此脚本默认改为单 engine 8 GPU。
- 2026-05-17 03:14 左右用默认 `ROLLOUT_NUM_GPUS_PER_ENGINE=8` 重跑，确认只启动
  1 个 SGLang engine，`tp_size=8`，torch distributed 初始化成功，但仍长期停在
  `Multi-thread loading shards: 0/2`。
- 已停止第二个 Ray job 并清理 GPU/Ray 进程。
- 轻量 checkpoint 检查结果：
  - `/path/to/models/Qwen3.5-4B` 下有 2 个 safetensors shard，
    总大小约 8.8GB。
  - `safetensors.safe_open` metadata 约 1.44 秒返回。
  - `AutoConfig` 可读取，类型为 `Qwen3_5Config`，architecture 为
    `Qwen3_5ForConditionalGeneration`。
- `/gy_1` 顺序读取较慢，实测小 shard 约 12MB/s，大 shard 前 256MiB
  约 21.6MB/s。直接从 `/gy_1` 加载 Qwen3.5-4B 或 Qwen3-4B 会表现为
  长时间停在 `Multi-thread loading shards`。
- 2026-05-18 把 `/path/to/models/Qwen3-4B` rsync 到
  `/tmp/slime-harbor-model-cache/Qwen3-4B` 后，SGLang TP8 权重加载约 0.6 秒。
- 2026-05-18 首次 Qwen3-4B local-cache smoke 进入 Harbor，但 SGLang
  `/generate` 返回 400：
  `invalid_generate_request: data did not match any variant of untagged enum InputIds`。
  原因是 tokenizer 返回 `BatchEncoding`，旧 adapter 把 mapping key 当成 token
  发给 router。
- 2026-05-18 修复 `SlimeSGLangLLM` token 标准化后，重跑：
  - `HF_CKPT=/tmp/slime-harbor-model-cache/Qwen3-4B`
  - `MODEL_SCRIPT=scripts/models/qwen3-4B.sh`
  - `MEGATRON_TO_HF_MODE=`
  - `QKV_FORMAT=`
  - `DEBUG_ROLLOUT_ONLY=1`
  - `NUM_ROLLOUT=1`
  - `N_SAMPLES_PER_PROMPT=1`
  - `GLOBAL_BATCH_SIZE=1`
  - `ROLLOUT_MAX_RESPONSE_LEN=128`
  结果成功，debug dump 写到
  `/tmp/slime_harbor_qwen3_4b_rollout_smoke_after_tokenfix/rollout_dumps/0.pt`。
  slime metrics 显示：
  - `harbor/num_groups: 1`
  - `harbor/num_samples: 20`
  - `harbor/num_failed_trials: 0`
  - `harbor/num_missing_rollout_details: 0`
  - `harbor/reward_mean: 0.0`
  - `rollout/response_len/mean: 128.0`

当前模型侧结论：Harbor + Apptainer + SGLang + ATIF-to-Sample 链路已经可以跑通
最小 rollout。`/gy_1` checkpoint 读取慢是模型加载“卡住”的主要原因，应优先使用
节点本地模型缓存。Qwen3.5-4B 仍未用本地缓存复测，且基础 Qwen3-4B 在该 prompt 下
没有输出 valid Terminus JSON，因此 reward 仍为 0；这是后续 prompt/model/RL 质量问题，
不是 glue 链路 blocker。

标准 Qwen3 对照命令：

```bash
DEBUG_ROLLOUT_ONLY=1 \
NUM_ROLLOUT=1 \
N_SAMPLES_PER_PROMPT=1 \
GLOBAL_BATCH_SIZE=1 \
ROLLOUT_MAX_RESPONSE_LEN=128 \
HF_CKPT=/tmp/slime-harbor-model-cache/Qwen3-4B \
MODEL_SCRIPT=/path/to/workspace/code/slime/scripts/models/qwen3-4B.sh \
MEGATRON_TO_HF_MODE= \
QKV_FORMAT= \
SAVE_SUBDIR=Qwen3-4B_harbor_slime \
BASE_FOLDER=/tmp/slime_harbor_qwen3_4b_rollout_smoke \
CONFIG=/path/to/workspace/code/slime/examples/harbor_terminal_rl/harbor_slime_config.yaml \
bash examples/harbor_terminal_rl/run_qwen35_4b_single_node.sh
```

### `examples/harbor_terminal_rl/install_apptainer_ubuntu.sh`

新增 Ubuntu launcher pod 的 Apptainer 安装脚本。

行为：

- 使用 `APT_PROXY` 配置 apt/add-apt-repository 代理，默认是当前集群可用的 `http://star-proxy.oa.com:3128`。
- 添加官方 `ppa:apptainer/ppa`。
- 安装 `uidmap` 和 `apptainer`。
- 补 root fakeroot 所需的 subordinate uid/gid：
  - `/etc/subuid`: `root:100000:65536`
  - `/etc/subgid`: `root:100000:65536`
- 如果提供 `SIF_SMOKE_TEST`，会执行 Harbor-style 参数 smoke：
  - `--no-mount home,tmp,bind-paths,proc`
  - `--writable-tmpfs`
  - `--fakeroot`
  - `--containall`
  - `--pid`

### `examples/harbor_terminal_rl/README.md`

新增运行说明、路径说明、single-node script 用法和 sample mode 说明。

关键说明：

- 当前 Taiji/Jizhi job 内 `/path/to/shared_storage` 是 stale mount。
- 远端应使用 `/path/to/shared_storage`。
- launcher 可以看到 `user/<user>/code`。
- worker-0 只看到 `models` 和 `_tmp_bench`，不能看到 `code`，多节点训练前必须修复 mount 视图。
- 当前 launcher 已安装 `apptainer 1.5.0`，并配置了 root subordinate uid/gid。
- 当前 pod 需要 `singularity_no_mount: home,tmp,bind-paths,proc` 才能避开 `/proc` mount 权限问题。

### `tests/test_harbor_atif_adapter.py`

新增纯 adapter 单测。

覆盖：

- `turn` 模式
- `last_turn` 模式
- `trajectory` 模式下 environment observation delta mask 为 0
- 非 prefix trajectory 抛错
- `TrialResult` reward/status/metadata 抽取

本地测试环境没有 torch 时，测试内提供了 minimal torch stub，避免纯转换逻辑依赖完整训练环境。

### `tests/test_harbor_llm.py`

新增 LLM adapter 的 tokenizer 输出标准化单测。

覆盖：

- mapping/batch 形态：`{"input_ids": [[1, 2, 3]]}`
- object 形态：`.input_ids`
- tensor-like 形态：`.tolist()`
- 多 batch 输出时抛错，避免把多条 prompt 混成一条发给 SGLang

这个测试对应实际远程问题：Qwen3 tokenizer 在远程环境里返回 `BatchEncoding`，
旧代码 `list(BatchEncoding)` 会得到 `["input_ids", "attention_mask"]`，造成 SGLang
router 返回：

```text
400 Bad Request
invalid_generate_request
Failed to deserialize /generate request: data did not match any variant of untagged enum InputIds
```

## 完整代码变更清单

本轮 Harbor/slime glue 遵循“不改 Harbor core、不改 slime core”的原则。新增逻辑集中在
`slime_plugins/harbor/`、`examples/harbor_terminal_rl/` 和两个测试文件。

### Glue package

- `slime_plugins/harbor/__init__.py`
  - package marker，便于 slime 的动态 import path 加载。
- `slime_plugins/harbor/atif_adapter.py`
  - ATIF/rollout detail 到 slime `Sample` 的转换层。
  - 保证 `tokens`、`response_length`、`reward`、`status`、`loss_mask` 字段符合 slime 训练入口要求。
  - 支持 `turn`、`last_turn`、`trajectory` 三种样本构造模式。
- `slime_plugins/harbor/data_source.py`
  - slime data source hook。
  - 把 Harbor task entry 包成 pending `Sample`，实际 task 信息放在 metadata 中。
  - 支持 native Harbor entry 和 stable_rl_tb_rl parquet/json 物化入口。
- `slime_plugins/harbor/stable_tasks.py`
  - stable_rl_tb_rl 数据物化为 Harbor task 目录。
  - 解析 parquet/json 中的 instruction、dockerfile、pytest、weights、附加文件。
  - 生成 Harbor 可消费的 `instruction.md`、`task.toml`、`environment/`、`tests/`、`solution/`。
  - 映射 stable SIF cache，避免重新 Docker build。
- `slime_plugins/harbor/llm.py`
  - Harbor `BaseLLM` 风格 adapter。
  - 调 slime SGLang router `/generate`，再把 response 包成 Harbor `LLMResponse`。
  - 后续补了 `_normalize_token_ids`，修复 Qwen3 tokenizer 返回 `BatchEncoding` 时的 400。
  - 400 时会带上 status、prompt token 数、max_new_tokens 和响应体，方便定位 router/schema 问题。
- `slime_plugins/harbor/agent.py`
  - `SlimeTerminus2Mixin` 和 `SlimeTerminus2`。
  - 只覆盖 Terminus2 的 LLM 初始化；`llm_backend=slime_sglang` 时使用 `SlimeSGLangLLM`。
- `slime_plugins/harbor/rollout.py`
  - slime rollout function hook。
  - 读取 data source 的 task entry，构造 Harbor `TrialConfig`，运行 trial，再转成 `RolloutFnTrainOutput`。
  - 汇总 Harbor 相关 metrics，例如 sample 数、failed trial 数、reward mean。
- `slime_plugins/harbor/singularity_env.py`
  - 继承 Harbor 原始 `SingularityEnvironment`。
  - 使用 Apptainer/Singularity 运行 terminal env。
  - patch staging 里的 server host 为 `0.0.0.0`，不改 Harbor 源码。
  - 固定子进程 cwd 为 `/tmp`，避免 contain 模式下 chdir 到 shared fs 失败。
  - bind 宿主 `/etc/resolv.conf`，解决容器 DNS/proxy 解析。
  - 增加 `singularity_bind_host_proc`，配合 `--no-mount proc` 时仍给 tmux 一个可用 `/proc`。
  - health check 等待窗口扩到 240 秒，覆盖 stable task 首次 apt/pip setup。
- `slime_plugins/harbor/bootstrap_singularity.sh`
  - 容器内启动脚本。
  - 设置 apt/pip proxy 默认值。
  - 引导 pip 并安装 `uvicorn`、`fastapi`。
  - 执行 stable task setup。
  - 检查 tmux/asciinema。

### Example and scripts

- `examples/harbor_terminal_rl/harbor_slime_config.yaml`
  - 示例 custom config。
  - 配置 stable_rl_tb_rl 数据源、SIF cache、物化目录。
  - 配置 `SlimeTerminus2`、`SlimeSingularityEnvironment`、`harbor_sample_mode=turn`。
  - 配置 `singularity_no_mount: home,tmp,bind-paths,proc` 和 `singularity_bind_host_proc: true`。
- `examples/harbor_terminal_rl/run_qwen35_4b_single_node.sh`
  - 单节点 8 卡启动脚本。
  - 默认 Qwen3.5-4B；可通过 `HF_CKPT`、`MODEL_SCRIPT` 切换到其他 Qwen3 模型。
  - 支持 `DEBUG_ROLLOUT_ONLY`、`CLEAN_START`、`ROLLOUT_NUM_GPUS_PER_ENGINE`、
    `SGLANG_MEM_FRACTION_STATIC`、`MEGATRON_TO_HF_MODE`、`SAVE_SUBDIR`、`QKV_FORMAT`。
  - 清理逻辑只杀 Ray/SGLang/train.py 相关进程，不再 broad `pkill python`。
  - Ray runtime env 设置 Harbor/slime/Megatron 的 `PYTHONPATH`。
- `examples/harbor_terminal_rl/smoke_stable_nop.py`
  - 不启动模型，只用 Harbor `nop` agent 验证 stable task 物化、Apptainer env、server、verifier。
- `examples/harbor_terminal_rl/install_apptainer_ubuntu.sh`
  - Ubuntu launcher pod 安装 Apptainer 的辅助脚本。
  - 安装 `uidmap`、`apptainer`，配置 root subordinate uid/gid。
- `examples/harbor_terminal_rl/README.md`
  - 用户侧运行说明、路径说明、模型 smoke 命令、sample mode 说明。
- `examples/harbor_terminal_rl/INTEGRATION_SUMMARY.md`
  - 本文档，记录设计、代码变更、运行尝试和后续路线。

### Tests

- `tests/test_harbor_atif_adapter.py`
  - ATIF adapter 的纯单测。
- `tests/test_harbor_llm.py`
  - LLM tokenizer 输出标准化的纯单测。

## 为了跑起来做过的尝试和结论

### 1. 远程路径确认

用户指出远程节点上 `/path/to/shared_storage` 被改名为
`/path/to/shared_storage`。验证结果：

- 远程 `/path/to/shared_storage` 是 stale/broken mount，会报
  `Transport endpoint is not connected`。
- 后续远程命令全部切到 `/path/to/shared_storage`。
- launcher 可见 `/path/to/workspace/code`。
- worker-0 不可见 `code`，只看到 `models` 和 `_tmp_bench`，所以先退回单节点 8 卡。

### 2. Docker 不可用，转 Apptainer/Singularity

集群不能用 Docker，因此验证 Apptainer 路线：

- launcher 上最终可用：
  - `/usr/bin/apptainer`
  - `/usr/bin/singularity`
  - version `1.5.0`
- 配置了 root fakeroot 所需：
  - `/etc/subuid`: `root:100000:65536`
  - `/etc/subgid`: `root:100000:65536`
- 使用 stable_rl_tb_rl 已有 SIF：
  - `ubuntu-24-04_latest.sif`
  - `python-3-13_20250620.sif`

最初 Harbor-style 参数会在挂载 `/proc` 时失败：

```text
mount proc->/proc error: operation not permitted
```

因此配置：

```yaml
singularity_no_mount: home,tmp,bind-paths,proc
```

但完全没有 `/proc` 后 tmux 又失败，所以补了：

```yaml
singularity_bind_host_proc: true
```

对应命令逻辑是 `--no-mount proc` 加 `-B /proc:/proc`。这个组合在当前 pod 中可用，
并解决了 tmux 的 `/proc` 依赖。

### 3. stable_rl_tb_rl 资产调研

用户建议检查：

```text
/path/to/workspace/code/stable_rl_tb_rl
```

远程等价路径：

```text
/path/to/workspace/code/stable_rl_tb_rl
```

实际可用资产：

- `data/train.parquet`
- `data/val.parquet`
- `data/tasks_info.json`
- `data/sif_cache/ubuntu-24-04_latest.sif`
- `data/sif_cache/python-3-13_20250620.sif`

结论：这里解决的是 SIF 和 task 数据来源，不是缺 Apptainer 软件本身。缺的是把
stable task 格式转成 Harbor task/trial config，因此新增了 `stable_tasks.py`。

### 4. NOP smoke

目的：不启动模型，只验证 Harbor + stable task + Apptainer + verifier。

结果：

- Harbor FastAPI server 可启动。
- task setup 可执行。
- verifier 可执行。
- reward JSON 可被 Harbor 解析。
- `exception_info=null`。
- `reward=0.0`，符合 nop agent 预期。

这一步证明 terminal env/backend 方向可行。

### 5. Qwen3.5-4B 直接从 `/gy_1` 加载

目标：用户提出“一个 8 卡节点 + `/path/to/models/Qwen3.5-4B` 是否可以”。

尝试：

- 单节点 8 卡。
- 默认 colocate。
- 初始尝试 `ROLLOUT_NUM_GPUS_PER_ENGINE=2`，会启动 4 个 SGLang engine。
- 后续改为 `ROLLOUT_NUM_GPUS_PER_ENGINE=8`，只启动 1 个 TP8 SGLang engine。

结果：

- Ray placement 成功。
- SGLang router 成功。
- SGLang torch distributed 初始化成功。
- 但 checkpoint loading 长时间停在 `Multi-thread loading shards: 0/2`。
- 无 traceback/OOM。

结论：不是 Harbor/Apptainer 层的问题，问题在模型加载或远程 FS IO。

### 6. Qwen3-4B 直接从 `/gy_1` 加载

为了排除 Qwen3.5 自定义模型实现，换标准 Qwen3-4B：

```text
/path/to/models/Qwen3-4B
```

结果仍停在：

```text
Multi-thread loading shards: 0/3
```

随后做 IO probe：

- 小 shard 约 `12 MB/s`
- 大 shard 前 256MiB 约 `21.6 MB/s`

结论：`/gy_1` 顺序读取太慢，SGLang 多线程 shard loading 看起来像“卡住”。

### 7. 把 Qwen3-4B 复制到本地 `/tmp`

执行：

```bash
rsync -a /path/to/models/Qwen3-4B/ /tmp/slime-harbor-model-cache/Qwen3-4B/
```

结果：

- 模型总量约 7.6GB。
- rsync 平均约 23.8MB/s。
- 本地 `/tmp` 读取约 9.7GB/s。
- SGLang TP8 权重加载约 0.6 到 0.8 秒完成。

结论：模型加载 blocker 主要是远程 FS IO，优先使用节点本地模型缓存。

### 8. tmux `/proc` 问题

第一次 local-cache run 中，模型已经正常加载，Harbor 容器也启动，但 tmux 报：

```text
no server running on /tmp/.harbor-tmux/tmux-0/default
```

ATIF 中：

```json
"rollout_details": []
```

原因：为了绕过 Apptainer proc mount 权限，容器没有可用 `/proc`，tmux 依赖 `/proc`。

修复：

- `singularity_env.py` 增加 `singularity_bind_host_proc`。
- config 中启用：

```yaml
singularity_no_mount: home,tmp,bind-paths,proc
singularity_bind_host_proc: true
```

验证：

```text
singularity exec ... --no-mount proc ... -B /proc:/proc ...
readlink /proc/self/exe
PROC_BIND_OK
```

### 9. SGLang `/generate` 400

修复 `/proc` 后，Harbor trial 能进入 LLM 调用，但 SGLang router 返回 400。

最初日志只有：

```text
Client error '400 Bad Request' for url ... /generate
```

因此增强 `SlimeSGLangLLM` 错误信息，带上 status、prompt token 数、max_new_tokens、
response body。下一轮得到：

```text
SGLang /generate request failed status=400, prompt_tokens=2, max_new_tokens=512,
body={"error":{"type":"Bad Request","code":"invalid_generate_request","message":"Failed to deserialize /generate request: data did not match any variant of untagged enum InputIds"}}
```

`prompt_tokens=2` 对一个 Terminus prompt 明显不合理。单独在远程测 tokenizer：

```python
ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
print(type(ids), len(ids), ids[:10], [type(x).__name__ for x in list(ids)[:5]])
```

输出显示：

```text
<class 'transformers.tokenization_utils_base.BatchEncoding'> 2 ...
['str', 'str']
```

也就是说旧代码 `list(ids)` 得到的是 mapping keys，而不是 token ids。

修复：

- 新增 `_normalize_token_ids`。
- 支持 mapping/object/tensor-like/batched input ids。
- `SlimeSGLangLLM._encode_messages` 和 fallback encode 都走标准化。
- 新增 `tests/test_harbor_llm.py`。

### 10. 成功 rollout smoke

最终成功命令形态：

```bash
cd /path/to/workspace/code/slime
CLEAN_START=1 \
DEBUG_ROLLOUT_ONLY=1 \
NUM_ROLLOUT=1 \
N_SAMPLES_PER_PROMPT=1 \
GLOBAL_BATCH_SIZE=1 \
ROLLOUT_MAX_RESPONSE_LEN=128 \
HF_CKPT=/tmp/slime-harbor-model-cache/Qwen3-4B \
MODEL_SCRIPT=/path/to/workspace/code/slime/scripts/models/qwen3-4B.sh \
MEGATRON_TO_HF_MODE= \
QKV_FORMAT= \
SAVE_SUBDIR=Qwen3-4B_harbor_slime \
BASE_FOLDER=/tmp/slime_harbor_qwen3_4b_rollout_smoke_after_tokenfix \
CONFIG=/path/to/workspace/code/slime/examples/harbor_terminal_rl/harbor_slime_config.yaml \
bash examples/harbor_terminal_rl/run_qwen35_4b_single_node.sh
```

结果：

- Ray job 成功。
- SGLang `/generate` 多次返回 200。
- Harbor trajectory dump 写到：
  `/tmp/slime-harbor-trials/slime-r0-g0-s0-i0/agent/trajectory.json`
- slime debug rollout dump 写到：
  `/tmp/slime_harbor_qwen3_4b_rollout_smoke_after_tokenfix/rollout_dumps/0.pt`
- 产出 20 条 trainable sample。
- 没有 failed trial。
- 没有 missing rollout details。

不足：

- model 输出非 JSON，Terminus parser 反复 warning：

```text
Parser warnings: - No valid JSON object found
```

- 所以 reward 仍为 0。

这一步证明 glue code 的数据链路可用，但不代表当前 base model 已经能完成 terminal task。

### 11. 进程清理

每次失败或成功 smoke 后都做过清理：

```bash
ray stop --force
pkill -9 sglang || true
pkill -9 -f "raylet|gcs_server|dashboard_agent|runtime_env_agent|log_monitor|python3 train.py|python train.py" || true
```

最后一次清理后确认：

- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`
  无 compute 进程。
- `ps ... | grep -E "train.py|SGLangEngine|sglang|raylet|gcs_server|ray::"` 无相关进程。

## slime 接入点

启动参数需要包含：

```bash
--rollout-function-path slime_plugins.harbor.rollout.generate_rollout
--data-source-path slime_plugins.harbor.data_source.HarborTaskDataSource
--custom-config-path examples/harbor_terminal_rl/harbor_slime_config.yaml
```

slime core 的默认 train data converter 会继续负责把 `Sample` 转成训练 batch。当前 adapter 保证 `loss_mask` 长度等于 `response_length`，满足 slime 的校验条件。

## 远端环境验证结果

用户提供的 job：

```text
self_improving_agent_RL_training_qwen3_5_16GPU
8b1d82379e1b64a1019e1f38661402ef
```

launcher 节点验证结果：

- 可进入远端。
- `nvidia-smi -L` 显示 8 张 A100-SXM4-40GB。
- `/path/to/workspace/code/slime/slime_plugins/harbor` 可见。
- `/path/to/workspace/code/harbor/src/harbor/agents/terminus_2` 可见。
- `/path/to/models/Qwen3.5-4B` 存在。
- Qwen3.5-4B 模型大小约 `8.8G`。
- tokenizer 可加载，输出：
  - class: `TokenizersBackend`
  - vocab size: `248044`
- `ray --version`: `2.55.1`
- Python package import 检查：
  - torch `2.9.1+cu129`
  - ray `2.55.1`
  - sglang `0.5.10.post1`

worker-0 验证结果：

- `nvidia-smi -L` 显示 8 张 A100-SXM4-40GB。
- `/path/to/shared_storage` 存在。
- 但 `/path/to/workspace/code` 不存在。
- 同一路径有重复 mount 记录。
- worker-0 当前不能直接参与从 shared code tree 启动的多节点训练。

旧路径状态：

- `/path/to/shared_storage` 在远端返回 `Transport endpoint is not connected`。
- 后续远端命令和脚本应使用 `/path/to/shared_storage`。

container runtime 状态：

- launcher 已安装 `apptainer 1.5.0`。
- launcher `/usr/bin/singularity` 可用，输出同为 `apptainer version 1.5.0`。
- launcher 已补 root subordinate uid/gid：
  - `/etc/subuid`: `root:100000:65536`
  - `/etc/subgid`: `root:100000:65536`
- 普通 SIF 启动已通过：
  - `apptainer exec .../ubuntu-24-04_latest.sif cat /etc/os-release`
- Harbor 风格参数如果不加 `--no-mount proc` 会失败：
  - `mount proc->/proc error: operation not permitted`
- 加上 `--no-mount proc` 后已通过：
  - `singularity exec --no-mount home --no-mount tmp --no-mount bind-paths --no-mount proc --writable-tmpfs --fakeroot --containall --pid ... echo harbor_exact_no_mount_ok`

`stable_rl_tb_rl` 里可复用的资产：

- SIF cache:
  - `data/sif_cache/python-3-13_20250620.sif`
  - `data/sif_cache/ubuntu-24-04_latest.sif`
- 数据:
  - `data/tasks_info.json`
  - `data/train.parquet`
  - `data/val.parquet`
- `tasks_info.json` 当前是 331 个 entry 的 dict，entry 字段包括：
  - `task_path`
  - `dockerfile_contents`
  - `py_test_file_contents`
  - `additional_files`
  - `test_weights`
  - `max_test_timeout_sec`

这些数据不是 Harbor native task directory/config，还需要转换成 Harbor task/verifier/environment entry，或写一个 data-source adapter 直接从 `tasks_info.json` 生成 `TrialConfig`。

## 已执行验证

本地 slime repo：

```bash
python -m py_compile \
  slime_plugins/harbor/__init__.py \
  slime_plugins/harbor/atif_adapter.py \
  slime_plugins/harbor/data_source.py \
  slime_plugins/harbor/llm.py \
  slime_plugins/harbor/agent.py \
  slime_plugins/harbor/rollout.py
```

结果：通过。

本地 adapter 单测：

```bash
PYTHONPYCACHEPREFIX=/tmp/slime-harbor-pycache \
python -m pytest tests/test_harbor_llm.py tests/test_harbor_atif_adapter.py
```

结果：`9 passed`。

远端 launcher adapter 单测：

```bash
cd /path/to/workspace/code/slime
PYTHONPYCACHEPREFIX=/tmp/slime-harbor-pycache \
python -m pytest tests/test_harbor_llm.py tests/test_harbor_atif_adapter.py
```

结果：`9 passed`。

远端 launcher py_compile：

```bash
cd /path/to/workspace/code/slime
PYTHONPATH=/path/to/workspace/code/harbor/src:/path/to/workspace/code/slime \
python -m py_compile slime_plugins/harbor/*.py
```

结果：通过。

远端 launcher import：

```bash
PYTHONPATH=/path/to/workspace/code/harbor/src:/path/to/workspace/code/slime \
python -c "import slime_plugins.harbor.atif_adapter; import slime_plugins.harbor.data_source"
```

结果：通过。

远端 single-node script syntax：

```bash
cd /path/to/workspace/code/slime
bash -n examples/harbor_terminal_rl/run_qwen35_4b_single_node.sh
```

结果：通过。

## 当前没有完成的部分

还没有跑完整 Harbor terminal-agent RL train/update end-to-end；目前验证到
`--debug-rollout-only` 的 rollout dump 产出。

主要 blocker：

1. 多节点暂时不可用，因为 worker-0 的 `/path/to/workspace`
   目录视图和 launcher 不一致，看不到 `code`。
2. 当前 Apptainer 安装和 `/etc/subuid`/`/etc/subgid` 修改是在这个运行中的
   launcher pod 上完成的；新 job 如果镜像不持久化，可能需要重复安装或做进
   启动脚本/基础镜像。
3. Qwen3.5-4B 直接从 `/gy_1` 加载过慢；建议先复制到节点本地 `/tmp` 再测。
4. 基础 Qwen3-4B 在当前 prompt 下反复输出非 JSON，rollout 链路可用但 task reward
   为 0，需要后续调 prompt、parser、stop tokens、模型或先用 SFT/更合适的
   instruct/checkpoint 做 smoke。

次要风险：

1. `SlimeSGLangLLM` 当前依赖 SGLang `/generate` 的 `meta_info.output_token_logprobs` schema。
2. `trajectory` 模式要求 Harbor 相邻 prompt histories 是严格 prefix extension，不满足时需要用默认 `turn` 模式。
3. Qwen3.5-4B 单节点 colocate 会让 actor 和 rollout engine 共享 8 卡，smoke 配置已经保守，但真正训练时还要按显存和吞吐调 batch/response len。
4. Harbor task 的 verifier reward key 默认是 `reward`，如果实际 verifier 使用其他 key，需要配置 `harbor_reward_key`。

## 当前推荐推进路径

推荐先按单节点 8 卡做最小 smoke，不碰 worker-0。

步骤：

1. 保持 stable task limit 为 1，用本地缓存模型继续做单节点 8 卡 smoke。
2. 确认 terminal environment backend：
   - 当前 launcher 已有 `apptainer`/`singularity`。
   - 保持 `singularity_no_mount: home,tmp,bind-paths,proc`。
   - 新 job 如果没有 runtime，需要在启动前安装 Apptainer 或提供 `singularity` shim。
3. 用 `NUM_ROLLOUT=1`、`ROLLOUT_BATCH_SIZE=1`、`N_SAMPLES_PER_PROMPT=1`
   跑 `--debug-rollout-only`，确认 dump 字段。
4. 检查生成的 Harbor trial result 是否包含 `rollout_details`。
5. 检查 slime debug rollout dump 中样本字段：
   - `tokens`
   - `response_length`
   - `reward`
   - `status`
   - `loss_mask`
6. 再取消 `DEBUG_ROLLOUT_ONLY`，跑 1 个 rollout 的 train/update/save smoke。
7. 最后扩大到当前脚本默认 smoke：`NUM_ROLLOUT=2`、`N_SAMPLES_PER_PROMPT=2`。

## 和当前工作树其他变化的关系

当前 `git status` 还显示一些 Qwen3.5 脚本、Megatron/Qwen 相关文件和文档处于 modified/untracked 状态，例如：

- `slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py`
- `slime/ray/train_actor.py`
- `scripts/run-qwen3.5-4B.sh`
- `scripts/run-qwen3.5-27B-*.sh`
- `QWEN3.5-*.md`

这些不属于本 Harbor-Terminus/slime glue 总结范围。本文档覆盖的是：

- `slime_plugins/harbor/`
- `examples/harbor_terminal_rl/`
- `tests/test_harbor_atif_adapter.py`
- `tests/test_harbor_llm.py`

## 2026-05-19 全量 89 任务 GRPO 2 epoch 运行计划

这次目标是用 64 张 H20 跑 Qwen3.5-9B + Terminus2 agent RL：

- task set: `/path/to/workspace/code/terminal-bench-2-main`
- task count: 89 个 `*/task.toml`
- HF checkpoint: `/path/to/models/Qwen3.5-9B`
- torch_dist checkpoint: `/path/to/models/Qwen3.5-9B_torch_dist`
- SIF cache: `/path/to/workspace/harbor_sif_cache_tb2_full`
- 禁止把模型放到 `/cq_1`

新增/修改文件：

- `examples/harbor_terminal_rl/harbor_terminal_bench2_full89_gy2_sif_last_turn.yaml`
  - 使用全量 `*/task.toml`，不设置 limit。
  - 使用 gy2 上的 full SIF cache。
  - `enable_thinking: false`。
  - `harbor_sample_mode: last_turn`。
  - `harbor_num_concurrent_trials: 16`。
- `examples/harbor_terminal_rl/run_full89_grpo_2epoch_n16.sh`
  - 全量 89 任务的推荐启动 wrapper。
  - `n_samples_per_prompt=16`。
  - `rollout_batch_size=16`。
  - `global_batch_size=256`。
  - `num_rollout=12`。
  - `rollout_max_response_len=8096`。
  - `rollout_max_context_len=16384`。
  - `seq_length=16384`。
- `examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh`
  - 增加 `SAVE_INTERVAL` 环境变量。
  - 增加 `ROLLOUT_MAX_CONTEXT_LEN` 环境变量。
  - 增加 `SEQ_LENGTH`/`MAX_TOKENS_PER_GPU`/`USE_DYNAMIC_BATCH_SIZE`/`LOG_PROBS_CHUNK_SIZE` 环境变量。

为什么不用 `--num-epoch 2`：

slime 当前 `num_epoch` 会用 `len(data_source) // rollout_batch_size` 计算每个 epoch 的 rollout 数。89 个任务、`rollout_batch_size=16` 时，`89 // 16 = 5`，2 epoch 只会得到 10 个 rollout，也就是 160 个 task groups，会丢掉每轮尾部 9 个任务。

因此这次显式设置：

```bash
NUM_ROLLOUT=12
ROLLOUT_BATCH_SIZE=16
N_SAMPLES_PER_PROMPT=16
GLOBAL_BATCH_SIZE=256
```

12 个 rollout 会覆盖 192 个 task groups，相当于完整覆盖 89 个任务两遍以后再 rollover 14 个任务。这样比 `--num-epoch 2` 更接近“全量 2 epoch”，并且不会丢尾部任务。

为什么第一版用 `last_turn`：

- `turn` 模式会把一条 terminal trajectory 展成最多 20 个 slime Samples；n=16 时每个 task group 会变成最多 320 条样本，reward grouping 和训练 batch 会被 turn count 放大。
- `trajectory` 模式语义更完整，但要求相邻 prompt history 是严格 prefix extension，并且样本长度可能明显变长。
- `last_turn` 先保证每个 task group 正好 16 条样本，GRPO 组内比较最稳定，也更适合第一版全量训练 smoke。

当前推荐启动方式：

```bash
cd /path/to/workspace/code/slime
MASTER_ADDR=10.0.0.4 \
RAY_NODE_IP_PREFIX=29.68 \
RUN_ROLE=launcher \
examples/harbor_terminal_rl/run_full89_grpo_2epoch_n16.sh
```

worker 节点：

```bash
cd /path/to/workspace/code/slime
MASTER_ADDR=10.0.0.4 \
RAY_NODE_IP_PREFIX=29.68 \
RUN_ROLE=worker \
WORKER_KEEP_ALIVE=1 \
examples/harbor_terminal_rl/run_full89_grpo_2epoch_n16.sh
```

保存位置：

```text
/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_grpo_n16_2epoch/Qwen3.5-9B_terminal_bench2_full89_grpo_n16_2epoch/
```

注意：

- `SAVE_INTERVAL=4`，并且 slime 的 `should_run_periodic_action` 会在最后一个 rollout 强制 save，所以 12 个 rollout 正常完成时一定会保存最终 checkpoint。
- 这个保存是 slime/Megatron checkpoint，不是 HF checkpoint；如果需要 HF checkpoint，还要在训练完成后做一次 torch_dist/Megatron 到 HF 的转换。
- 这次全量训练仍然是第一次 89-task n=16 规模运行。前置 smoke 已经确认 8096 token 下 regex-log JSON parser 没有 hard failure，但其他 88 个任务还没有逐个跑过。

## 2026-05-19 全量运行记录

### 第一次 64 H20 尝试

第一次按 `run_full89_grpo_2epoch_n16.sh` 启动后，Ray 8 节点和 56 个 rollout GPU 都能起来，但 actor checkpoint load 卡住。日志里出现：

```text
(TP, PP) mismatch after resume ((2, 1) vs (1, 8) from checkpoint): RNG state will be ignored
```

本次 torch_dist checkpoint 是按 `TP=1, PP=8` 转出来的，而脚本最初使用 `TP=2, PP=1`。为了避免运行时重切分和 load 阶段卡住，脚本已经改成：

```bash
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=8
SEQUENCE_PARALLEL=0
```

### 启动脚本补丁

`run_qwen35_9b_terminal_bench2_64h20.sh` 还做了两个集群启动稳定性修正：

- worker 侧 `ray start` 增加 retry，避免 head GCS 刚起来时 worker 直接退出。
- worker/head 侧 `ray status` 改成 `ray status || true`，避免状态查询短暂失败导致 tmux session 退出。

### 当前正在跑的作业

北京时间 2026-05-19 13:28 左右重新 clean start，13:31 左右 Ray cluster 到 8/8 节点，13:32 提交 Ray job：

```text
raysubmit_6YcnvR5T392YdGFv
```

当前入口参数确认：

- `--rollout-batch-size 16`
- `--n-samples-per-prompt 16`
- `--num-rollout 12`
- `--global-batch-size 256`
- `--rollout-max-response-len 8096`
- `--rollout-max-context-len 16384`
- `--seq-length 16384`
- `--tensor-model-parallel-size 1`
- `--pipeline-model-parallel-size 8`
- `--rollout-num-gpus 56`
- `--actor-num-gpus-per-node 8`

截至 13:36 左右的状态：

- Ray job 状态是 `RUNNING`。
- Ray cluster 是 8/8 active nodes、64 GPUs。
- 56 个 SGLang rollout engines 已开始加载 HF 权重并陆续通过 health check。
- Megatron actor 已建立 8-rank Gloo group。
- actor 已开始从 `/path/to/models/Qwen3.5-9B_torch_dist` 加载 `release` distributed checkpoint。
- 本次重启后的日志里没有再看到 TP/PP mismatch。

13:40 以后确认：

- distributed checkpoint 已成功加载：

```text
successfully loaded checkpoint from /path/to/models/Qwen3.5-9B_torch_dist [ t 1/1, p 1/8 ] at iteration 0
```

- actor 已把权重同步给 56 个 SGLang rollout engines。
- rollout engines 已恢复 generation。
- RolloutManager 已进入 Harbor/Terminus2 rollout，开始使用 gy2 SIF cache 启动 task container。
- 第一批日志显示正在启动 `alexgshaw_adaptive-rejection-sampler_20251031.sif` 的并发 trials。

13:47 左右继续确认：

- `/tmp/slime-harbor-terminal-bench2-full89-trials` 下已经有 16 个 `trajectory.json`。
- trial 目录已经从 `g0` 继续到 `g1`，说明不是卡在第一组 sample 初始化。
- 日志中可以看到模型生成、parser 解析、tmux 向 task container 发送命令、trajectory 持续 dump。
- 当前 parser 主要是 warning，例如：

```text
Parser warnings: - Extra text detected before JSON object
Parser warnings: - No valid JSON object found
```

这些 warning 目前没有导致 Ray job 失败；需要等 rollout 汇总阶段再确认是否有 conversion hard failure。

14:08 左右状态：

- Ray job `raysubmit_6YcnvR5T392YdGFv` 仍是 `RUNNING`。
- Ray cluster 仍是 8/8 active nodes，没有 pending nodes 或 recent failures。
- 日志仍在持续推进，能看到 SGLang `/generate`、Terminus2 `trajectory.json` dump、Singularity task container 启动/清理。
- `/tmp/slime-harbor-terminal-bench2-full89-trials` 下已有 33 个 `trajectory.json`。
- 当前已经跑到后续任务，例如 `alexgshaw_bn-fit-modify_20251031.sif` 和 `alexgshaw_break-filter-js-from-html_20251031.sif`。
- `rollout_dumps` 和 checkpoint 目录暂时还没有文件，说明第一轮 slime rollout 还没有全部收齐并进入训练/update/save 阶段。
- 有 parser warnings 和 Singularity container retry/force-kill cleanup 日志，但没有看到 Ray job failure。

15:07 左右状态：

- Ray job `raysubmit_6YcnvR5T392YdGFv` 仍是 `RUNNING`。
- Ray cluster 仍是 8/8 active nodes，没有 pending nodes 或 recent failures。
- `/tmp/slime-harbor-terminal-bench2-full89-trials` 下已有 64 个 `trajectory.json`。
- `rollout_dumps=0`，`checkpoints=0`，仍在第一轮 rollout 收样本阶段。
- 日志显示正在跑后续 task，例如需要下载/构建旧 POV-Ray 的任务；仍有 parser warning，但没有 Ray job failure。
- 已在 launcher 节点启动 `slime_full89_monitor`，每 30 分钟向 `/tmp/slime_full89_monitor.log` 追加一次状态摘要。

23:09 左右状态：

- Ray job `raysubmit_6YcnvR5T392YdGFv` 已失败。
- 最终没有写出 `rollout_dumps`，也没有 checkpoint。
- `/tmp/slime-harbor-terminal-bench2-full89-trials` 下有 149 个 `trajectory.json`。
- 失败前多个 `slime-r1-g15-*` trials 出现环境启动超时：

```text
Trial slime-r1-g15-s0-i240 failed: Environment start timed out after 600.0 seconds
...
Trial slime-r1-g15-s15-i255 failed: Environment start timed out after 600.0 seconds
```

- slime 在 rollout 汇总阶段触发硬检查：

```text
ValueError: Not enough samples 149 for global_batch_size 256
Job 'raysubmit_6YcnvR5T392YdGFv' failed
```

初步结论：

- Qwen3.5-9B 权重加载、SGLang engines、Harbor/Terminus2 rollout、SIF cache 都已经端到端跑通。
- 当前卡点是全量 terminal task rollout 的有效样本数不稳定，`global_batch_size=256` 对第一轮 16 prompts x 16 samples 太严格。
- 下一轮建议先把训练 batch 改成能容忍部分 trial failure 的设置，例如 `GLOBAL_BATCH_SIZE=128`，或者同时降低 `ROLLOUT_BATCH_SIZE` 到 8 并保持 `N_SAMPLES_PER_PROMPT=16`。
- 长期更稳的做法是在 Harbor adapter/rollout 侧补齐失败 trial 的样本，给失败样本 reward=0/status=failed/loss_mask 正常生成，这样每个 prompt 仍然保留 n=16 的 group size。

还没完成的确认点：

- `rollout_dumps/{rollout_id}.pt` 是否开始落盘。
- 12 个 rollout 是否能完成并写出最终 Megatron checkpoint。
- 如果要得到 HF 格式模型，训练完成后还需要从 Megatron/torch_dist checkpoint 再转换一次。

### 23:45 复盘：async RL、W&B、container timeout

#### 当前到底有没有用 async RL

本次启动用的是 slime 的 async driver，不是同步 `train.py`。launcher 实际提交的是：

```bash
python3 train_async.py \
  --rollout-function-path slime_plugins.harbor.rollout.generate_rollout \
  ...
```

`train_async.py` 的行为是：先启动 rollout N，拿到 N 的数据后，立刻启动 rollout N+1，然后训练 N。因此它能把下一轮 rollout 和当前轮训练重叠起来。但这不是 `examples/fully_async/` 里展示的 fully async RL。

fully async 示例依赖 `fully_async_rollout.generate_rollout_fully_async`，里面有一个常驻后台 worker 不断拉 prompt、生成、把完成样本放进队列；训练侧每次只是 drain 已完成样本。我们没有直接用它，原因是 Harbor/Terminus2 不是普通 SGLang prompt rollout：

- Harbor rollout 每个 sample 都要启动/管理一个 stateful terminal task container。
- 当前 `slime_plugins.harbor.rollout.generate_rollout` 会等一个 rollout batch 的 Harbor trials 汇总后才返回给 slime。
- Harbor trial 失败时当前逻辑会丢掉该 sample；没有 fully async 所需的持久队列、失败补样、retry、按 group size 补齐机制。
- 直接换成 fully async 示例会绕开 Harbor 的 trial lifecycle，不能直接处理 ATIF/Terminus2 轨迹。

所以当前是“slime async driver + Harbor batch rollout”，不是“Harbor fully async worker”。如果要做 fully async，需要新增一个 Harbor-specific async rollout 模块，而不是直接改 slime 核心：常驻 worker 拉 Harbor tasks，按 `n_samples_per_prompt` 管 group，失败样本补齐或重试，训练侧从 completed queue 取完整 groups。

#### W&B 记录方式

已在 `run_qwen35_9b_terminal_bench2_64h20.sh` 加入 W&B 粘合参数，不把 token 写死进脚本：

- `USE_WANDB=1`
- `WANDB_MODE=online`
- `WANDB_PROJECT=slime-harbor-terminal-rl`
- `WANDB_GROUP=qwen35-9b-terminal-bench2`
- `WANDB_DIR=${BASE_FOLDER}/wandb`
- `WANDB_RUN_ID` 可选，用于 resume/固定 run id。

`run_full89_grpo_2epoch_n16.sh` 默认打开：

```bash
USE_WANDB=1
WANDB_PROJECT=slime-harbor-terminal-rl
WANDB_GROUP=qwen35-9b-tb2-full89-grpo-n16-2epoch
WANDB_MODE=online
```

启动前应在 tmux 里单独 export：

```bash
export WANDB_API_KEY="<user-provided-wandb-token>"
```

密钥不写入文档和 git 文件；当前文档只记录“使用用户提供的 W&B token”。脚本在 `ray job submit` 前会临时关闭 xtrace，避免 `--runtime-env-json` 把 `WANDB_API_KEY` 明文打进 launcher 日志。

同时在 launcher 脚本增加了 `USE_DYNAMIC_GLOBAL_BATCH_SIZE` 开关。默认不开；它只是为下一步实验提供控制面，不能单独解决 Harbor failed sample 导致的 GRPO group size 不完整问题。

#### Terminus task container 启动超时的判断

本次 failure 不是模型 GPU OOM，也不是 Qwen 权重或 SGLang engine 没起来。证据：

- actor checkpoint 已按 `TP=1, PP=8` 成功加载。
- 56 个 SGLang engines 已服务 `/generate`。
- Harbor/Terminus2 已产生 149 个 `trajectory.json`。
- 失败点在 Harbor task environment start 阶段。

代表性日志：

```text
Trial slime-r1-g15-s10-i250 failed: Environment start timed out after 600.0 seconds
singularity.py:661 - Sent SIGTERM to Singularity process ...
singularity.py:666 - Graceful shutdown timed out, force killing
singularity image preserved at .../alexgshaw_count-dataset-tokens_20251031.sif for reuse
```

同一段日志还能看到 container server bootstrap 在 task image 内安装依赖：

```text
[server] [harbor] Installing server dependencies (uvicorn/fastapi)...
[server] Collecting uvicorn
[server] Collecting fastapi
...
[server] Successfully installed ...
[server] [harbor] Installing tmux...
```

所以更准确的判断是：container 启动路径过慢，受到 Apptainer/Singularity 启动、`--writable-tmpfs --fakeroot --containall --pid`、pip/apt 安装、共享文件系统/网络、以及 `harbor_num_concurrent_trials=16` 并发的叠加影响。它属于 CPU/IO/网络/容器 runtime 竞争，不是 H20 GPU 数量不足。

这也解释了为什么最后 slime 报的是：

```text
ValueError: Not enough samples 149 for global_batch_size 256
```

Harbor 失败 trial 被当前 rollout glue 丢弃，导致第一轮原本应有 `16 * 16 = 256` 个样本，最终只有 149 个有效样本。slime 在 `_get_rollout_data` 中按 `global_batch_size=256` 检查并 trim；149 小于 256，没有可训练 batch，因此直接失败，还没进入真正的 train step。

#### 下一步建议

短期先解决可训练性：

- 降低 `harbor_num_concurrent_trials`，例如 16 -> 8，减少 container startup 竞争。
- 增大 Harbor `timeout_multiplier`，例如 1.0 -> 2.0 或 3.0，先验证这些 SIF 是否只是慢启动。
- 不建议只改 `GLOBAL_BATCH_SIZE=128` 作为唯一修复，因为 GRPO/pass@k 还默认按 `rollout_batch_size * n_samples_per_prompt` 看完整 group；单纯 trim 可能破坏 group 形状。
- 更稳的 glue 修复是在 Harbor rollout 侧对 failed trial 生成占位 Sample：`reward=0`、`status=FAILED`、`response_length=1`、`loss_mask=[0]`，这样保持每个 prompt 的 n=16 group size，同时失败样本不贡献 policy gradient。

中期再做 Harbor fully async：

- 新增 `slime_plugins.harbor.fully_async_rollout`，不改 slime 原始 rollout。
- worker 常驻拉 tasks 和 sample ranks，container trial 完成后写入 completed queue。
- 训练侧每次只取完整 groups；超时/失败则 retry 或补占位 sample。
- W&B 记录 rollout/container metrics：启动耗时、超时数、失败 SIF、有效样本数、补样数、每组 pass@k。

## 2026-05-20 container 反复安装深挖

### 直接原因

反复安装不是 terminal-bench SIF cache 没命中。日志和远端检查都确认 SIF cache 命中了：

```text
Using cached Singularity image: /path/to/workspace/harbor_sif_cache_tb2_full/alexgshaw_count-dataset-tokens_20251031.sif
```

但这个预构建 SIF 只是 task docker image 本身，不包含我们运行 Harbor environment server 所需的 shim 依赖。

远端对失败 task 的 SIF 做了检查：

```bash
apptainer inspect /path/to/workspace/harbor_sif_cache_tb2_full/alexgshaw_count-dataset-tokens_20251031.sif
```

结果显示它是从同事预构建的 docker archive 转成 SIF：

```text
org.label-schema.usage.singularity.deffile.bootstrap: docker-archive
org.label-schema.usage.singularity.deffile.from: /path/to/workspace/Agent_training/tmp/tb2_docker_archives/alexgshaw_count-dataset-tokens_20251031.tar
```

检查依赖：

```bash
apptainer exec .../alexgshaw_count-dataset-tokens_20251031.sif python3 -c "import uvicorn, fastapi"
```

失败：

```text
ModuleNotFoundError: No module named 'uvicorn'
```

检查 `tmux`：

```bash
apptainer exec .../alexgshaw_count-dataset-tokens_20251031.sif which tmux
```

也失败，说明镜像里没有 `tmux`。

因此，预构建 Docker/SIF 成功解决的是“不要运行时 docker build / docker pull”；没有解决“每次 Harbor task container 启动时安装 Harbor server/tmux runtime deps”。

### 触发安装的位置

安装逻辑来自我们 glue 层的 `slime_plugins/harbor/bootstrap_singularity.sh`：

```bash
if ! "$_HARBOR_PY" -c "import uvicorn; import fastapi" 2>/dev/null; then
  echo "[harbor] Installing server dependencies (uvicorn/fastapi)..." >&2
  "$_HARBOR_PY" -m pip install --break-system-packages uvicorn fastapi ...
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "[harbor] Installing tmux..." >&2
  apt-get install -y -qq tmux ...
fi
```

这个 bootstrap 是 `SlimeSingularityEnvironment._start_server()` 每次启动 task container server 时复制到 staging 目录再执行的。启动命令包含：

```text
singularity exec ... --writable-tmpfs --fakeroot --containall --pid ... <task>.sif bash -c 'exec /staging/bootstrap.sh "$@"' ...
```

关键点是 `--writable-tmpfs`：pip/apt 安装写入的是本次 container 的临时可写层，不会回写到 `.sif`。所以同一个 SIF 的下一次 trial 仍然会发现缺 `uvicorn/fastapi/tmux`，然后再次安装。

### 具体卡在哪里

失败 trial `slime-r1-g15-s10-i250` 的 `result.json` 显示：

```text
task_name: terminal-bench/count-dataset-tokens
environment_setup.started_at: 2026-05-19T09:03:56Z
environment_setup.finished_at: 2026-05-19T09:23:57Z
exception_type: EnvironmentStartTimeoutError
exception_message: Environment start timed out after 600.0 seconds
```

外层 Harbor 逻辑在 `harbor/trial/trial.py`：

- `_environment_build_timeout_sec = task.environment.build_timeout_sec * timeout_multiplier`
- 对这个 task，`build_timeout_sec = 600.0`
- `_start_environment_with_retry()` 对 `EnvironmentStartTimeoutError` 最多重试 2 次

所以单个 trial 可以花接近 20 分钟：`600s * 2`，这正好匹配 `09:03:56Z -> 09:23:57Z`。

traceback 停在我们的 `_start_server()` health polling：

```text
slime_plugins/harbor/singularity_env.py line 221, in _start_server
    await asyncio.sleep(1)
```

这说明 subprocess 没有退出，但 FastAPI `/health` 一直没 ready。

launcher 日志里失败前最后的 server bootstrap 输出是：

```text
[server] Successfully installed ... uvicorn ... fastapi
[server] [harbor] Installing tmux...
```

后面没有：

```text
Slime Singularity FastAPI server is ready
```

因此本次最可疑的卡点是 `apt-get install -y -qq tmux` 或其后续 server exec 阶段。由于 stderr 被重定向、`apt-get` 用 `-qq`，它卡住时几乎没有更多日志。并发 16 个 trials 时，每个 trial 都可能同时执行 `apt-get update`、pip install、`apt-get install tmux`，这会放大代理、Debian apt 源、tmpfs、fakeroot、文件系统和 dpkg lock/CPU 竞争。

### 为什么 pre-build Docker 仍然不够

terminal-bench 的 task image 目标是“任务运行环境”，不是 Harbor runtime shim。Harbor 的 Singularity backend 需要在任务容器内启动一个 Python FastAPI server，让宿主通过 HTTP 调 `exec/copy/health` 等操作；Terminus2 还需要 tmux 维持交互 session。

这些是我们运行方式额外引入的依赖：

- `uvicorn`
- `fastapi`
- `pydantic-core` 等 Python wheel
- `tmux`
- 可能还包括 `ca-certificates/curl/asciinema` 等辅助工具

如果这些没有在 task Dockerfile 里，转换成 SIF 后也不会凭空存在。

### 修复方向

最干净的修复是在 SIF cache 层增加一个 Harbor runtime deps augmentation，不改 terminal-bench 源任务：

1. 以现有 task SIF 为 base，用 Apptainer definition 重新 build 一个 augmented SIF。
2. `%post` 里安装 `tmux` 和 `uvicorn/fastapi`。
3. 输出到新的 gy2 cache，例如 `harbor_sif_cache_tb2_full_harbor_runtime/`，不要覆盖同事原始 cache。
4. config 里把 `singularity_image_cache_dir` 指到新 cache。

这样每个 trial 启动时 bootstrap 的 dependency check 会直接通过，不再 pip/apt 安装。

备选方案是做运行时依赖 cache/overlay，例如按 Python 版本和 image id bind 一个持久 `site-packages` 目录，但要处理 Python 3.12/3.13 ABI、并发写入和 tmux apt 包，复杂度更高。

短期调参只能缓解，不能根治：

- 降低 `harbor_num_concurrent_trials` 到 4 或 8。
- 增加 `environment_build_timeout_multiplier`。
- 打开更详细 bootstrap 日志，去掉 apt/pip 的静默重定向。

但只要 SIF 里没有这些 runtime deps，反复安装就会继续发生。

### 已加的诊断补丁

已更新 `slime_plugins/harbor/bootstrap_singularity.sh`，给关键安装步骤加时间戳和耗时日志，并去掉 pip/apt 的 stderr 静默吞掉行为。下一次启动时会看到：

- `apt-get update starting/finished`
- `Installing server dependencies (uvicorn/fastapi)`
- `server dependencies install finished in Ns`
- `Running task setup.sh`
- `Installing tmux`
- `tmux install step finished in Ns`
- `starting Harbor environment server: ...`

这样如果仍然卡住，可以直接从日志判断卡在 apt update、pip install、task setup、tmux install，还是 server 启动后的 `/health` 阶段。`bash -n slime_plugins/harbor/bootstrap_singularity.sh` 已通过。

## 2026-05-20 Runtime SIF Augmentation Layer

### 目标

为避免每个 Terminus2 trial 在 `--writable-tmpfs` 临时层里反复安装 Harbor runtime 依赖，新增一个独立的 SIF augmentation layer：

- 保留同事预构建的原始 SIF cache：
  `/path/to/workspace/harbor_sif_cache_tb2_full`
- 新增增强后的 SIF cache：
  `/path/to/workspace/harbor_sif_cache_tb2_full_harbor_runtime`
- 增强 cache 使用相同 `.sif` 文件名，让 Harbor 原来的 docker image -> safe SIF filename lookup 不需要改。

### 新增/修改文件

- `examples/harbor_terminal_rl/augment_harbor_runtime_sifs.sh`
  - 以已有 `.sif` 为 `Bootstrap: localimage` base，重新 build 到增强 cache。
  - `%post` 安装 `ca-certificates`、`tmux`、`uvicorn`、`fastapi`。
  - 如果 base image 没有 `python3`，在 build 阶段安装 `python3`，然后用 `get-pip.py` bootstrap pip。
  - build 完后执行 validation：
    `python3 -c 'import fastapi, uvicorn'` 和 `tmux -V`。
  - 默认串行处理全部 89 个 SIF；可传单个文件名做单镜像验证。
- `examples/harbor_terminal_rl/harbor_terminal_bench2_full89_gy2_runtime_sif_last_turn.yaml`
  - 和 full89 config 一致，但 `singularity_image_cache_dir` 指向增强 cache。
- `examples/harbor_terminal_rl/run_full89_grpo_2epoch_n16.sh`
  - 默认 `CONFIG` 已切到 `harbor_terminal_bench2_full89_gy2_runtime_sif_last_turn.yaml`。
  - 如需回退原始 SIF cache，可以显式传 `CONFIG=.../harbor_terminal_bench2_full89_gy2_sif_last_turn.yaml`。
- `examples/harbor_terminal_rl/smoke_tb2_nop.py`
  - 对单个 TB2 task 用 Harbor `nop` agent 跑完整 trial，用于验证环境启动路径。
  - 只在 smoke 脚本内给 `importlib.metadata.version("harbor")` 做源码运行 fallback，避免要求 Harbor editable install。
- `slime_plugins/harbor/bootstrap_singularity.sh`
  - 进一步改成 apt lazy update：只有真的需要通过 apt 安装 `python3/python3-pip/tmux` 时才执行 `apt-get update`。
  - 对增强后的 SIF，bootstrap 应直接跳到 `starting Harbor environment server`，不再进行 apt/pip/tmux 安装。

### 已验证的单镜像

原始 `count-dataset-tokens` SIF：

```text
/path/to/workspace/harbor_sif_cache_tb2_full/alexgshaw_count-dataset-tokens_20251031.sif
```

原始镜像内缺 Harbor runtime deps：

```text
python3 -c "import uvicorn, fastapi" -> ModuleNotFoundError: uvicorn
which tmux -> not found
```

增强后镜像：

```text
/path/to/workspace/harbor_sif_cache_tb2_full_harbor_runtime/alexgshaw_count-dataset-tokens_20251031.sif
```

validation 通过：

```text
python runtime ok 0.47.0 0.136.1
/usr/bin/tmux
tmux 3.3a
```

第一次 build 尝试把 `python3-pip` 作为 apt 包安装，拉了大量 Python 3.11 依赖，速度慢且容易卡住。后续改为：

1. apt 只装 `ca-certificates`、`tmux`，必要时装 `python3`；
2. 优先复用 base image 里的 `python3`；
3. 只有缺 pip 时才用 `get-pip.py` bootstrap pip；
4. 再用 pip 安装 `uvicorn fastapi`。

这个路径在 `count-dataset-tokens` 上成功。

### Harbor NOP Smoke 结果

在 64 H20 job 的 0 号节点执行：

```bash
PYTHONPATH=/path/to/workspace/code/slime:/path/to/workspace/code/harbor/src:$PYTHONPATH \
python3 examples/harbor_terminal_rl/smoke_tb2_nop.py \
  --task-dir /path/to/workspace/code/terminal-bench-2-main/count-dataset-tokens \
  --sif-cache-dir /path/to/workspace/harbor_sif_cache_tb2_full_harbor_runtime \
  --trials-dir /tmp/slime-harbor-tb2-runtime-smoke-lazyapt
```

trial log 关键输出：

```text
Using cached Singularity image: /path/to/shared_storage/.../harbor_sif_cache_tb2_full_harbor_runtime/alexgshaw_count-dataset-tokens_20251031.sif
Starting Slime Singularity container with server on port 45715 (attempt 1/3)
[server] [harbor][2026-05-19T17:33:14+00:00] starting Harbor environment server: /staging/_hbexec.py --port 45715 --workdir /app
[server] INFO:     Started server process [105]
[server] INFO:     Application startup complete.
[server] INFO:     Uvicorn running on http://0.0.0.0:45715
Slime Singularity FastAPI server is ready
```

这里没有出现：

- `apt-get update starting`
- `Installing server dependencies`
- `Installing tmux`

说明 runtime layer 已经生效，Terminus2 environment startup 不再依赖运行时 apt/pip。

NOP trial 最终结果为 reward 0，这是预期的，因为 NOP agent 没写 `/app/answer.txt`：

```json
{
  "exception_info": null,
  "status": null,
  "verifier_result": {
    "rewards": {
      "reward": 0.0
    }
  }
}
```

### 额外发现

直接 `PYTHONPATH=/.../harbor/src python3 ...` 导入 Harbor 时会因为 `harbor.__init__` 调用 `importlib.metadata.version("harbor")` 而报：

```text
PackageNotFoundError: No package metadata was found for harbor
```

smoke 脚本内做了局部 fallback。尝试用 `uv run --frozen` 运行 Harbor 项目时，uv 会下载 standalone Python 3.13：

```text
Failed to download https://github.com/astral-sh/python-build-standalone/...
operation timed out
```

因此当前集群上不应把 `uv run` 放在 smoke/训练关键路径里。

### 当前批量增强任务

已在 64 H20 job 的 0 号节点启动：

```text
tmux session: harbor_sif_aug_all
log: /tmp/harbor_sif_aug_all.log
source cache count: 89
augmented cache count at start: 1
```

启动命令：

```bash
cd /path/to/workspace/code/slime
export APPTAINER_CACHEDIR=/tmp/slime-harbor-apptainer-cache
bash examples/harbor_terminal_rl/augment_harbor_runtime_sifs.sh 2>&1 | tee /tmp/harbor_sif_aug_all.log
```

截至 2026-05-20 01:39 CST，已完成 2 个增强 SIF；随后 01:42 CST 检查时已完成 3 个，正在 build 第 4 个：

```text
alexgshaw_count-dataset-tokens_20251031.sif
alexgshaw_adaptive-rejection-sampler_20251031.sif
alexgshaw_bn-fit-modify_20251031.sif
```

`adaptive-rejection-sampler` 的 base image 没有 `python3`，脚本在 build 阶段安装了 `python3`、bootstrap pip、安装 `uvicorn/fastapi` 并通过 validation。这说明增强脚本已覆盖“极简 task image”的情况。

### 2026-05-20 02:48 CST Recheck

再次检查时，首轮批量任务已经退出，增强 cache 中有 5 个 SIF。失败点是：

```text
alexgshaw_build-pmars_20251031.sif
```

失败原因来自 Debian/Trixie base image 的系统 Python 包管理：

```text
error: uninstall-no-record-file
Cannot uninstall typing_extensions 4.13.2
The package was installed by debian.

error: externally-managed-environment
```

也就是说 `pip install --break-system-packages uvicorn fastapi` 在升级 `typing_extensions` 时试图卸载 apt 管理的 `typing_extensions 4.13.2`，但该包没有 pip RECORD metadata，导致 build 中断。

已修改 `augment_harbor_runtime_sifs.sh` 的 pip 安装策略：

```bash
python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed uvicorn fastapi
```

这样 pip 不再尝试卸载 Debian 自带包，而是直接安装满足依赖的新 wheel。`bash -n` 已通过。

已重新启动批量增强：

```text
tmux session: harbor_sif_aug_all_retry
log: /tmp/harbor_sif_aug_all_retry.log
```

retry 启动后已确认会跳过已完成并验证通过的 SIF：

```text
[skip] alexgshaw_adaptive-rejection-sampler_20251031.sif already has Harbor runtime deps
[skip] alexgshaw_bn-fit-modify_20251031.sif already has Harbor runtime deps
[skip] alexgshaw_break-filter-js-from-html_20251031.sif already has Harbor runtime deps
[skip] alexgshaw_build-cython-ext_20251031.sif already has Harbor runtime deps
[build] alexgshaw_build-pmars_20251031.sif
```

后续等待 180s 再查，`build-pmars` 已通过，说明 `--ignore-installed` 修复有效。retry 日志显示后续 SIF 也在继续 build 和 validation：

```text
Successfully installed ... fastapi-0.136.1 ... typing-extensions-4.15.0 ... uvicorn-0.47.0
installed Harbor runtime deps 0.47.0 0.136.1
[done] /path/to/shared_storage/.../alexgshaw_caffe-cifar-10_20251031.sif
[build] alexgshaw_cancel-async-tasks_20251031.sif
```

截至 2026-05-20 02:54 CST：

```text
enhanced SIF count: 8 / 89
tmux session harbor_sif_aug_all_retry: running
grep ERROR/FATAL/Traceback/failed in retry log: no current matches
```

### 2026-05-20 Progress Check

再次检查 retry 任务：

```text
tmux session harbor_sif_aug_all_retry: running
enhanced SIF count: 62 / 89
current build: alexgshaw_protein-assembly_20251031.sif
latest completed: alexgshaw_portfolio-optimization_20251031.sif
```

最近日志持续出现 `[done]` 和下一条 `[build]`，说明增强流程在稳定推进。`grep ERROR/FATAL/Traceback/failed` 目前只匹配到 pip 的 dependency resolver 文本：

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

这条是 pip 的非致命 dependency conflict warning；对应镜像后续仍完成了 `Successfully installed ...`、validation 和 `[done]`，不是 build failure。

### 2026-05-20 Later Completion Check

再次检查时，`harbor_sif_aug_all_retry` 已经退出，但并未完成全部 89 个。进度停在：

```text
enhanced SIF count: 67 / 89
failed image: alexgshaw_qemu-alpine-ssh_20251031.sif
```

失败原因是该镜像使用 Python 3.9，常规 `https://bootstrap.pypa.io/get-pip.py` 不支持 Python 3.9。脚本先改为 Python <3.10 使用版本化 get-pip URL：

```text
https://bootstrap.pypa.io/pip/3.9/get-pip.py
```

随后又发现该 base image 缺 `distutils`：

```text
ModuleNotFoundError: No module named 'distutils.cmd'
```

已补充：

- `augment_harbor_runtime_sifs.sh`
  - Python <3.12 且缺 `distutils.cmd` 时，build 阶段 apt 安装 `python3-distutils`。
  - 增加 `TRUST_EXISTING=1`，用于 retry 时直接跳过已完成增强的 SIF，避免每次重新 `apptainer exec` validate 67 个已完成镜像。
- `bootstrap_singularity.sh`
  - 同步补 Python <3.12 的 `python3-distutils` fallback，避免极端情况下运行时 get-pip 遇到同类问题。

已启动：

```text
tmux session: harbor_sif_aug_all_retry4
log: /tmp/harbor_sif_aug_all_retry4.log
TRUST_EXISTING=1
```

`qemu-alpine-ssh` 已通过：

```text
Successfully installed ... fastapi-0.128.8 ... uvicorn-0.39.0
python runtime ok 0.39.0 0.128.8
/usr/bin/tmux
tmux 3.1c
[done] /path/to/shared_storage/.../alexgshaw_qemu-alpine-ssh_20251031.sif
```

截至当前检查：

```text
enhanced SIF count: 68 / 89
tmux session harbor_sif_aug_all_retry4: running
current build: alexgshaw_qemu-startup_20251031.sif
```

### 2026-05-20 Final Runtime SIF Completion

最终检查确认 runtime SIF augmentation 已完成：

```text
enhanced SIF count: 89 / 89
tmux session harbor_sif_aug_all_retry4: finished
log: /tmp/harbor_sif_aug_all_retry4.log
```

日志尾部确认最后一批镜像均完成 build、validation 和 `[done]`，最终输出：

```text
[done] /path/to/workspace/harbor_sif_cache_tb2_full_harbor_runtime/alexgshaw_write-compressor_20251031.sif
Augmented 89 image(s) into /path/to/workspace/harbor_sif_cache_tb2_full_harbor_runtime
```

最终可用于训练的 SIF cache：

```text
/path/to/workspace/harbor_sif_cache_tb2_full_harbor_runtime
```

`run_full89_grpo_2epoch_n16.sh` 默认已经指向 `harbor_terminal_bench2_full89_gy2_runtime_sif_last_turn.yaml`，也就是这个 runtime SIF cache。

## 2026-05-20 Full89 Training Restart

### Preflight

重新启动训练前确认：

```text
runtime SIF count: 89 / 89
default config: harbor_terminal_bench2_full89_gy2_runtime_sif_last_turn.yaml
launcher WANDB_API_KEY before setup: unset
```

为避免 W&B key 进入 bash xtrace 日志，已把：

```text
examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
```

开头的 `set -ex` 改为 `set -e`。随后在 launcher 的 `/tmp/slime_full89_wandb_env.sh` 写入本次运行用的 W&B env，权限 `600`，不放入 repo。

### Launch

使用 64 H20 job 的 8 个节点重新启动：

```text
worker tmux session: slime_full89_worker
launcher tmux session: slime_full89_launcher
launcher log: /tmp/slime_full89_launcher_runtime_sif.log
MASTER_ADDR: 10.0.0.4
RAY_NODE_IP_PREFIX: 29.68
```

启动结果：

```text
Ray cluster has 8/8 nodes
Total resources: 64 GPU
Ray job id: raysubmit_NhdCiAxM67Sawzzn
W&B: USE_WANDB=1, project=slime-harbor-terminal-rl, group=qwen35-9b-tb2-full89-grpo-n16-2epoch
```

Ray job entrypoint 已确认使用 runtime SIF config：

```text
--custom-config-path /path/to/workspace/code/slime/examples/harbor_terminal_rl/harbor_terminal_bench2_full89_gy2_runtime_sif_last_turn.yaml
```

关键训练参数：

```text
rollout_batch_size: 16
n_samples_per_prompt: 16
global_batch_size: 256
num_rollout: 12
rollout_max_response_len: 8096
rollout_max_context_len: 16384
```

### Failure: Megatron Train OOM

本次 run 已失败，不是 Harbor/Terminus 容器启动失败，也不是 JSON parser 失败。实际已经完成至少一个 rollout dump，并进入 Megatron actor train 后在 policy loss 的 entropy/logprob 计算阶段 OOM。

```text
Ray job id: raysubmit_SAGKephuEetLXBVB
W&B run: https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/bui0bqvd
failure time: 2026-05-20 10:17 CST
error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 30.55 GiB.
stack:
  slime/backends/megatron_utils/loss.py:policy_loss_function
  slime/utils/ppo_utils.py:calculate_log_probs_and_entropy
  slime/utils/ppo_utils.py:compute_entropy_from_logits
```

关键现象：

```text
rollout dump: /path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_grpo_n16_2epoch/rollout_dumps/1.pt
num_samples: 256
response_len min/max/mean: 52 / 2595 / 379.84
total_len min/max/mean/median: 2836 / 57219 / 20448.26 / 19238.5
total_len > 16k: 157 / 256
num response_len at 8096 cap: 0
```

结论：

- 不是 `ROLLOUT_MAX_RESPONSE_LEN=8096` 导致的 response 截断问题；response 实际都远小于 8096。
- OOM 主要来自 last-turn prompt 太长，训练侧 `tokens = prompt + response` 最大到 57k。
- slime 原生 `policy_loss_function` 在 `entropy-coef=0.00` 时仍然以 `with_entropy=True` 调 `get_log_probs_and_entropy`，会为 `[T, vocab]` logits 额外构造 entropy 图；这次 57k token 样本在 GPU 7 上额外申请 30.55 GiB 时失败。

### Detailed Failure Analysis

这次失败需要区分三个长度：

```text
trajectory length:
  Harbor/Terminus agent 在 terminal 环境里多轮交互产生的完整轨迹长度。

response_length:
  slime 训练样本里 response suffix 的长度，也是 loss_mask 必须对齐的长度。

total_length:
  slime 训练样本里的完整 tokens 长度，即 prompt + response。
```

我们当前使用 `harbor_sample_mode: last_turn`。这可以避免把整条 trajectory 的所有 assistant actions 都当成 response 来训练，但最后一轮 prompt 仍然可能很长，因为 prompt 里会包含 terminal history、之前命令、环境 observation、文件内容、错误输出等上下文。也就是说，`last_turn` 降低了 response side 的训练长度，但没有保证 prompt side 有界。

失败 dump 说明这一点很清楚：

```text
response_len max: 2595
total_len max: 57219
```

因此模型最后要学习的 response 并不长，但为了计算这个 response 的条件概率，训练 forward 仍然需要完整上下文：

```text
tokens = prompt + response
loss_mask 只覆盖 response suffix
model forward 仍然吃完整 tokens
```

这也是 terminal agent RL 和普通短 prompt RL 的差异。对 terminal agent 来说，长 trajectory 是合理现象；问题不是 agent 不该产生长轨迹，而是当前 slime/Megatron 训练路径把一条长轨迹展开成一条 dense LM sequence 来算。哪怕 loss 只在 response 上，attention forward、packed sequence、logits 和 logprob extraction 都会受 `total_length` 影响。

这次 OOM 的直接触发点在 policy loss：

```text
policy_loss_function
  -> get_log_probs_and_entropy(... with_entropy=True ...)
  -> calculate_log_probs_and_entropy
  -> compute_entropy_from_logits
```

更具体地说：

- Qwen3.5-9B 的 vocab 很大，policy logits 是 `[T, vocab_parallel_shard]` 级别的大张量。
- 当 `T` 接近 57k 时，logprob/entropy 阶段会对 logits 做 clone、归一化、softmax/entropy 相关中间张量。
- 原始 slime policy loss 即使 `--entropy-coef 0.00`，也仍然传 `with_entropy=True`，所以会构造一个实际不会贡献 loss 的 entropy 计算图。
- OOM 日志里 GPU 已占用约 `78.48 GiB`，空闲只有 `16.51 GiB`，而 entropy 阶段还要再申请 `30.55 GiB`，因此失败。

所以这次失败不是：

```text
不是 Harbor runtime SIF 问题
不是 apptainer/singularity 启动问题
不是 Terminus JSON parser 问题
不是 response 生成到了 8096 token 上限
```

而是：

```text
长 terminal history 进入训练样本 prompt
  -> total_length 过大
  -> Megatron actor train 对完整 sequence forward/loss
  -> policy loss 中 logprob/entropy 显存峰值过高
  -> CUDA OOM
```

### Current Mitigation Strategy

当前采用的是一个保守的粘合层解决方案，目标是先让 full89 GRPO 训练链路跑通，同时尽量不修改 slime/Harbor 原始核心逻辑。

第一层：训练前裁掉最早的 prompt prefix。

```text
HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768
```

逻辑：

```text
if len(tokens) > cap:
    tokens = tokens[-cap:]
```

但有一个重要约束：必须完整保留 response suffix。裁剪只发生在 prompt 前缀上，不改：

```text
response_length
loss_mask
reward
status
```

这个修复在：

```text
slime_plugins/harbor/sample_postprocess.py
  cap_total_tokens_in_samples
  cap_total_tokens_in_rollout_data
```

`cap_total_tokens_in_samples` 是主路径：在 rollout sample 转成 train_data 前裁剪。`cap_total_tokens_in_rollout_data` 是 backstop：如果有 debug/train-data 路径绕过 sample hook，actor 侧仍会再检查一次。

第二层：logprob 分块。

```text
LOG_PROBS_CHUNK_SIZE=512
```

这个使用 slime 已有能力，让 `calculate_log_probs_and_entropy` 不再一次性处理整段 `[T, vocab]`，降低 logprob 阶段的瞬时显存峰值。

第三层：当 entropy 系数为 0 时不算 entropy。

```text
USE_CUSTOM_NO_ENTROPY_LOSS=1
CUSTOM_LOSS_FUNCTION_PATH=slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled
```

这个 custom loss 和原始 policy loss 保持同样的 GRPO/PPO 主体逻辑，但在：

```text
entropy_coef == 0
```

时传：

```text
with_entropy=False
```

这样不会构造无用的 entropy 图，也避免这次 OOM 栈里最直接的显存峰值。

### Tradeoffs And Longer-Term Direction

当前方案是一个 pragmatic mitigation，不是最终最优的 terminal-agent RL memory design。

它的好处：

- 不改 slime/Megatron 核心训练代码。
- 不改 Harbor/Terminus 主体 agent 逻辑。
- 保留完整 response suffix，因此 reward、status、loss_mask 的 ATIF -> slime 训练语义不变。
- 能把超长 terminal history 控制到固定训练上限，避免单条样本拖垮 actor train。

它的代价：

- rollout 生成时模型可能看到了更早的 terminal history，但训练 logprob 计算时只保留最近 32k token；这会带来一定 context mismatch。
- 如果某个任务真的依赖很早以前的 observation，裁掉最早 prefix 可能影响这一条样本的精确信用分配。
- 32k 是当前显存和吞吐之间的经验折中，不代表最优值。后续可以根据新 run 的显存、速度、reward 分布再调成 24k、32k、48k 等。

长期更干净的方向包括：

```text
1. 在 agent/ATIF 层做 trajectory summarization 或 rolling memory，让 prompt 本身有界。
2. 在 Harbor adapter 层按 terminal turn/window 生成训练样本，而不是让最后一轮 prompt 无限膨胀。
3. 在 slime/Megatron loss 层做 response-position-only 的更细粒度 logprob extraction，避免对无 loss 的 prompt token 产生不必要的 logits/loss 中间量。
4. 使用更强的并行配置承载长 context，例如更合适的 TP/CP/sequence packing 策略。
5. 对 terminal task 按 estimated prompt length 做 dynamic batching，避免多个长样本落到同一训练 microbatch。
```

当前重启采用：

```text
cap32k + logprob chunk512 + no-entropy custom loss
```

这是为了先验证 full89、2 epoch、n=16 的端到端训练链路是否可行。如果它稳定跑过第一轮 actor train，再根据监控数据决定是否放宽 cap 或优化更底层的 loss/packing。

### Fix Added After OOM

为保持 slime/Harbor 主体逻辑完整，修复放在 Harbor 粘合层和脚本默认参数里：

```text
slime_plugins/harbor/rollout.py
  - Harbor rollout 现在支持 slime 现有的 --rollout-sample-filter-path 和 --rollout-all-samples-process-path hook。

slime_plugins/harbor/sample_postprocess.py
  - 新增 cap_total_tokens_in_samples：在 rollout sample 转 train_data 前裁 prompt 前缀。
  - 新增 cap_total_tokens_in_rollout_data：作为 actor 侧 backstop。
  - 裁剪只保留 suffix，完整保留 response；response_length、reward、status、loss_mask 不变。

slime_plugins/harbor/train_loss.py
  - 新增 policy_loss_no_entropy_when_disabled。
  - 当 entropy_coef == 0 时不构造 entropy 图，避免无意义的 [T, vocab] entropy 显存峰值。

examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
  - 支持 ROLLOUT_ALL_SAMPLES_PROCESS_PATH、ROLLOUT_DATA_POSTPROCESS_PATH。
  - 支持 USE_CUSTOM_NO_ENTROPY_LOSS 和 CUSTOM_LOSS_FUNCTION_PATH。
  - Ray runtime env 传递 HARBOR_MAX_TRAIN_TOTAL_TOKENS 和 PYTORCH_CUDA_ALLOC_CONF。

examples/harbor_terminal_rl/run_full89_grpo_2epoch_n16.sh
  - 默认 LOG_PROBS_CHUNK_SIZE=512。
  - 默认 HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768。
  - 默认启用 sample-level token cap、rollout_data backstop、自定义 no-entropy loss。
  - 默认输出目录改为 *_cap32k_chunk512_noentropy，避免覆盖失败 run。
```

本地静态校验：

```text
bash -n examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
bash -n examples/harbor_terminal_rl/run_full89_grpo_2epoch_n16.sh
PYTHONPYCACHEPREFIX=/tmp/slime_pycache_check python -m py_compile \
  slime_plugins/harbor/sample_postprocess.py \
  slime_plugins/harbor/train_loss.py \
  slime_plugins/harbor/rollout.py
sample cap fake-object test: sample cap ok
```

### Restart With OOM Fix

已用上述修复重新启动 full89 GRPO：

```text
worker tmux session: slime_full89_worker
worker log: /tmp/slime_full89_worker_cap32k_chunk512_noentropy.log
launcher tmux session: slime_full89_launcher
launcher log: /tmp/slime_full89_launcher_cap32k_chunk512_noentropy.log
monitor tmux session: slime_full89_monitor
monitor log: /tmp/slime_full89_monitor_cap32k_chunk512_noentropy.log
Ray job id: raysubmit_pA7eM2rPYNm9wiL5
W&B run: https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/fh3rea3i
```

Ray job entrypoint 已确认包含新保护参数：

```text
--rollout-all-samples-process-path slime_plugins.harbor.sample_postprocess.cap_total_tokens_in_samples
--loss-type custom_loss
--custom-loss-function-path slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled
--log-probs-chunk-size 512
--rollout-data-postprocess-path slime_plugins.harbor.sample_postprocess.cap_total_tokens_in_rollout_data
HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

当前状态：

```text
ray job status: RUNNING
Ray active nodes: 8 / 8
W&B online run initialized
SGLang router launched
SGLang engines starting across rollout nodes
checkpoint_file_count: 0
rollout_dumps: none yet at first monitor snapshot
```

## 2026-05-21 Progress: Second Failure And Backfill Fix

### What Happened

第二次 run 已经失败，但失败点和上一次不同：

```text
Ray job id: raysubmit_pA7eM2rPYNm9wiL5
W&B run: https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/fh3rea3i
launcher log: /tmp/slime_full89_launcher_cap32k_chunk512_noentropy.log
```

关键结论：

- OOM mitigation 生效了。第一轮 actor train 已完整跑完 `256/256` microbatch。
- 训练 step 1 有正常 metric：`train/loss=-0.016579...`，`train/entropy_loss=0.0`。
- 失败发生在下一轮 rollout 完成后、交给 slime 校验 batch 大小时。

最终报错：

```text
ValueError: Not enough samples 255 for global_batch_size 256
```

对应 slime 逻辑在 `slime/ray/rollout.py::_get_rollout_data`：

```text
if len(data) % global_batch_size != 0:
    trim_len = (len(data) // global_batch_size) * global_batch_size
    if trim_len == 0:
        raise ValueError(...)
```

这次第二轮 rollout 里 256 个 Terminus/Harbor trial 有 1 个没有产出可训练 sample，最终 flatten 后只有 255 条。因为 `global_batch_size=256`，slime 无法裁出一个完整 batch，于是直接失败。

日志里能看到相关 timeout：

```text
Trial slime-r2-g14-s6-i486 failed: Verifier execution timed out after 3600.0 seconds
```

以及 token cap 已在失败前正常执行：

```text
Harbor sample token cap trimmed 59/255 samples, cap=32768, max_total_before=88124, max_total_after=32768
```

因此这不是上一轮的 actor train OOM，而是 Harbor rollout 返回样本数不满足 slime batch contract。

### Why 255 Is Especially Bad For GRPO

当前配置：

```text
rollout_batch_size=16
n_samples_per_prompt=16
global_batch_size=256
harbor_sample_mode=last_turn
```

也就是说 slime/GRPO 期望每轮 rollout 是 16 个 prompt group，每个 group 16 条 response。失败 1 条后变成 255，不仅 batch size 不够，reward normalization 也会失去固定的 `[16, 16]` group 结构。

直接把 `global_batch_size` 降到 128 可以绕过这个错误，但会在 255 条时丢掉 127 条样本；开启 dynamic global batch 也会让 reward normalization 退化成不理想的非固定 group。更稳妥的做法是在 Harbor rollout 插件里保证返回完整 group。

### Fix Added

修改仍然只放在粘合层，不改 Harbor 和 slime 原始核心逻辑：

```text
slime_plugins/harbor/rollout.py
```

新增行为：

- 对 `last_turn` / `trajectory` 且 `n_samples_per_prompt > 1` 的 rollout，默认要求完整 prompt group。
- 如果某个 group 少于 `n_samples_per_prompt` 条 sample，就丢掉这个不完整 group。
- 从 `HarborTaskDataSource` 继续取新的 task group 做 backfill，直到凑够 `rollout_batch_size` 个完整 group。
- 默认最多尝试 `rollout_batch_size * harbor_rollout_group_attempt_multiplier` 个 group。
- 保留 metric，便于后续看失败率：
  - `harbor/num_group_attempts`
  - `harbor/num_incomplete_groups_dropped`
  - `harbor/num_converted_samples`
  - `harbor/num_overflow_samples_trimmed`

配置显式写入：

```text
examples/harbor_terminal_rl/harbor_terminal_bench2_full89_gy2_runtime_sif_last_turn.yaml

harbor_require_full_rollout_groups: true
harbor_rollout_group_attempt_multiplier: 4
harbor_retry_missing_samples: 0
```

`harbor_retry_missing_samples=0` 是故意的。当前失败来自 verifier 3600s timeout，重试同一个 sample 很可能再等一小时；更合理的是丢掉这个不完整 group，换一个新 task group backfill。

新的默认输出目录也改为 backfill 后缀，避免误读前一个失败 run：

```text
BASE_FOLDER=.../slime_harbor_qwen35_9b_tb2_full89_grpo_n16_2epoch_cap32k_chunk512_noentropy_backfill
SAVE_SUBDIR=Qwen3.5-9B_terminal_bench2_full89_grpo_n16_2epoch_cap32k_chunk512_noentropy_backfill
WANDB_GROUP=qwen35-9b-tb2-full89-grpo-n16-2epoch-cap32k-chunk512-noentropy-backfill
```

### Validation

远程节点环境静态校验通过：

```text
PYTHONPYCACHEPREFIX=/tmp/slime_pycache_check python -m py_compile \
  slime_plugins/harbor/rollout.py \
  slime_plugins/harbor/sample_postprocess.py \
  slime_plugins/harbor/train_loss.py
```

本地脚本语法校验通过：

```text
bash -n examples/harbor_terminal_rl/run_full89_grpo_2epoch_n16.sh
bash -n examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
```

远程 fake rollout backfill 测试通过：构造 2 个 group、每组 4 条，其中 1 条返回空；插件丢掉 3/4 的不完整 group，并 backfill 新 group，最终返回 `[4, 4]`。

测试输出：

```text
Dropping incomplete Harbor rollout group with 3/4 samples; will backfill another group if attempts remain.
{'harbor/num_groups': 2, 'harbor/num_samples': 8, 'harbor/num_converted_samples': 11,
 'harbor/num_failed_trials': 0, 'harbor/num_missing_rollout_details': 1,
 'harbor/num_group_attempts': 3, 'harbor/num_incomplete_groups_dropped': 1,
 'harbor/num_overflow_samples_trimmed': 0, 'harbor/reward_mean': 1.0}
```

### Restart With Backfill Fix

已重新启动 full89、2 epoch、n=16 训练：

```text
Ray job id: raysubmit_ym3HN7NixzLiZjEK
W&B run: https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/vy1x8ct8
launcher tmux: slime_full89_launcher
worker tmux: slime_full89_worker
monitor tmux: slime_full89_monitor
launcher log: /tmp/slime_full89_launcher_cap32k_chunk512_noentropy_backfill.log
monitor log: /tmp/slime_full89_monitor_cap32k_chunk512_noentropy_backfill.log
```

新的 run 使用：

```text
BASE_FOLDER=/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_grpo_n16_2epoch_cap32k_chunk512_noentropy_backfill
SAVE_SUBDIR=Qwen3.5-9B_terminal_bench2_full89_grpo_n16_2epoch_cap32k_chunk512_noentropy_backfill
```

启动后首个状态：

```text
ray job status: RUNNING
Ray active nodes: 8 / 8
W&B online run initialized
SGLang router launched at 10.0.0.4:3816
SGLang engines launching across 56 rollout GPUs
checkpoint_file_count: 0
rollout_dumps: none yet at first monitor snapshot
```

监控 session `slime_full89_monitor` 已按 30 分钟间隔写快照，首个快照时间为 `2026-05-21T00:17:49+08:00`。

### 2026-05-21 05:13 Status

当前 backfill run 仍在正常运行：

```text
Ray job id: raysubmit_ym3HN7NixzLiZjEK
status: RUNNING
Ray active nodes: 8 / 8
rollout dumps: 1.pt, 2.pt
checkpoint_file_count: 0
```

关键进展：

- step 1 actor train 已完成，`256/256` microbatch 全部跑完。
- step 1 train metric 正常，未再出现 OOM：

```text
train/loss: 0.015537279812794812
train/entropy_loss: 0.0
actor_train_time: 3290.56s
log_probs_time: 2628.33s
actor_train_tok_per_s: 1549.25
```

- rollout 2 已成功保存：

```text
rollout dump: .../rollout_dumps/2.pt
harbor/num_groups: 16
harbor/num_samples: 256
harbor/num_converted_samples: 256
harbor/num_group_attempts: 16
harbor/num_incomplete_groups_dropped: 0
harbor/reward_mean: 0.0390625
rollout/response_len/mean: 500.90
rollout/response_len/max: 3080
rollout/truncated_ratio: 0.0234375
```

这说明上次的 `Not enough samples 255 for global_batch_size 256` 问题目前没有复现。即使日志中仍可见少量 task/verifier timeout，rollout 2 最终仍交给 slime 一个完整 256-sample batch。

当前仍无 checkpoint 是预期的：当前默认 `SAVE_INTERVAL=4`，需要完成第 4 个训练 step 才会保存。
