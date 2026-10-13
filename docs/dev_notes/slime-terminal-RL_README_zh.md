# slime-terminal-RL

本仓库是基于 slime v0.3 的 Terminal Agent RL 分支，集成了 Harbor、Daytona、
SGLang 和 Megatron。当前主要用途是复现 **Qwen3.5-9B 在 Terminal-Bench 2
train-v2 上的 32-GPU PPO overfit 实验**。

本文给出从一组干净 Taiji 节点开始运行该实验的完整步骤。这里的“干净节点”
是指已经安装 GPU 训练栈的 Taiji 镜像，而不是裸 Ubuntu 服务器。Harbor 和
Daytona 的 Python 依赖可以由本仓库脚本安装，但 CUDA、Megatron、SGLang、
Ray 和 Apptainer/Singularity 必须由基础镜像提供。

已验证的参考实验在同一批 89 个任务上，将 Qwen3.5-9B 从 baseline 的
`18/89` 提升到训练后的 `23/89`。结果只在相同模型、任务包、agent 配置和
evaluation protocol 下可直接比较。

## 参考实验配置

| 项目 | 设置 |
| --- | --- |
| 模型 | Qwen3.5-9B |
| 资源 | 4 节点 x 8 GPU，共 32 GPU |
| 资源划分 | 1 个 actor/train 节点 + 3 个 rollout 节点 |
| 算法 | PPO，actor 和 critic 共用训练 placement group |
| rollout 模式 | `standard` async，不使用 `fully_async` |
| 每步采样 | 8 prompts x 4 trajectories = 32 trajectories |
| oversampling | 发出 9 个 prompt groups，接收前 8 个完整 groups |
| 训练长度 | 56 rollout steps，约 5 epochs x 89 tasks |
| 长度上限 | rollout context 32k，训练 token cap 32k |
| 单 GPU 动态 batch 上限 | `MAX_TOKENS_PER_GPU=24576` |
| actor / critic LR | `5e-7` / `5e-6` |
| reward | terminal checker reward + CTRF progress reward |
| eval | 训练前一次，之后每 10 steps 一次 |
| checkpoint | 每 10 steps 一次，actor/critic 分目录保存 |
| Daytona storage | 10240 MB / sandbox |
| 容错 | `USE_FAULT_TOLERANCE=1` |

虽然主 wrapper 的文件名中仍有 `fully_async`，默认的
`HARBOR_ROLLOUT_MODE=standard` 才是当前正确配置。文件名仅为保留历史调用兼容。
`fully_async + oversampling` 会使快任务更容易进入训练 batch，产生 completion
time 相关的 survivorship bias，因此 overfit 复现实验不要改回 `fully_async`。

### Reward 语义

YAML 中的 `harbor_reward_key: reward` 是 Terminal-Bench verifier 给出的终局成功
信号，通常为 0/1。训练 wrapper 另外启用：

```text
slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards
```

成功 trajectory 保持 reward 1；失败 trajectory 若产生 `verifier/ctrf.json`，则
使用测试通过比例构造 progress reward；没有有效 CTRF 的失败 trajectory 仍为 0。
同一 trajectory 派生出的训练片段共享该 shaped reward；PPO advantage 仍由
critic/GAE 路径计算，不做 GRPO 式的组内 reward normalization。eval 不运行训练
用的 reward post-process，报告的仍是 terminal verifier pass rate。

## 目录和关键文件

```text
examples/harbor_terminal_rl/
├── start_tb2_train_v2_32gpu_9b_daytona_ppo_n4_bringup_tmux.sh
├── run_tb2_train_v2_32gpu_9b_daytona_ppo_n4_bringup_fully_async.sh
├── run_qwen35_9b_terminal_bench2_64h20.sh
├── harbor_terminal_bench2_train_v2_local_32gpu_9b_daytona_ppo_n4_bringup.yaml
├── megatron_roles_tb2v2_9b_criticlr10x.yaml
├── prepare_tb2_train_v2_local_tasks.sh
├── install_harbor_runtime_deps.sh
├── monitor_tb2v2_9b_ppo_n4_32k_job.sh
└── daytona_cleanup_sessions_by_label.py
```

启动脚本只负责本实验。更通用的 slime 文档仍可参考
[upstream 中文 README](README_zh.md) 和 [docs](docs/)。

## 1. 准备四个节点

四个节点应使用相同镜像，并满足：

- 每个节点有 8 张可见 GPU，合计 32 张；
- Python 环境中已有 PyTorch、Ray、SGLang 0.5.12 和 slime v0.3 对应依赖；
- `/root/Megatron-LM` 存在；
- `tmux`、`git`、`rsync`、`curl` 和 `apptainer` 或 `singularity` 可用；
- 四个节点可互访 Ray/NCCL 端口；
- 代码、模型和输出目录所在共享文件系统在四个节点路径一致；
- Daytona 配额至少能承载 36 个并发 sandbox；
- 当前 run 使用独立的 Daytona API key，不与其他并行实验共用。

在每个节点执行基础检查：

```bash
nvidia-smi -L | wc -l                 # 期望为 8
test -d /root/Megatron-LM
command -v ray
command -v tmux
command -v apptainer || command -v singularity
ulimit -Hn                            # 通常应接近 1048576
```

launcher 会在启动 Ray 之前将 soft `nofile` 提升到 hard limit。低于 1024 的
`nofile` 会在高并发 trial 下导致 socket/future 无法回传，表现为 rollout 长时间
停在 collected 0。

## 2. 获取代码

建议保持如下目录布局，因为脚本默认按该布局寻找 slime 和 Harbor：

```bash
export CODE_ROOT=/path/to/workspace/code
mkdir -p "${CODE_ROOT}"

git clone \
  git@git.woa.com:TencentAILab/nlp/research/self_improving/slime-terminal-RL.git \
  "${CODE_ROOT}/slime-v0.3-harbor"

git clone \
  git@git.woa.com:TencentAILab/nlp/research/self_improving/harbor.git \
  "${CODE_ROOT}/harbor"
```

参考实验使用的 Harbor revision 是 `4e66476ef01975364f82bb46fc509bea2c0fb7cf`。
需要严格复现时固定该版本：

```bash
git -C "${CODE_ROOT}/harbor" switch --detach \
  4e66476ef01975364f82bb46fc509bea2c0fb7cf
git -C "${CODE_ROOT}/slime-v0.3-harbor" rev-parse HEAD
```

第二条命令的输出应记录到实验日志中。共享文件系统场景只需 clone 一次；如果节点
之间不共享代码目录，则必须在每个节点部署完全相同的 revision。

## 3. 准备凭证

不要把任何 key 写入仓库、启动脚本或 shell history。在每个节点准备同一路径的
权限为 `0600` 的环境文件，例如：

```bash
touch ${HOME}/.daytona_harbor.env
chmod 600 ${HOME}/.daytona_harbor.env
```

文件内容格式如下，值由运行者在节点上填写：

```bash
export DAYTONA_API_KEY="<unique-key-for-this-run>"
export WANDB_API_KEY="<wandb-key>"
```

同一 Daytona key 不要同时用于两个高并发 run。只需要 offline W&B 时，可不填
W&B key，并在启动环境中设置 `WANDB_MODE=offline`。

验证凭证能被当前 Python 环境读取，但不要打印 key 本身：

```bash
set -a
source ${HOME}/.daytona_harbor.env
set +a
python3 -c 'import os; assert os.environ.get("DAYTONA_API_KEY"); print("daytona key: set")'
python3 -c 'import os; assert os.environ.get("WANDB_API_KEY"); print("wandb key: set")'
```

## 4. 准备模型

当前复现配置需要两份 Qwen3.5-9B 权重：

```bash
export HF_CKPT=/path/to/models/Qwen3.5-9B
export PRETRAINED_CKPT=/path/to/models/Qwen3.5-9B_torch_dist

test -d "${HF_CKPT}"
test -f "${HF_CKPT}/config.json"
test -d "${PRETRAINED_CKPT}"
```

`HF_CKPT` 用于 tokenizer、SGLang rollout 和权重同步；`PRETRAINED_CKPT` 是
Megatron 可加载的 torch distributed checkpoint。当前参考配方默认使用两者，
不会自动将 HF 权重转换为 torch_dist。若共享目录中没有 torch_dist 权重，需要先
使用与当前 Qwen3.5 Megatron spec 匹配的转换流程生成并在小规模 load test 中验证。

四个节点必须看到相同路径。共享模型 I/O 较慢时可设置
`STAGE_HF_CKPT_TO_TMP=1` 和 `STAGE_TORCH_DIST_CKPT_TO_TMP=1`，让 wrapper 在各
节点将权重复制到本地 `/tmp`；默认关闭，首次复现建议保持默认。

## 5. 准备 Terminal-Bench 任务

本实验必须使用给定 tar 中的 89 个 train-v2 tasks：

```text
/path/to/workspace/terminal-bench-2-train-v2.tar.gz
```

不要直接从共享 tar 内运行，也不要改用其他 checkout。必须在 **每个节点** 将 tar
复制到节点本地盘再解压：

```bash
cd "${CODE_ROOT}/slime-v0.3-harbor"

TB2_TRAIN_V2_TAR_PATH=/path/to/workspace/terminal-bench-2-train-v2.tar.gz \
TB2_TRAIN_V2_DEST_ROOT=/tmp/slime-harbor-tasks \
bash examples/harbor_terminal_rl/prepare_tb2_train_v2_local_tasks.sh

find /tmp/slime-harbor-tasks/terminal-bench-2-train-v2 \
  -mindepth 2 -maxdepth 2 -name task.toml | wc -l
```

最后一条命令必须在每个节点返回 `89`。准备脚本带完整性 marker，可安全重复执行；
task 数不对时会直接失败。

## 6. 安装运行时依赖

在四个节点分别执行：

```bash
cd "${CODE_ROOT}/slime-v0.3-harbor"
INSTALL_DAYTONA=1 \
bash examples/harbor_terminal_rl/install_harbor_runtime_deps.sh

PYTHONPATH="${CODE_ROOT}/harbor/src:${CODE_ROOT}/slime-v0.3-harbor" \
python3 -c 'import daytona, tenacity, toml, slime_plugins.harbor; print("runtime imports: ok")'
```

tmux launcher 默认也会执行安装脚本，但建议提前安装和检查，避免某个 worker 在 Ray
集群启动后才因缺少 `tenacity`、Daytona SDK 或 Harbor 依赖退出。该脚本不会安装
CUDA、PyTorch、Megatron、Ray、SGLang 或 Apptainer。

## 7. 网络与 proxy

W&B、Daytona 和镜像 registry 可能需要 `star-proxy`，而 Ray、SGLang 控制面和
NCCL 的节点间流量必须直连。基础 launcher 会在四个 Ray 节点加入后，将实际节点
IP 动态追加到 `NO_PROXY/no_proxy` 并传入 Ray runtime env。

启动前在每个节点确认训练网卡地址，并设置能匹配这些地址的前缀：

```bash
hostname -I
export RAY_NODE_IP_PREFIX="<comma-separated-prefixes>"  # 例如 29.68
```

不要设置会强制接管所有流量的 `ALL_PROXY`。如果覆盖了 `NO_PROXY`，至少保留
`localhost,127.0.0.1` 和四个训练节点 IP。日志中出现 Squid HTML、503 或 504
通常表示内网请求误走 proxy；直连节点端口出现 `[Errno 111] Connection refused`
则通常是对应 SGLang engine 已退出，不是 proxy 问题。

## 8. 启动训练

选 node 0 作为 launcher。先在四个节点设置完全相同的 run 变量；只有
`MASTER_ADDR` 指向 node 0：

```bash
export CODE_ROOT=/path/to/workspace/code
export SLIME_DIR="${CODE_ROOT}/slime-v0.3-harbor"
export HARBOR_SRC="${CODE_ROOT}/harbor/src"

export RUN_TAG="q35_9b_ppo_overfit_$(date +%Y%m%d_%H%M%S)"
export MASTER_ADDR="<node-0-training-ip>"
export RAY_NODE_IP_PREFIX="<comma-separated-prefixes>"
export DAYTONA_ENV_FILE=${HOME}/.daytona_harbor.env

export BASE_FOLDER="/path/to/workspace/runs/${RUN_TAG}"
export SAVE_SUBDIR="Qwen3.5-9B_${RUN_TAG}"
export RUN_TRIALS_DIR="/path/to/workspace/trials/${RUN_TAG}"

export HF_CKPT=/path/to/models/Qwen3.5-9B
export PRETRAINED_CKPT=/path/to/models/Qwen3.5-9B_torch_dist
```

确保 `RUN_TAG` 在四个节点完全一致。不要在每个节点分别执行 `date` 后再复制变量；
应在 node 0 生成一次并原样传给其他节点。

在 node 0 启动 launcher：

```bash
cd "${SLIME_DIR}"
bash examples/harbor_terminal_rl/start_tb2_train_v2_32gpu_9b_daytona_ppo_n4_bringup_tmux.sh \
  launcher "${MASTER_ADDR}"
```

紧接着在 node 1、2、3 启动 worker：

```bash
cd "${SLIME_DIR}"
bash examples/harbor_terminal_rl/start_tb2_train_v2_32gpu_9b_daytona_ppo_n4_bringup_tmux.sh \
  worker "${MASTER_ADDR}"
```

worker 会自动重试连接 Ray head，launcher 最多等待 900 秒收齐 4 个节点。脚本会在
每个节点创建 tmux session：

- node 0：`slime_tb2v2_9b_ppo_n4_launcher`
- node 1-3：`slime_tb2v2_9b_ppo_n4_worker`

启动脚本已固定参考实验参数。正常复现时不要覆盖下列变量：

```text
HARBOR_ROLLOUT_MODE=standard
ROLLOUT_BATCH_SIZE=8
OVER_SAMPLING_BATCH_SIZE=9
N_SAMPLES_PER_PROMPT=4
GLOBAL_BATCH_SIZE=32
NUM_ROLLOUT=56
SEQ_LENGTH=32768
HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768
MAX_TOKENS_PER_GPU=24576
SAVE_INTERVAL=10
EVAL_INTERVAL=10
SKIP_EVAL_BEFORE_TRAIN=0
USE_FAULT_TOLERANCE=1
USE_WANDB=1
```

## 9. 确认正常启动

node 0 上检查：

```bash
tmux ls
ray status
tail -f "/tmp/slime_tb2v2_9b_ppo_n4_${RUN_TAG}_launcher.log"
```

每个 worker 上检查：

```bash
tmux ls
tail -f "/tmp/slime_tb2v2_9b_ppo_n4_${RUN_TAG}_worker.log"
```

健康启动应满足：

1. `ray status` 显示 4 个 active nodes、32 张 GPU，且没有长期 pending node；
2. actor/train 使用 node 0 的 8 张 GPU，24 个单 GPU SGLang rollout engines 分布
   在另外 3 个节点；
3. 日志中打印 `mode=standard`、`num_rollout=56`、`USE_WANDB=1`；
4. W&B 中能看到 project `slime-harbor-terminal-rl` 的新 run；
5. 因 `SKIP_EVAL_BEFORE_TRAIN=0`，训练前先出现 `eval/tb2`；
6. rollout step 0 只 warm up critic，step 1 起 actor 和 critic 都训练；
7. `${BASE_FOLDER}/rollout_dumps/0.pt` 开始出现，trial 目录持续增加。

不要仅以 tmux session 存在判断训练健康；必须同时检查 train process、Ray 状态、
rollout dump 更新时间和 launcher 日志中的 rollout step。

## 10. 定时监控

可以在能调用 `jizhi_client`/`exec_multinode.exp` 的登录节点启动仓库内监控器。必须
显式覆盖任务和 run 信息，不要使用脚本中的历史默认值：

```bash
cd "${CODE_ROOT}/slime-v0.3-harbor"

SERVER_NAME="<taiji-server-name>" \
INSTANCE_ID="<taiji-instance-id>" \
REMOTE_HOST_INDEX=0 \
RUN_TAG="${RUN_TAG}" \
RUN_BASE="${BASE_FOLDER}" \
SAVE_DIR="${BASE_FOLDER}/${SAVE_SUBDIR}" \
TRIALS_DIR="${RUN_TRIALS_DIR}" \
DAYTONA_CLEAN_CREATED_AFTER="<run-start-UTC-ISO-time>" \
INTERVAL_SECONDS=3600 \
nohup examples/harbor_terminal_rl/monitor_tb2v2_9b_ppo_n4_32k_job.sh --loop \
  >/tmp/slime_tb2v2_9b_monitor.launch.log 2>&1 &
```

监控输出默认写入：

```text
/tmp/slime_tb2v2_9b_ppo_n4_32k_monitor.log
/tmp/slime_tb2v2_9b_ppo_n4_32k_monitor.nohup.log
/tmp/slime_tb2v2_9b_ppo_n4_32k_monitor.pid
```

监控器只在 train process 消失、未完成 56 steps 且日志命中明确 fatal signature 时
触发 failguard。清理时它从本 run 的 `TRIALS_DIR` 生成精确 Harbor session 列表，
先 audit，再按 `harbor_managed=1, harbor_session=<exact-session>` 删除。不要把它改成
扫描并删除所有 live Daytona sandboxes。

## 11. 输出与 checkpoint

主要产物：

```text
${BASE_FOLDER}/
├── config/                         # 本 run 的 Harbor YAML
├── rollout_dumps/<step>.pt         # 每步 rollout 数据
├── wandb/                          # W&B 本地状态
└── ${SAVE_SUBDIR}/                 # actor checkpoints

${BASE_FOLDER}/${SAVE_SUBDIR}_critic/  # critic checkpoints
${RUN_TRIALS_DIR}/                     # Harbor trial/verifier artifacts
```

`SAVE_INTERVAL=10` 通常会产生 `iter_0000009`、`iter_0000019` 等目录，训练正常结束
时还应保存最终 iteration。actor 与 critic 必须始终是两个 sibling 目录：

```bash
export SAVE_DIR="${BASE_FOLDER}/${SAVE_SUBDIR}"
find "${SAVE_DIR}" -maxdepth 1 -type d -name 'iter_*' | sort
find "${SAVE_DIR}_critic" -maxdepth 1 -type d -name 'iter_*' | sort
```

代码会在 critic 分配前断言 actor/critic save path 不同。检查 checkpoint tensor 时，
Qwen3.5-9B actor 的 output layer 应对应 vocab head（逻辑 shape
`[248320, 4096]`），critic 应为 value head（逻辑 shape `[1, 4096]`）。如果 actor
目录只剩 value head，说明使用了未包含 checkpoint 隔离修复的旧代码，不能用于评测。

## 12. 停止和 Daytona 清理

优先在 node 0 中断 launcher，让退出 trap 写 worker stop marker：

```bash
tmux send-keys -t slime_tb2v2_9b_ppo_n4_launcher C-c
```

确认训练进程退出后，再检查三个 worker 是否收到 stop marker。若训练异常退出并遗留
sandbox，只能按本 run 的 trial session 精确清理。先生成 session 列表并 audit：

```bash
export SESSION_FILE="/tmp/${RUN_TAG}_harbor_sessions.txt"
find "${RUN_TRIALS_DIR}" -maxdepth 1 -type d -name 'slime-r*' -printf '%f\n' \
  | sort -u > "${SESSION_FILE}"

set -a
source "${DAYTONA_ENV_FILE}"
source "${SLIME_DIR}/examples/harbor_terminal_rl/registry_proxy_env.sh"
set +a

python3 "${SLIME_DIR}/examples/harbor_terminal_rl/daytona_cleanup_sessions_by_label.py" \
  --session-file "${SESSION_FILE}" --action audit
```

确认候选项全部属于当前 run 后再删除：

```bash
python3 "${SLIME_DIR}/examples/harbor_terminal_rl/daytona_cleanup_sessions_by_label.py" \
  --session-file "${SESSION_FILE}" --action delete
```

不要使用无 label、无 session list 的全局 Daytona 清理命令。其他 live sandboxes 可能
属于并行实验，不能触碰。

## 常见问题

### Ray 只有 1-3 个节点

检查 worker log、`MASTER_ADDR`、`RAY_NODE_IP_PREFIX` 和节点间端口。四个节点必须使用
同一个 `RUN_TAG`、Ray port 范围和 shared code revision。

### `ModuleNotFoundError: tenacity` 或 Daytona import 失败

在报错节点重新运行 `install_harbor_runtime_deps.sh`。不要只在 launcher 安装依赖，
Harbor trial worker 会在 rollout 节点 import 同样的模块。

### 发现 0 个 task 或 task path 不存在

`/tmp` 不共享。必须在出错节点重新执行 `prepare_tb2_train_v2_local_tasks.sh`，并确认
`task.toml` 数量严格为 89。

### W&B 页面看不到 run

确认 `USE_WANDB=1`、`WANDB_MODE=online`，且环境文件中的 `WANDB_API_KEY` 能被
launcher 读取。查看 launcher log 是否有 `No API key configured` 或网络错误。

### SGLang engine 退出或权重同步卡住

`USE_FAULT_TOLERANCE=1` 会检测 engine health，并在下一次 `update_weights` 时恢复
异常 engine。若日志是直连 `[Errno 111]`，优先检查对应节点的 CUDA OOM、CUBLAS、
SIGQUIT 和 SGLang engine 日志；若是 Squid 503/504，再检查 `NO_PROXY`。

### 32k 配置 OOM

先降低 `MAX_TOKENS_PER_GPU`，例如从 `24576` 调到 `20480`，让 dynamic batching
切出更小 micro-batch。不要把 `SEQ_LENGTH` 或
`HARBOR_MAX_TRAIN_TOTAL_TOKENS` 偷改成 16k，否则已经不是同一个 32k 实验。

### checkpoint 只有 critic，没有 actor

确认 actor 保存到 `${SAVE_DIR}`、critic 保存到 `${SAVE_DIR}_critic`，并运行回归测试：

```bash
cd "${SLIME_DIR}"
python3 -m pytest -q tests/test_critic_checkpoint_isolation.py
```

当前代码在启动 critic 前会对路径冲突 fail fast；不要绕过该断言。

## 复现检查清单

- [ ] 4 x 8 GPU 可见，Ray 最终看到 4 个 active nodes；
- [ ] 四个节点使用相同 slime/Harbor revision；
- [ ] HF 和 torch_dist 模型在四个节点可读；
- [ ] 89 个 task 已在四个节点本地 `/tmp` 解压；
- [ ] 本 run 使用独立 Daytona key，W&B key 已通过安全 env file 注入；
- [ ] `HARBOR_ROLLOUT_MODE=standard`，不是 `fully_async`；
- [ ] 56 steps、32k token cap、`MAX_TOKENS_PER_GPU=24576`；
- [ ] n=4、batch 8、oversampling 9、global batch 32；
- [ ] actor LR `5e-7`，critic LR `5e-6`；
- [ ] process reward、eval 和 fault tolerance 已开启；
- [ ] 每 10 steps 保存，actor/critic checkpoint 目录隔离；
- [ ] 监控器按小时检查，失败时只清理本 run 的 Daytona sessions。

## 相关文档

- [Process reward 设计](examples/harbor_terminal_rl/PARTIAL_REWARD_PROGRESS_SCORE.md)
- [Terminal-Bench eval 集成](examples/harbor_terminal_rl/EVAL_TB2_INTEGRATION.md)
- [多节点 trial 派发](examples/harbor_terminal_rl/MULTI_NODE_TRIAL_DISPATCH.md)
- [Harbor rollout 长尾与 async 记录](examples/harbor_terminal_rl/HARBOR_ROLLOUT_LONG_TAIL_ASYNC_NOTES.md)
- [slime 上游使用文档](docs/en/get_started/usage.md)
