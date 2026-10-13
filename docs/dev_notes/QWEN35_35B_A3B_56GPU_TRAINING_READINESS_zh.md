# Qwen3.5-35B-A3B 在 56 卡集群上的训练就绪状态与待办清单

> 目标模型：`/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B`
> 目标集群：当前 Ray 集群 **7 节点 × 8 H20 = 56 GPU**（其中 **32 卡训练** / 24 卡 rollout）
> 生成日期：2026-07-23
> 相关脚本：`examples/harbor_terminal_rl/run_35b_a3b_56gpu_daytona_actor4_rollout3_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.sh`（下称 **wrapper**）+ 其 tmux launcher + yaml config。

## TL;DR

**训练侧（Megatron + slime）已基本就绪；rollout 侧（Harbor/Terminus2 agent + 沙箱后端 + 任务池）还缺依赖，所以现在还不能一键开跑。**

56 卡的脚本/配置文件都已经为**当前这套 7 节点集群**写好了（4 actor 节点=32 卡训练，3 rollout 节点=24 卡推理，TP2×PP1×CP1×EP8）。补齐下面 §2 的 4 项后即可启动。§1 的 checkpoint 路径 bug 已在本次修复。

---

## 0. 集群与并行度映射（已确定）

| 项 | 值 | 说明 |
|---|---|---|
| 总节点 | 7 (1 head + 6 worker) | 见 `ray status` |
| 总 GPU | 56 (7×8 H20) | |
| **训练 (actor)** | **4 节点 / 32 卡** | `ACTOR_NUM_NODES=4`，`ACTOR_NUM_GPUS_PER_NODE=8` |
| **推理 (rollout)** | **3 节点 / 24 卡** | `ROLLOUT_NUM_GPUS=24`，3 个 SGLang engine（每 engine 8 卡）|
| actor 并行度 | TP2 × PP1 × **CP1** × EP8 → DP16 | `TP*CP*DP=32` 能被 `EP*ETP=8` 整除 |
| CP 必须=1 | GatedDeltaNet 线性注意力是 per-rank 递归 kernel，**非 context-parallel-safe**（见 `slime_plugins/models/qwen3_5.py`）|
| rollout 并行度 | 每 engine dp-attention + EP8 | `--sglang-enable-dp-attention --sglang-dp-size 8 --sglang-ep-size 8` |

> 用户需求「32 卡训练」= wrapper 默认的 `ACTOR_NUM_NODES=4 × 8`，无需改动。

---

## 1. ✅ 已修复：torch_dist checkpoint 路径 bug

**问题**：wrapper 原本把 `PRETRAINED_CKPT` 指向 `Qwen3.5-35B-A3B_torch_dist`，但该目录**磁盘上不存在**。raw 模式下 base 脚本会 `test -d "${PRETRAINED_CKPT}"` 直接失败；且本 run `USE_KL_LOSS=1` 要求 `--ref-load` 指向一个含 `latest_checkpointed_iteration.txt` 的有效 Megatron ckpt（`slime/utils/arguments.py:1683`）。

**磁盘实际存在的 torch_dist**：
```
/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B_torch_dist_tp4_ep4   （65G，release/，含 .distcp + metadata）
/path/to/shared_storage/hf_models/Qwen/Qwen3.5-35B-A3B-Base_torch_dist_tp4_ep4
```

**为什么能直接复用（无需重转）**：Megatron 的 `torch_dist`（PyTorch DCP）格式在**加载时按目标并行度重新分片**——sharded state dict 存的是**全局张量形状**，`_tp4_ep4` 只是保存时（convert 用 world_size=8→PP8）的命名标签，不是加载约束。所以 TP2/EP8 的本 run 可以直接 `--load` / `--ref-load` 这个 `_tp4_ep4` 目录。

**已改动**：wrapper 里
```bash
PRETRAINED_CKPT 默认  Qwen3.5-35B-A3B_torch_dist  →  Qwen3.5-35B-A3B_torch_dist_tp4_ep4
STAGED_TORCH_DIST_CKPT 改用 $(basename "${PRETRAINED_CKPT}") 自动跟随
```
> ⚠️ **前提假设**：`_tp4_ep4` 这份 torch_dist 是从**同一个 Instruct HF 权重**（`Qwen3.5-35B-A3B`，非 `-Base`）转出的。文件时间戳（HF 6/17 11:55，torch_dist 6/17 13:45，同源同日）支持这一判断，但**首跑前建议核对**（见 §3 冒烟）。若不放心，用下面的方式重新生成一份干净命名的：

**可选：重新转换（约 1 小时，单节点 8 卡）**
```bash
cd /path/to/workspace/slime-terminal-RL
bash scripts/convert-qwen3.5-35B-A3B-gy4.sh      # 用 tools/convert_hf_to_torch_dist.py
# 产物: /path/to/shared_storage/.../Qwen/Qwen3.5-35B-A3B_torch_dist
# 然后启动时 export PRETRAINED_CKPT=/path/to/shared_storage/.../Qwen/Qwen3.5-35B-A3B_torch_dist 覆盖即可
```
该转换脚本内部：world_size=8 → PP8（满足 `world_size ≤ num_layers=40`），source `scripts/models/qwen3.5-35B-A3B.sh`，输出为 `--ref-load`/`--load` 消费的 release ckpt。

---

## 2. ❌ 启动前必须补齐的 4 项（当前阻塞点）

### 2.1 Harbor / Terminus2 agent 运行时（rollout 硬依赖）

- **现象**：`python3 -c "import harbor"` → `ModuleNotFoundError`；rollout 入口 `slime_plugins/harbor/agent.py` 的 `SlimeTerminus2` 继承自 `harbor.agents.terminus_2.Terminus2`，import 即失败。
- **本 checkout 不含 harbor/src**（只有 slime 侧的 `slime_plugins/harbor/` 适配层）。
- **要做**：拿到 harbor 运行时源码目录（`harbor/src`），启动前 `export HARBOR_SRC=/path/to/harbor/src`（tmux launcher 与 wrapper 都强制要求，未设会直接报错退出）。它会被加进 `PYTHONPATH`。
- **配套 pip 依赖**：launcher 默认 `INSTALL_HARBOR_RUNTIME_DEPS=1`，会自动跑 `install_harbor_runtime_deps.sh` 装 `shortuuid/toml/tenacity/litellm/dirhash/claude-agent-sdk/supabase/openai==2.6.1`（+ Daytona）。**注意这个安装要在每个节点都发生**（tmux launcher 对 launcher 和每个 worker 都会执行 install 命令）。

### 2.2 沙箱后端凭证：Daytona（rollout 硬依赖）

- 当前 yaml `environment.type: daytona`，需要账号凭证；`env | grep -i daytona` → 空。
- **要做**（二选一）：
  - 内联：`export DAYTONA_API_KEY=dtn_xxx`
  - 或 `export DAYTONA_ENV_FILE=/path/to/daytona.env`（内含 `DAYTONA_API_KEY` 或 `DAYTONA_JWT_TOKEN`+`DAYTONA_ORGANIZATION_ID`，可选 `DAYTONA_API_URL`）
- `daytona` python 包由 2.1 的 installer 在 `INSTALL_DAYTONA=1` 时装（launcher 默认开）。
- Daytona 并发受账号配额限制（yaml `harbor_num_concurrent_trials: 96`），按你的配额调。`HARBOR_CREATE_RATE_PER_SEC=6.0` 控制建沙箱速率。

  **备选（无 Daytona 时）**：改用本地 Singularity/Apptainer 后端（参考 `harbor_terminal_bench2_config.yaml` 的 `singularity_env:SlimeSingularityEnvironment` 块）。但**当前机器上 `apptainer`/`singularity` 均未安装**（`which apptainer` 为空），选这条路要先装（见 `install_apptainer_ubuntu.sh` / `augment_harbor_runtime_sifs.sh`）并预备 `.sif` 镜像缓存——工作量比配 Daytona key 大，除非无法用 Daytona 否则不推荐。

### 2.3 terminal-bench 任务池 `HARBOR_TASKS_DIR`（launcher 硬依赖）

- tmux launcher：未设 `HARBOR_TASKS_DIR` 直接报错退出。
- **要做**：`export HARBOR_TASKS_DIR=/path/to/terminal-bench task pool`，目录结构须含 `*/task.toml`（yaml 的 `harbor_tasks_glob: "*/task.toml"`）。
- 本次未在 `/path/to/shared_storage` 下找到现成任务池，需要你提供路径或准备（可参考 `prepare_valid359_daytona_dataset.sh` / `prepare_tb2_train_v2_local_tasks.sh`）。
- `HARBOR_TRIALS_DIR`（trial 输出 scratch 目录）launcher 会自动生成到 `RUN_TRIALS_DIR` 并 sed 进 per-run config，一般无需手动设。

### 2.4 dop-fuse（已完成✅）

新 docker 的 fuse 客户端 `dop-fuse` 已在**全部 7 个节点**安装完成（本会话上一步）。若后续训练依赖 fuse 挂载共享盘，这一步已就绪。

---

## 3. 启动步骤（补齐 §2 后）

### 3.1 单机冒烟（强烈建议先做，验证 checkpoint + agent 链路）

```bash
cd /path/to/workspace/slime-terminal-RL
# 只跑 rollout、不训练，dump 一次 rollout 数据，验证 ckpt 能加载 + Harbor agent 能起：
DEBUG_ROLLOUT_ONLY=1 NUM_ROLLOUT=1 \
HARBOR_SRC=/path/to/harbor/src \
DAYTONA_API_KEY=dtn_xxx \
HARBOR_TASKS_DIR=/path/to/tb2-tasks \
bash examples/harbor_terminal_rl/run_35b_a3b_56gpu_daytona_actor4_rollout3_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.sh
```
（如需先验证 checkpoint 是否与 HF 同源：可临时把 `MEGATRON_TO_HF_MODE=bridge` 跑一次——bridge 模式直接从 HF 权重经 mbridge 加载 actor，绕过 torch_dist，但 bridge 与 `--use-kl-loss` 不兼容，需同时 `USE_KL_LOSS=0`。仅用于对比冒烟，正式训练仍用 raw + torch_dist。）

### 3.2 多节点正式启动（56 卡 fully-async）

在 **node0（launcher）**：
```bash
DAYTONA_API_KEY=dtn_xxx \
HARBOR_SRC=/path/to/harbor/src \
HARBOR_TASKS_DIR=/path/to/terminal-bench-2 \
bash examples/harbor_terminal_rl/start_35b_a3b_56gpu_daytona_fully_async_tmux.sh launcher <node0_ip>
```
在**其余 6 个 worker 节点**（用同一个 node0_ip）：
```bash
HARBOR_SRC=/path/to/harbor/src \
bash examples/harbor_terminal_rl/start_35b_a3b_56gpu_daytona_fully_async_tmux.sh worker <node0_ip>
```
- launcher 提交 Ray job（`train_async.py`，fully-async 要求 **colocate 关闭**）；worker 加入 Ray head 并靠 STOP marker 保活。
- 日志：`BASE_FOLDER/logs/launcher_<tag>.log`。
- checkpoint 存到 `BASE_FOLDER/SAVE_SUBDIR`，`SAVE_INTERVAL=2`。

---

## 4. 关键超参（wrapper 默认，来自已验证的 96 卡配方）

| 类别 | 值 |
|---|---|
| context | 统一 64k（rollout / SGLang / actor train 全 64k）|
| batch | `ROLLOUT_BATCH_SIZE=8`（8 个 task group）× `N_SAMPLES_PER_PROMPT=32` = 256 in-flight；`GLOBAL_BATCH_SIZE=256` |
| 动态 batch | `USE_DYNAMIC_BATCH_SIZE=1`，`MAX_TOKENS_PER_GPU=16384`（train 吞吐受限时先调这个）|
| 算法 | GRPO，`ENTROPY_COEF=0.005`，`USE_KL_LOSS=1` `KL_LOSS_COEF=0.001`，`KL_COEF=0`（两者不能同时非零），`LR=5e-7` |
| sample 模式 | `trajectory_normalized`（容忍前缀漂移；同一 trial 的 chunk 共享 group_id）|
| reward | CTRF 进度 partial reward，`HARBOR_PARTIAL_REWARD_MAX=0.4` |
| MoE dispatch | 训练侧 DeepEP all-to-all；rollout 侧 dp-attention |
| agent | `max_turns=120`，agent 墙钟 3600s，verifier 900s |

---

## 5. 依赖就绪核对表

| # | 项 | 状态 | 需要动作 |
|---|---|---|---|
| 1 | HF 权重 `Qwen3.5-35B-A3B` | ✅ 完整 | — |
| 2 | 模型形状脚本 `qwen3.5-35B-A3B.sh` | ✅ | — |
| 3 | torch_dist ckpt | ✅ 复用 `_tp4_ep4`（路径已修）| 首跑核对同源；或跑 `convert-qwen3.5-35B-A3B-gy4.sh` |
| 4 | 训练环境 torch/sglang/ray/mbridge/slime/Megatron | ✅ 均可导入 | — |
| 5 | 56 卡脚本 / yaml / launcher | ✅ 已存在（本会话修了 ckpt 路径）| — |
| 6 | dop-fuse（全 7 节点）| ✅ 本会话已装 | — |
| 7 | **Harbor 运行时 `HARBOR_SRC`** | ❌ 缺 | 提供 harbor/src 路径并 export |
| 8 | **Harbor pip 依赖** | ⚠️ 自动装 | launcher 会在每节点自动装（默认开）|
| 9 | **Daytona 凭证** | ❌ 缺 | export `DAYTONA_API_KEY`（或 env file）|
| 10 | **terminal-bench 任务池 `HARBOR_TASKS_DIR`** | ❌ 缺 | 提供含 `*/task.toml` 的目录 |

**补齐 #7 / #9 / #10 三个外部资源后，即可按 §3 启动。**
