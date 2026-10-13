# Harbor Terminal RL 训练中断复盘

更新时间：2026-06-09

本文只记录已经导致 Ray job / 训练闭环中断的错误。普通 parser warning、reward 为 0、GPU 利用率低但任务仍在跑等问题不放在这里。

## 总览

| 时间 | 实验 | 中断位置 | 直接错误 | 核心判断 |
| --- | --- | --- | --- | --- |
| 2026-05-20 | 9B full89 n=16 | actor train | CUDA OOM | terminal history 太长，训练 forward 吃完整 prompt+response；原生 entropy/logprob 路径显存峰值过高 |
| 2026-05-21 | 9B full89 n=16 | rollout 汇总到 train_data | `Not enough samples 255 for global_batch_size 256` | 单个 verifier timeout 导致样本少 1 条，破坏 slime 固定 batch/group contract |
| 2026-05-20/21 | 9B full89 初版 | rollout 汇总前 | `Environment start timed out after 600.0 seconds` 后样本不足 | container 启动慢/依赖安装慢，失败 trial 被丢弃后样本数不足 |
| 2026-05-23 | 64GPU 27B | actor train | CUDA OOM | 27B 在 TP=1/PP=8 下 logprob/cross-entropy 峰值贴满显存 |
| 2026-05-23 | 64GPU 27B resp4096 | actor train | CUDA OOM | `LOG_PROBS_CHUNK_SIZE=32` 后仍 OOM，说明需要 TP 分摊，不只是调 chunk |
| 2026-05-28 | 64GPU 27B trajectory | ATIF -> slime sample conversion | `RolloutConversionError`，大量 sample 被丢弃 | strict whole-trajectory 拼接假设不成立，真实 chat template 不是 token-level prefix extension |
| 2026-05-28/29 | 64GPU 27B high concurrency | 第二次 `update_weights` | Ray node heartbeat miss / `ActorDiedError` | rollout 节点被 Ray 判死，SGLangEngine actor 消失 |
| 2026-05-29/30 | 32GPU 9B turn_grouped | 第 5 次 `update_weights` 前后 | `ActorUnavailableError` / `FD Shutdown` | 高并发长时间运行后 rollout engine actor 不可连接 |
| 2026-05-29 | 32GPU 9B turn_grouped | 单个 trial LLM call | SGLang 400: requested tokens > 128k | 局部 trial 失败，不是主 job 直接中断；但会抬高失败率和 truncated 指标 |
| 2026-05-30 | 32GPU 9B actor2/rollout2 128k train | actor train | CUDA OOM at fused vocab CE | 128k turn_grouped train 太重，`TP=1/PP=8/CP=2` 下 logprob/cross-entropy 峰值贴满 95GiB；OOM 后 worker keepalive 导致实例空挂，最终被 0% GPU 查杀 |
| 2026-06-02 | 32GPU 9B actor2/rollout2 64k dynbatch | 第 7 次 train 后 `update_weights` | Ray host memory OOM / SGLangEngine 被 Ray kill | actor train 已完成，但 rollout 节点内存超过 Ray 95% 阈值；driver 崩溃后 Ray/Megatron/SGLang 残留占显存但 GPU util 为 0 |
| 2026-06-06 | 32GPU 9B actor1/rollout3 trajectory_normalized | 第 1 次 train 与 rollout 2 并行阶段 | Ray driver 收到 `SIGTERM`，Ray job stopped | 日志不支持 CUDA OOM / adapter exception / worker node died；直接原因是外部 SIGTERM 停掉 Ray job。性能背景是 96 trajectory 被拆成 644 train samples，actor train 只跑到 19/644，rollout 2 仍 0/3 |
| 2026-06-07 | 32GPU 9B actor1/rollout3 trajectory_retokenized | rollout 2 后 actor train | CUDA OOM at fused vocab CE | live run 确实进入 `trajectory_retokenized`，但 `last_assistant_message_span` 全为 `None`，retokenized fallback 从未命中，仍退化成大量 split/长 chunk；rollout 1/2 完成后 actor train 在 GPU 7 OOM |
| 2026-06-08 | 32GPU 9B Daytona 2train/2rollout trajectory_retokenized smoke | rollout 后 actor train 静态 microbatch 切分 | `AssertionError: static path: num_mbs (65) is not a multiple of dp_size * mb_group (2)` | 不是 Daytona sandbox 卡死，也不是 CUDA OOM；rollout/Daytona 路径已经产出结果和 dump，fatal 在训练侧 batch 形状：静态 `micro_batch_size=1`、DP=2 要求每步 sample/microbatch 数为偶数，但实际进入一步的是 65 |
| 2026-06-15 | 96GPU 27B TerminalWorld full1530 Daytona n20 partial16 bs16 | 第 20 步后 rollout 21 汇总 | `RuntimeError: Harbor fully-async rollout could not produce enough complete trainable groups: 15/16 groups after 160 completed group tasks` | Daytona sandbox 泄漏（`stop()` 在 `self._sandbox is None` 时跳过删除，累计 1815 次漏删 → 538 个孤儿）撞 RAM 1000GiB 配额 → 某个 group 反复建不出 sandbox → 凑不齐 `require_full_rollout_groups=true` 要求的 16/16 → 抛异常退出。已存 iter_9/iter_19，从 iter_19 续训 |

## 1. 9B full89 actor train OOM

来源：`Qwen3.5-9B_terminal_bench2_full89_grpo_n16_2epoch` 早期 full89 run。

### 现象

run 已经完成 rollout dump，并进入 Megatron actor train，随后在 policy loss 的 logprob/entropy 路径 OOM：

```text
Ray job id: raysubmit_SAGKephuEetLXBVB
failure time: 2026-05-20 10:17 CST

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 30.55 GiB.

stack:
  slime/backends/megatron_utils/loss.py:policy_loss_function
  slime/utils/ppo_utils.py:calculate_log_probs_and_entropy
  slime/utils/ppo_utils.py:compute_entropy_from_logits
```

失败 dump 摘要：

```text
num_samples: 256
response_len min/max/mean: 52 / 2595 / 379.84
total_len min/max/mean/median: 2836 / 57219 / 20448.26 / 19238.5
total_len > 16k: 157 / 256
num response_len at 8096 cap: 0
```

### 直接原因

不是 response 生成到了上限，也不是 Harbor runtime/SIF 问题。真正导致 OOM 的是训练样本的 `total_length = prompt + response` 太长。即使 `loss_mask` 只覆盖 response suffix，actor forward 仍然要吃完整 sequence。

当时使用 `last_turn` sample，最后一轮 response 不长，但最后一轮 prompt 里包含 terminal history、之前命令、环境 observation、文件内容和错误输出，所以 prompt 很长。

原生 slime loss 当时还会在 `entropy_coef=0.00` 时构造 entropy/logprob 图，导致 `[T, vocab]` 相关中间张量峰值很高。Qwen3.5 vocab 很大，`T` 接近 57k 时显存峰值不可控。

### 影响

- Ray job failed。
- 有 rollout dump，但没有可用训练 checkpoint。
- 说明 terminal-agent RL 不能只看 `response_length`，还必须控制训练侧 `total_length`。

### 已采取缓解

- 增加训练侧 token cap：`HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768`。
- 增加 `rollout-all-samples-process-path` 和 `rollout-data-postprocess-path`，在样本进入训练前裁剪 prompt prefix。
- 加 custom loss：`slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled`，当 `entropy_coef=0` 时不构造无用 entropy 图。
- 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，降低碎片化风险。

## 2. 9B full89 样本数不足：255 vs global_batch_size 256

来源：OOM 修复后的第二次 9B full89 run。

### 现象

第一轮 actor train 已经完整跑完，说明 OOM mitigation 生效。第二轮 rollout 汇总后失败：

```text
Ray job id: raysubmit_pA7eM2rPYNm9wiL5

ValueError: Not enough samples 255 for global_batch_size 256
```

相关 timeout：

```text
Trial slime-r2-g14-s6-i486 failed: Verifier execution timed out after 3600.0 seconds
```

失败前 token cap 正常执行：

```text
Harbor sample token cap trimmed 59/255 samples, cap=32768,
max_total_before=88124, max_total_after=32768
```

### 直接原因

配置期望每轮 rollout 有：

```text
rollout_batch_size=16
n_samples_per_prompt=16
global_batch_size=256
```

也就是 16 个 prompt group，每组 16 条 response，总共 256 条样本。第二轮有 1 个 Terminus/Harbor trial 没产出可训练 sample，flatten 后只剩 255 条。

slime 在 rollout 数据进入训练前要求至少能裁出一个完整 `global_batch_size`：

```text
if len(data) % global_batch_size != 0:
    trim_len = (len(data) // global_batch_size) * global_batch_size
    if trim_len == 0:
        raise ValueError(...)
```

255 小于 256，所以直接失败。

### 影响

- 不是 actor train OOM。
- 不是模型权重或 SGLang 启动问题。
- 训练已经能跑，但 Harbor rollout 返回样本数不能满足 slime/GRPO 固定 batch contract。
- 对 GRPO 更严重的是 group 结构也被破坏：某个 prompt 的 `n=16` 不完整，reward normalization 不再可靠。

### 已采取/建议修复

- 后续在 Harbor rollout 中增加 backfill / over-sampling，尽量凑满完整 group。
- 对 failed trial 生成占位 sample：`reward=0`、`status=FAILED`、`response_length=1`、`loss_mask=[0]`，保持 group size，同时不贡献 policy gradient。
- 不建议单纯把 `GLOBAL_BATCH_SIZE` 降到 128 作为唯一修复，因为会丢掉大量样本并破坏 group 结构。

## 3. 早期 container 启动超时导致样本不足

来源：9B full89 初版。

### 现象

第一轮 rollout 还没进入 train，多个 trial 环境启动超时：

```text
Trial slime-r1-g15-s0-i240 failed: Environment start timed out after 600.0 seconds
Trial slime-r1-g15-s15-i255 failed: Environment start timed out after 600.0 seconds

Job 'raysubmit_6YcnvR5T392YdGFv' failed
ValueError: Not enough samples 149 for global_batch_size 256
```

日志显示 container 内还在安装 server 依赖：

```text
[server] [harbor] Installing server dependencies (uvicorn/fastapi)...
[server] Collecting uvicorn
[server] Collecting fastapi
[server] Successfully installed ...
[server] [harbor] Installing tmux...
```

### 直接原因

不是 H20 GPU 不够，也不是 SGLang 没起来。失败点在 Harbor task environment start。

环境启动路径被多重因素叠加拖慢：

- Apptainer/Singularity 启动。
- `--writable-tmpfs --fakeroot --containall --pid`。
- 容器内 pip/apt 安装。
- 共享文件系统/网络。
- 多 trial 并发启动。

失败 trial 被当前 rollout glue 丢弃后，原本应有 256 条样本，实际只剩 149 条，最终触发 batch size 检查失败。

### 已采取缓解

- 预构建/增强 SIF，避免每次启动重复安装依赖。
- 把 SIF cache 放到 `/path/to/shared_storage/...`，避免 `/cq_1` 存模型或大缓存。
- 提高/调整 Harbor env timeout multiplier。
- 后续引入 backfill/over-sampling，避免个别环境失败直接打断训练。

## 4. 64GPU 27B actor train OOM：logprob/cross-entropy 峰值

来源：64GPU 27B 高并发 run `raysubmit_Yx6tHC8Me24x78cm`。

### 现象

run 失败在 actor 训练阶段，不在 Terminus env 或 rollout 阶段：

```text
ray::MegatronTrainRayActor.train()
slime_plugins/harbor/train_loss.py -> get_log_probs_and_entropy
slime/utils/ppo_utils.py -> compute_log_probs(logits_chunk.clone(), ...)
Megatron fused_vocab_parallel_cross_entropy
torch.OutOfMemoryError: Tried to allocate 244.00 MiB
GPU 6 free: 231.06 MiB
```

当时参数：

```text
rollout_max_response_len: 2048
rollout_max_context_len: 131072
seq_length: 32768
harbor_max_train_total_tokens: 32768
log_probs_chunk_size: 256
actor_num_nodes: 2
pipeline_model_parallel_size: 8
rollout_batch_size: 24
n_samples_per_prompt: 16
global_batch_size: 384
```

### 直接原因

rollout 已经能产出训练 batch，async 侧还在跑下一轮 trial。真正打挂 Ray job 的是第一轮 actor train 中单个 microbatch 的 logprob/cross-entropy 显存峰值。

`log_probs_chunk_size=256` 对 27B + 大 vocab 会产生 `chunk_size * vocab` 级别的临时 buffer。日志里只差十几 MiB 就能通过，但整体显存已经贴满。

### 已采取缓解

- `LOG_PROBS_CHUNK_SIZE=256 -> 32`。
- 保留 `SEQ_LENGTH=32768` / `HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768`，避免 27B actor train 输入过长。
- 后续把 `rollout_max_response_len` 提到 4096 是为减少单 step response 截断，不是为解决训练 OOM。

## 5. 64GPU 27B resp4096 第二次 actor OOM：需要 TP 分摊

来源：`resp4096_ctx128k_actorpp8_thd_rb24_os48_conc384_pn64_traincap32k_lpc32`。

### 现象

这轮已经能完成 SGLang engine 启动、TrialWorker rollout、`update_weights` 和第一轮 rollout dump，但 actor train 早期再次 OOM：

```text
actor train micro-batch: 20/192
torch.OutOfMemoryError: Tried to allocate 32.00 MiB
GPU 7 total: 95 GiB
GPU 7 free: 11.06 MiB
PyTorch allocated: 90.60 GiB
```

### 直接原因

这次已经把 `LOG_PROBS_CHUNK_SIZE` 降到 32，继续调小 chunk 收益有限。核心问题是 27B 在 `TP=1, PP=8` 下，每张 GPU 仍承担过大的 vocab/logprob/cross-entropy 峰值。

增加 actor 节点但保持同样 TP/PP 只会增加 data-parallel replica，不会降低每卡模型/loss 峰值。真正需要的是 tensor parallel。

### 已采取缓解

重启配置改为：

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

代价是 rollout 节点减少，env 侧吞吐会下降；收益是先降低 27B actor train 的单卡显存峰值。

## 6. strict trajectory adapter 转换失败

来源：2026-05-28 的 64GPU/27B trajectory 模式尝试。

### 现象

Ray/SGLang/Apptainer 都已经跑通，TrialWorker 正在跑 task，但 ATIF -> slime `trajectory` sample 转换阶段大量样本被丢弃：

```text
RolloutConversionError:
Cannot build whole-trajectory slime sample because the next Harbor prompt is not a prefix extension
at detail=0 turn=0. Use mode='turn' or 'last_turn'.

Dropping failed Harbor rollout sample ...
```

调用栈：

```text
slime_plugins/harbor/rollout.py:_run_one_group
slime_plugins/harbor/rollout.py:_run_one_with_semaphore
slime/ray/rollout.py:TrialWorker.run_trial
slime_plugins/harbor/rollout.py:_run_one_trial
slime_plugins/harbor/atif_adapter.py:samples_from_trial_result
slime_plugins/harbor/atif_adapter.py:samples_from_rollout_details
slime_plugins/harbor/atif_adapter.py:_trajectory_sample_from_detail
```

### 直接原因

strict `trajectory` adapter 假设相邻 LLM call 满足：

```text
next_prompt_token_ids == current_tokens + observation_delta
current_tokens = initial_prompt + completion_0 + observation_delta_0 + ...
```

真实 Terminus2 / SlimeSGLang 路径中，下一轮 prompt 是 chat history 通过 chat template 重新序列化后的完整 prompt。它可能包含上一轮 assistant message、tool/observation/user wrapper、special tokens、role tokens 等，不保证和上一轮 `completion_token_ids` 逐 token 拼成严格 prefix。

所以 `last_turn` 能工作，`turn` 能工作，但 strict whole-trajectory 拼接不成立。

### 影响

- 大量 sample 被丢弃。
- 可能进一步导致 rollout 有效样本不足或 group 不完整。
- 这不是 Ray/SGLang/Apptainer 启动问题。

### 已采取缓解

- 放弃 strict `trajectory` 模式，新增 `turn_grouped`：
  - 每个 LLM turn 是一个训练 sample。
  - GRPO grouping / reward normalization 仍按 trajectory。
  - 每个 turn 的 advantage 直接等于对应 trajectory 的 normalized advantage。

## 7. 64GPU 27B high concurrency：Ray node heartbeat miss / SGLangEngine 消失

来源：64GPU 27B high concurrency run，Taiji instance `8B1D81989E25E915019E3C1BB0361459`。

### 现象

这次已经完成第一轮 rollout 和第一个 train step：

```text
Timer actor_train end (elapsed: 5487.8s)
Timer train end (elapsed: 5488.0s)

step 1: {
  'train/loss': 0.010738333687186241,
  'train/pg_loss': 0.010738333687186241,
  'train/pg_clipfrac': 0.05203016474843025,
  'train/ppo_kl': 0.06377356499433517,
  'train/train_rollout_logprob_abs_diff': 0.0,
  'train/grad_norm': 0.7311346924990939,
  'train/step': 1
}
```

停止发生在 step 1 完成后，进入下一次 `update_weights` 时：

```text
ray.exceptions.RayTaskError(ActorDiedError):
  ray::MegatronTrainRayActor.update_weights()

ray.exceptions.ActorDiedError:
  class_name: SGLangEngine
  ip: 10.0.0.9
  The actor died because its node has died.
  the actor's node was terminated unexpectedly:
  health check failed due to missing too many heartbeats
```

### 直接原因

Ray 把 rollout 节点 `10.0.0.9` 判死，导致该节点上的 SGLangEngine actor 消失。第二次 `update_weights` 需要对所有 rollout engines 发 `pause_generation` / 同步权重，缺一个 actor 就失败。

### 根因判断

更偏资源/系统稳定性，而不是模型或 adapter：

- 高并发配置约 `TrialWorker=192`。
- rollout 节点有大量 Apptainer/Singularity/fuse/overlay 进程。
- monitor 里看到大量 timeout/traceback/failed 文本，说明 env/agent 层长尾很重。
- Ray 报错是 heartbeat miss，不是 CUDA OOM、不是 `RolloutConversionError`、不是 logprob mismatch。

### 已采取缓解

- 降低 64GPU high-concurrency 参数，例如 `harbor_num_concurrent_trials 192 -> 96`、`harbor_trials_per_node 48 -> 24`。
- 把 TrialWorker healthcheck timeout 提高到 30s。
- 设置 `harbor_kill_unhealthy_trial_workers: false`，避免忙碌 worker 被健康检查误杀。
- 设置 `harbor_fallback_to_inprocess_trials: false`，避免所有 worker 不可用时把容器压力集中打到 head。

## 8. 32GPU 9B turn_grouped：第 4 步后 rollout engine actor 不可用

来源：32GPU 9B turn_grouped run，Taiji instance `8B1D81049E49C12E019E5B4BDF170FCA`。

### 现象

2026-05-30 01:19 检查时，Ray head 不存在，launcher tmux 已退出，只剩 monitor 和 gpu_keeper：

```text
ConnectionError: Could not find any running Ray instance.

tmux:
  gpu_keeper
  slime_full89_32gpu_9b_turngrouped_monitor
```

实际进度：

```text
trajectory_json=2560
rollout_dumps=5
checkpoint_files=0
```

没有 checkpoint，因为 `SAVE_INTERVAL=20`，任务只跑到 rollout/train step 4 后失败。

失败点：

```text
Timer actor_train end (elapsed: 5650.8s)
step 4: ...
Timer update_weights start
Timer update_weights end (elapsed: 21.1s)

ray.exceptions.ActorUnavailableError:
  ray::MegatronTrainRayActor.update_weights()
  self.weight_updater.update_weights()
  ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])

The actor ... is unavailable:
RpcError: failed to connect to all addresses;
last error: UNKNOWN: ipv4:29.81.228.139:20012:
Failed to connect to remote host: FD Shutdown
```

### 直接原因

actor train 第 4 步完成了，但下一次 `update_weights` 需要先让所有 rollout engines `pause_generation`。其中一个 SGLang/Ray actor 在 worker 节点 `29.81.228.139:20012` 已不可连接，Ray 将 job 判失败。

### 根因判断

不是 loss 本身报错，也不是 32k train cap 直接导致。更像是：

- 384 TrialWorker 高并发过激。
- 每个 rollout 节点长时间保留大量 Apptainer/Singularity 相关进程。
- 长时间 actor train 期间 rollout engines 保持在线但环境/容器压力仍在。
- 后续 `pause_generation` 时某个 engine/Ray actor 已经不再可达。

### 影响

- 已有 5 个 rollout dump。
- 无 checkpoint。
- 说明 turn_grouped 能完成多轮 rollout/train，但 32GPU 9B 的高并发稳定性不足。

### 下次建议

- 降低并发到 `harbor_trials_per_node=64` / `harbor_num_concurrent_trials=192`，必要时再降到 96。
- 对 rollout engines 增加更强健康检查和自动重建策略。
- `SAVE_INTERVAL` 临时降到 2 或 4，避免长实验中断后没有 checkpoint。
- 清理或限制残留 Apptainer/fuse/overlay 进程。

## 9. 局部 trial context 超 128k：SGLang 400

来源：32GPU 9B turn_grouped run 中的单个 trial。

### 现象

日志里出现过 SGLang 400：

```text
SGLang /generate request failed status=400,
prompt_tokens=127678,
max_new_tokens=4096,
body={
  "error": {
    "message": "Requested token count exceeds the model's maximum context length of 131072 tokens.
    You requested a total of 131774 tokens: 127678 tokens from the input messages and 4096 tokens for the completion."
  }
}
```

### 直接原因

rollout context 确实是 128k：

```text
--rollout-max-context-len 131072
--sglang-context-length 131072
--sglang-max-prefill-tokens 131072
```

但当 prompt 已经接近 128k 时，仍请求固定 `max_new_tokens=4096`，总长度会超过 SGLang engine 的 context limit。

### 影响

- 这是局部 trial 失败，不是这次主 job 的直接中断原因。
- 但它会增加 failed/truncated 样本比例。
- 如果这些 failed samples 没有被 backfill/占位处理，会间接导致 group 不完整或样本不足。

### 与 `HARBOR_MAX_TRAIN_TOTAL_TOKENS` 的区别

这类错误发生在 rollout 生成阶段，与训练侧 32k cap 不是一回事：

```text
rollout-max-context-len / sglang-context-length:
  agent 在环境中生成时能看的上下文窗口。

HARBOR_MAX_TRAIN_TOTAL_TOKENS / seq-length:
  已生成样本进入 actor train 前的训练侧长度上限。
```

32GPU 9B turn_grouped 这次是：

```text
rollout 128k
train 32k
```

64GPU 27B turn_grouped 脚本是：

```text
rollout 128k
train 128k
```

`rollout/truncated` 或 `truncated_ratio` 主要来自生成阶段的 `finish_reason == "length"`，不是由训练侧 `HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768` 直接造成。

### 下次建议

- 在 Harbor LLM wrapper 中根据剩余 context 动态设置 `max_new_tokens`：

```text
remaining = context_len - prompt_tokens
effective_max_new_tokens = min(configured_max_new_tokens, max(1, remaining))
```

- 如果 `remaining` 太小，例如小于 128 或 256，直接提前终止该 trajectory 或压缩/裁剪 history。
- 记录 `harbor/context_overflow_samples`、`harbor/prompt_tokens_p95/p99/max`、`harbor/effective_max_new_tokens_min`。

## 10. 32k train cap 与 128k rollout context 的配置差异

这不是一个报错，但它和近期问题容易混淆，需要固定记录。

### 32GPU 9B turn_grouped

实际提交命令包含：

```text
--rollout-max-context-len 131072
--sglang-context-length 131072
--sglang-max-prefill-tokens 131072
--seq-length 32768
HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768
```

含义：

- rollout 生成侧是 128k。
- actor train 输入被 cap 到 32k。
- 这是为了避免 9B actor train 在 turn_grouped 的长上下文样本上 OOM。

### 64GPU 27B turn_grouped

脚本 `run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency_turn_grouped.sh` 显式设置：

```text
ROLLOUT_MAX_CONTEXT_LEN=131072
SGLANG_CONTEXT_LENGTH=131072
SEQ_LENGTH=131072
HARBOR_MAX_TRAIN_TOTAL_TOKENS=131072
CALCULATE_PER_TOKEN_LOSS=0
```

含义：

- rollout 生成侧是 128k。
- actor train 输入也是 128k。
- 这会显著提高训练侧显存和时间压力，需要更多 actor 并行/TP/CP 支撑。

## 11. 2026-05-30 32GPU 9B actor2/rollout2：128k train OOM 后空挂

来源：新 32GPU Taiji instance `8B1D820C9E6E175B019E76664E8F259C`，W&B run
`6iu6rpnf`，配置为 2 个 actor/train nodes + 2 个 rollout nodes，rollout/train 都对齐到 128k。

### 现象

训练已经进入 actor train，并且跑了很久：

```text
actor train: 9%|9         | 1157/12456 [5:27:53<280:41:16, 89.43s/microbatch]
actor train: 9%|9         | 1158/12456 [5:33:07<491:50:46, 156.72s/microbatch]
```

随后在 actor train 内部 OOM：

```text
ray.exceptions.RayTaskError(OutOfMemoryError): ray::MegatronTrainRayActor.train()
slime_plugins/harbor/train_loss.py: policy_loss_no_entropy_when_disabled
slime/backends/megatron_utils/loss.py: get_log_probs_and_entropy
slime/utils/ppo_utils.py: calculate_log_probs_and_entropy
slime/utils/ppo_utils.py: compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group)
Megatron fused_vocab_parallel_cross_entropy

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB.
GPU 7 total: 95.00 GiB
free: 19.06 MiB
process memory: 94.96 GiB
PyTorch allocated: 89.37 GiB

torchinductor generated:
buf10 = empty_strided_cuda((32, 1, 248320), ...)
```

Taiji 最终查杀信息：

```text
GPU利用率(最近3小时): 平均0.00% / 最大0.00% / 最小0.00%
系统已自动释放该实例
```

### 直接原因

这是确定的 CUDA OOM，不是 parser、rollout、SGLang metrics、或 Taiji 误杀。

OOM 点在当前 actor policy loss 计算当前策略 `log_probs` 的路径。虽然这轮打开了
`--use-rollout-logprobs`，old logprobs 可以使用 rollout 侧返回值，但 PPO ratio 仍然需要当前
actor 对 response token 的 logprob，所以当前 logprob/cross-entropy 仍然必须算。

同时 custom loss `policy_loss_no_entropy_when_disabled` 只跳过 entropy graph；它仍会调用
`get_log_probs_and_entropy(..., with_entropy=False)`，因此仍会进入：

```text
compute_log_probs -> fused_vocab_parallel_cross_entropy
```

### 为什么这轮特别慢

这轮把 32GPU 9B 的训练侧也提到 128k：

```text
SEQ_LENGTH=131072
HARBOR_MAX_TRAIN_TOTAL_TOKENS=131072
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=8
CP_SIZE=2
GLOBAL_BATCH_SIZE=256
sample mode: turn_grouped
```

`turn_grouped` 会把一个 trajectory 展开成多个 turn sample；每个 turn sample 的 prompt
可能包含很长 terminal history。128k train cap 让这些超长 turn 进入 actor forward/backward。
日志里的 `12456` 个 microbatch 和 `89-156s/microbatch` 已经说明该配置不可接受：
单个 actor train step 的 ETA 到数百小时量级。

### 为什么后面 GPU 变成 0

OOM 发生后，主 `train_async.py` Ray job 已经失败。但这次 worker 是用
`WORKER_KEEP_ALIVE=1` 启动的，worker 脚本会无条件：

```bash
tail -f /dev/null
```

所以主 job 失败后，Taiji 实例仍可能保持 Running，但没有 Megatron/SGLang 训练负载继续使用 GPU。
这解释了为什么最终最近 3 小时 GPU 平均/最大/最小都是 0，然后被低利用率规则自动释放。

### 结论

32GPU 9B 的 “2 actor nodes + 2 rollout nodes + 128k train + turn_grouped” 当前不可用：

- 显存不够，95GiB 卡只剩 19MiB 时还要申请 32MiB。
- 速度也不可接受，单 actor train step 内部有 12456 个 microbatch。
- `--use-rollout-logprobs` 和 no-entropy custom loss 不能消除当前 logprob/cross-entropy 的显存峰值。

### 下次建议

1. 32GPU 9B 不要直接用 128k train cap；先回到 `HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768` 或最多 `65536`。
2. 保留 rollout 128k，但训练侧对超过 cap 的 turn 做零 loss / drop，而不是强行送进 128k actor train。
3. 如果一定要训练 128k，先降低 `GLOBAL_BATCH_SIZE` / `ROLLOUT_BATCH_SIZE`，并启用 token-level dynamic batch；
   同时考虑 `TP=2` 分摊 vocab-parallel cross entropy，而不是 `TP=1`。
4. `LOG_PROBS_CHUNK_SIZE=32` 已经到较低值；继续降到 16 可能只缓解边缘 OOM，解决不了 12456 microbatch 的速度问题。
5. worker keepalive 必须改成带 watchdog/STOP marker：主 Ray job 失败后自动退出，避免空挂到 Taiji 查杀。

## 优先修复建议

1. **先防训练中断**：`SAVE_INTERVAL` 临时调低到 2-4，避免长时间训练没有 checkpoint。
2. **降低 Ray actor 不可用风险**：32GPU 高并发从 `384/128 per node` 降到 `192/64 per node`，64GPU 从 `192/48` 或更低开始。
3. **rollout context overflow 防护**：动态降低每步 `max_new_tokens`，剩余 context 太小时提前终止/压缩 history。
4. **训练侧 dynamic batch packing**：打开 `USE_DYNAMIC_BATCH_SIZE=1` 并设置 `MAX_TOKENS_PER_GPU`，减少 turn_grouped + `micro_batch_size=1` 的训练时间。
5. **保持 group 完整性**：failed/timeout trial 必须 backfill 或补占位 sample，不能直接丢导致 `Not enough samples`。
6. **27B actor 显存优先用 TP/CP 分摊**：不要只增加 actor nodes；增加 DP replica 不能降低单卡 logprob/cross-entropy 峰值。
7. **指标补齐**：至少记录 context overflow、effective max_new_tokens、SGLang actor reconnect/health、TrialWorker container counts、train sample length p95/p99/max。

## 2026-05-31 32GPU 9B 重启配置：避免 128k actor OOM

基于上面的 OOM 结论，新一轮 32GPU 9B 不再让 actor train 直接吃 128k，而是统一到 64k，并打开
token dynamic batch：

```text
instance: 8B1D801D9E6E1752019E7980B85A3387
script: examples/harbor_terminal_rl/run_full89_grpo_32gpu_9b_max100_actor2_rollout2_64k_timeout3600_turn_grouped_dynbatch.sh
rollout/train/SGLang context: 65536
HARBOR_MAX_TRAIN_TOTAL_TOKENS: 65536
USE_DYNAMIC_BATCH_SIZE: 1
MAX_TOKENS_PER_GPU: 32768
CP_SIZE: 2
LOG_PROBS_CHUNK_SIZE: 32
```

实际 `ray job submit` entrypoint 已确认带上：

```text
--rollout-max-context-len 65536
--seq-length 65536
--max-position-embeddings 65536
--use-dynamic-batch-size
--max-tokens-per-gpu 32768
--sglang-context-length 65536
--sglang-max-prefill-tokens 65536
--use-rollout-logprobs
```

这个配置的目的不是追求最大 context，而是先验证：

- `turn_grouped + progress reward` 在训练侧能稳定跑完首个 actor update。
- `rollout/truncated` 是否主要来自 64k context，还是来自 agent timeout / max turn / parser。
- dynamic batch 是否能把 128k 配置下 `12456 microbatch` 的极端情况降下来。
- 主 job 失败时，新的 STOP marker worker keepalive 是否能让实例主动退出，避免 GPU util 归零后被查杀。

## 2026-06-02 32GPU 9B 64k dynbatch：Ray host memory OOM 后空挂

来源：32GPU Taiji instance `8B1D801D9E6E1752019E7980B85A3387`，脚本：

```text
examples/harbor_terminal_rl/run_full89_grpo_32gpu_9b_max100_actor2_rollout2_64k_timeout3600_turn_grouped_dynbatch.sh
```

关键配置：

```text
ACTOR_NUM_NODES=2
ROLLOUT_NUM_GPUS=16
ROLLOUT_BATCH_SIZE=16
OVER_SAMPLING_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=16
GLOBAL_BATCH_SIZE=256
ROLLOUT_MAX_RESPONSE_LEN=4096
ROLLOUT_MAX_CONTEXT_LEN=65536
SEQ_LENGTH=65536
HARBOR_MAX_TRAIN_TOTAL_TOKENS=65536
USE_DYNAMIC_BATCH_SIZE=1
MAX_TOKENS_PER_GPU=32768
CP_SIZE=2
LOG_PROBS_CHUNK_SIZE=32
USE_ROLLOUT_LOGPROBS=1
CUSTOM_REWARD_POST_PROCESS_PATH=slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards
SAVE_INTERVAL=5
```

### 现象

该 run 不是一开始失败。它至少完成了 7 个 train step，并且已保存过 checkpoint：

```text
latest checkpoint: iter_0000004
checkpoint size: ~117G
```

第 7 个 actor train 本身已经跑完：

```text
actor train: 100%|██████████| 3817/3817 [5:42:28<00:00, 5.13s/microbatch]
Timer actor_train end (elapsed: 20573.0s)
perf 7:
  perf/actor_train_time: 20568.8s
  perf/actor_train_tok_per_s: 11910.4
  perf/train_wait_time: 77.7s
  perf/wait_time_ratio: 0.0038
```

随后进入 `actor_model.update_weights()` 时失败：

```text
Traceback:
  train_async.py: actor_model.update_weights()
  actor_group.py: ray.get([actor.update_weights.remote() ...])
  actor.py: self.weight_updater.update_weights()
  update_weight_from_distributed.py: ray.get([engine.pause_generation.remote() ...])

ray.exceptions.OutOfMemoryError:
1 worker(s) were killed due to the node running low on memory.
Memory on the node (IP: 29.81.244.168) was 2189.07GB / 2265.25GB (0.966369),
which exceeds the memory usage threshold of 0.950000.

Ray killed:
  Actor(... task name=SGLangEngine.__init__ ...)
```

Taiji 随后触发第 3 次低利用率告警：

```text
GPU利用率(最近3小时): 平均0.00% / 最大0.00% / 最小0.00%
```

### 直接原因

这次不是 CUDA OOM。直接原因是 **Ray host memory OOM**：

1. actor train step 7 已完成。
2. `update_weights()` 需要 pause / sync rollout 侧 SGLang engine。
3. rollout 节点 `29.81.244.168` 的 host memory 到达 96.6%，超过 Ray 默认 95% kill 阈值。
4. Ray memory monitor 杀掉一个 `SGLangEngine` worker。
5. `update_weights()` 抛 `RayTaskError(OutOfMemoryError)`，`train_async.py` driver 退出。
6. driver 退出后，Ray / MegatronTrainRayActor / SGLang / Apptainer 残留没有被脚本统一清理。
7. 残留进程占着 GPU 显存，但没有 CUDA kernel 继续跑，所以平台看到 GPU util 长时间为 0。

### 为什么 `WORKER_KEEP_ALIVE` 没保护 GPU

这轮 worker 命令带了：

```text
WORKER_KEEP_ALIVE=1
```

但脚本里的语义只是让 worker 节点 Ray 进程等待 stop marker：

```bash
while [ ! -f "${WORKER_STOP_MARKER}" ]; do
  sleep 60
done
```

它不是 GPU keeper，也不是 watchdog。它不会：

- 检测 `train_async.py` 是否已经异常退出。
- 检测 GPU util 是否为 0。
- 自动清理 Ray/SGLang/Megatron/Apptainer 残留。
- 自动从最新 checkpoint 重启。
- 启动任何 CUDA workload 来维持 GPU 利用率。

因此主 Ray job 崩溃后，worker keepalive 反而让实例继续空挂，直到 Taiji 低 GPU 利用率规则告警/查杀。

### 与 128k actor OOM 的区别

上一轮 128k 失败是 CUDA OOM：

```text
torch.OutOfMemoryError at fused_vocab_parallel_cross_entropy
```

这轮 64k 失败是 Ray host memory OOM：

```text
Ray memory threshold exceeded, SGLangEngine worker killed
```

二者共同点是失败后缺少统一 cleanup，都会造成“显存被占、GPU util 为 0”的空挂状态。

### 可疑诱因

这轮虽然把 train context 降到 64k 并打开 dynamic batch，但 rollout/env 侧仍然很重：

- `ROLLOUT_BATCH_SIZE=16`
- `OVER_SAMPLING_BATCH_SIZE=32`
- `N_SAMPLES_PER_PROMPT=16`
- 16 张 rollout GPU 上同时跑 SGLang engine、Terminus trial worker、Apptainer/FUSE。
- Ray 每节点预启动 worker 数很大，日志中可见 `--num_prestart_python_workers=384`。
- 长时间运行后 Apptainer/FUSE、SGLang scheduler、Ray worker 可能有内存累积。

最终在 `update_weights()` 触发 SGLang pause / engine 交互时踩到 Ray 的 host memory kill 阈值。

### 清理结果

后续手动清理时确认：

- node0、node1、node2 已释放干净，8 张卡显存均为 0，未再匹配到 Ray/SGLang/Megatron/Apptainer 进程。
- node3 的 `jizhi_client exec` 和从 node0 ssh 到 `29.81.244.168` 都异常/超时，无法可靠确认；用户决定重新启动该 32GPU 资源。

### 下次建议

1. **加 driver failure cleanup trap**：launcher 退出时统一执行 `ray stop --force`，并 kill `train_async.py`、`MegatronTrainRayActor`、`SGLangEngine`、`sglang`、`apptainer/singularity`、`squashfuse`、`fuse-overlayfs`。
2. **把 `WORKER_KEEP_ALIVE` 改名或注释清楚**：它是 worker shell keepalive，不是 GPU keeper。
3. **加真正 watchdog**：每 2-5 分钟检查 launcher log / `ray job status`。遇到 `Traceback`、`RayTaskError`、`OutOfMemoryError`、`ActorDiedError` 后自动 cleanup 或从 checkpoint 重启。
4. **降低 Ray/rollout host memory 压力**：先降低 `OVER_SAMPLING_BATCH_SIZE` 或 rollout 并发，减少每节点预启动 worker 数，避免长时间运行后 host memory 靠近 95%。
5. **临时调高 Ray memory 阈值不是首选**：`RAY_memory_usage_threshold` 可以绕过 Ray kill，但如果真实 host memory 已经接近 2.2TB，继续跑可能把节点拖死；优先减少并发和清理残留。
6. **缩短 checkpoint 间隔**：当前可恢复点是 `iter_0000004`，step 7 已训练完但未保存；长 run 建议保持 `SAVE_INTERVAL=2-5`，直到稳定性确认。

## 2026-06-06 32GPU 9B actor1/rollout3 trajectory_normalized：Ray job 被 SIGTERM 停掉

来源：

```text
Taiji server: self_improving_terminal_agent_RL_32GPU_slime_v4
instance: 8B1D80239E921BC7019E98D627820C74
Ray job id: raysubmit_tPND8RU2RMdQLwNB
run tag: 20260606_123111
launcher log: /tmp/slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_20260606_123111_launcher.log
monitor log: /tmp/slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_20260606_123111_monitor.log
base folder: /path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_actor1_rollout3_batch3_n32_fully_async_trajectory_normalized_timeout3600_20260606_123111
```

### 现象

Taiji 实例仍显示 `Running`，但训练闭环已经不在：

```text
node0 tmux: only gpu_keeper
ray status: Could not find any running Ray instance
ray job list: Could not find any running Ray instance
```

launcher 最后一段关键日志：

```text
2026-06-06 13:59:49
*** SIGTERM received at time=1780725589 on cpu 213 ***

stack:
  ray::core::GetRequest::Wait()
  ray::core::CoreWorkerMemoryStore::Get()
  ray::core::CoreWorker::GetObjects()
  ray._raylet.CoreWorker.get_objects()

Job 'raysubmit_tPND8RU2RMdQLwNB' was stopped

2026-06-06 13:59:57 launcher wrote worker stop marker:
.../.stop_workers_29_81_226_139_6379
```

3 个 worker 节点没有先报 OOM、ActorDied 或 NodeDied。worker 日志都是正常等待 stop marker，随后退出 Ray：

```text
worker keep-alive enabled; waiting for stop marker: .../.stop_workers_29_81_226_139_6379
worker stop marker detected; stopping Ray worker
Stopped all 4 Ray processes.
```

### 运行进度

rollout 1 已完成，并 dump 了数据：

```text
2026-06-06 13:45:52
Harbor fully-async rollout 1: done in 3757.4s
Save debug rollout data to .../rollout_dumps/1.pt
```

本轮转换结果：

```text
harbor/num_groups: 3
harbor/num_trajectory_samples: 96
harbor/num_samples: 644
harbor/num_converted_samples: 644
harbor/num_failed_trials: 0
harbor/num_missing_rollout_details: 0
rollout_logprob_samples: 644
train_context_trimmed_samples: 94
```

随后 actor train 开始，但因为 `trajectory_normalized` 把 96 条 trajectory 拆成 644 个 train samples，microbatch 数非常大：

```text
2026-06-06 13:46:03 actor train starts
actor train: 0/644
2026-06-06 13:59:20 actor train: 19/644 [13:16<6:37:16, 38.14s/microbatch]
```

fully-async rollout 2 同时启动，但到 SIGTERM 前仍没有完成任何 group：

```text
2026-06-06 13:47:30 Harbor fully-async rollout 2: target=3 queue_warm=0 max_concurrent_groups=3
2026-06-06 13:59:30 Harbor fully-async rollout 2: collected 0/3, queue=0, completed=0, elapsed=720.7s
```

### 已排除的原因

这次不是 CUDA OOM：

```text
timing summary: oom=0
launcher grep: no CUDA out of memory / OutOfMemory fatal tail
```

也不是 worker 节点先掉线：

```text
worker-0/1/2 logs:
  no RayActorError
  no ActorDied
  no NodeDied
  no worker died
  only stop marker detected after launcher wrote it
```

也不是 `trajectory_normalized` adapter 当场抛 conversion exception。rollout 1 的转换已经成功：

```text
harbor/num_converted_samples: 644
harbor/num_failed_trials: 0
harbor/num_missing_rollout_details: 0
```

日志里有 3 次 RolloutManager healthcheck timeout：

```text
Rollout auxiliary worker health check failed after 30.0s;
keeping the actor because worker kill is disabled.
A busy async actor can fail this probe while all concurrency slots are occupied.
ray.exceptions.GetTimeoutError: Get timed out: some object(s) not ready.
```

这些不是最终 fatal。日志明确说 `killed=0`，后续 rollout 2 继续启动：

```text
Rollout auxiliary worker health check succeeded for 0/3 workers; failed=3, killed=0.
Harbor fully-async rollout 2: target=3 queue_warm=0 max_concurrent_groups=3
```

### 直接原因

直接中断原因是 **Ray driver / Ray job 收到了外部 `SIGTERM`**。当时主进程正在 `ray.get()` 等 Ray object，收到信号后 Ray job 被标记 stopped，launcher 的 `EXIT` trap 写 stop marker，worker 节点跟随 stop marker 正常停 Ray。

从可见日志无法区分这个 `SIGTERM` 来自哪一层：

- Taiji 控制面或任务管理器主动 stop。
- 人工/自动化脚本停止 Ray job 或 launcher tmux。
- 外部 watchdog / 资源策略给进程发 SIGTERM。

但它不像 Python 代码主动抛异常退出，也不像 Ray memory monitor kill worker 后引发的 driver exception。

### 性能背景

这次 SIGTERM 不是由 sample split 本身直接触发，但 sample split 把训练阶段放大得很重：

```text
96 trajectory -> 644 train samples
actor train: 19/644 after 13m16s
estimated remaining: about 6h37m
```

同期 rollout 2 也在跑，12 分钟还没有收集到 3 个 group 中的任何一个。这说明当时系统处于非常长的 train/rollout overlap 阶段，即使不被 SIGTERM 停掉，也会是一轮很慢的训练。

后续对同一个 rollout dump 的 retokenized dry-run 显示：

```text
trajectory_normalized: 96 trajectory -> 644 samples
trajectory_retokenized: 96 trajectory -> 98 samples
real rewrite splits left: 2
```

因此 `trajectory_retokenized` 不是修这个 SIGTERM 的直接原因，但它能显著降低每轮 actor train 的 microbatch 数，避免训练阶段被 644 个 chunk 拖到数小时。

### 下次建议

1. **先切到 `trajectory_retokenized`**：目标是把这类 96 trajectory 的训练样本数从 600+ 降到接近 96，降低 actor train 时长。
2. **保留 launcher/worker cleanup trap**：当前 stop marker 机制能让 worker 跟随退出；后续还应确保 head 侧收到 SIGTERM 时统一 `ray stop --force` 并清理 SGLang/Apptainer 残留。
3. **记录 SIGTERM 来源**：在 launcher 外层加 signal trap，打印 `TERM` 捕获时间、父进程、Ray job status 和 `ps -ef --forest` 摘要，便于区分人工 stop、Taiji stop 和脚本 stop。
4. **监控 actor train 进度**：如果 `actor train` microbatch 数异常变大，比如数百个 chunk，应当在 monitor verdict 里标红，提示 sample split inflation。
5. **不要把这次误判成 CUDA OOM**：没有 OOM 证据；降低 `LOG_PROBS_CHUNK_SIZE` 或 train cap 不是对症修复。

## 2026-06-07 32GPU 9B actor1/rollout3 trajectory_retokenized：span 记录失效后 actor OOM

来源：

```text
server: self_improving_terminal_agent_RL_32GPU_slime_v4
instance: 8B1D80239E921BC7019E98D627820C74
Ray job: raysubmit_RgDyzCFP7KRXjWcu
run tag: 20260607_022356
base folder: /path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_actor1_rollout3_batch3_n32_fully_async_trajectory_retokenized_timeout3600_20260607_022356
launcher log: /tmp/slime_full89_32gpu_9b_a1r3_b3n32_trajretok_fa_20260607_022356_launcher.log
```

### 现象

任务不是还在继续跑，而是在 `2026-06-07 04:44:11` 失败退出。Ray job entrypoint 退出码为 1，launcher 写了 worker stop marker。

直接报错发生在 actor train 的 loss/logprob 路径：

```text
ray.exceptions.RayTaskError(OutOfMemoryError):
ray::MegatronTrainRayActor.train()

torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 32.00 MiB.
GPU 7 total capacity 95.00 GiB, free 33.06 MiB.
Process 1455506 has 94.94 GiB memory in use.

stack:
  slime_plugins/harbor/train_loss.py:policy_loss_no_entropy_when_disabled
  slime/backends/megatron_utils/loss.py:get_log_probs_and_entropy
  slime/utils/ppo_utils.py:calculate_log_probs_and_entropy
  megatron/core/fusions/fused_cross_entropy.py
```

rollout 进展：

| item | value |
| --- | ---: |
| rollout 1 dump | `1.pt`, 114 MB |
| rollout 1 samples | 608 |
| rollout 2 dump | `2.pt`, 58 MB |
| rollout 2 samples | 253 |
| checkpoints | 0 |

长度分布：

| dump | samples | total_len p50 | total_len p90 | total_len max | response_len p99 | response_len max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.pt` | 608 | 31,739 | 65,536 | 65,536 | 49,545 | 65,535 |
| `2.pt` | 253 | 29,335 | 53,038 | 65,536 | 50,760 | 65,535 |

### 根因

这次配置确实是：

```text
harbor_sample_mode: trajectory_retokenized
```

但 live dump metadata 显示 retokenized fallback 一次都没有命中：

| dump | retokenized prefix turns | normalized prefix turns | split-after turns |
| --- | ---: | ---: | ---: |
| `1.pt` | 0 | 5905 | 512 |
| `2.pt` | 0 | 5624 | 157 |

原因不是缺少文本字段。抽样 `slime-r1-g2-s27-i91`：

```text
completion_text: present, len=100
last_assistant_message_content: present, len=100
completion_text[i] == last_assistant_message_content[i+1]: 99/99
last_assistant_message_span: 0/100 non-null
```

`last_assistant_message_span` 全为 `None`，导致 `_retokenized_transition_context()` 直接返回 `None`，adapter 退化到 `trajectory_normalized` 的 token-prefix/normalized-prefix 逻辑，仍然产生大量 split。

span 记录失效的具体原因是 Qwen chat template 行为：当 assistant message 被单独 render 成最后一条消息时，template 会插入 thinking prefill：

```text
<think>

</think>
```

但同一条 assistant 在完整历史里后面跟着 user 时没有这个 prefill。旧 `_message_token_span()` 用“render 到当前 assistant 为止”的 token 序列去做 full prompt prefix check，因此 prefix check 失败，span 被记录成 `None`。

### 已修复

`slime_plugins/harbor/llm.py` 已改为优先用实际 prompt text 计算 span：

1. decode 当前 `/generate` 实际使用的 `prompt_token_ids`；
2. 在 decoded prompt text 里定位上一条 assistant message content；
3. 用 tokenizer `return_offsets_mapping=True` 把字符 span 映射回 token span；
4. 要求 re-tokenized prompt ids 与原始 `prompt_token_ids` 完全一致；
5. 要求 span token ids decode 后等于 assistant content；
6. 只有该路径不可用时才回退到旧 prefix-render 算法。

新增测试：

```text
tests/test_harbor_llm.py::test_last_assistant_history_projection_uses_prompt_text_when_template_prefix_drifts
```

验证：

```text
python -m py_compile slime_plugins/harbor/llm.py tests/test_harbor_llm.py tests/test_harbor_atif_adapter.py
python -m pytest tests/test_harbor_llm.py tests/test_harbor_atif_adapter.py
# 24 passed
```

live result sanity check：

```text
slime-r1-g2-s27-i91
before: 18 samples, retokenized_prefix_turns=0
after recomputing spans with fixed helper: 1 sample, retokenized_prefix_turns=17, split=0
```

top 10 most-split trajectories from `1.pt` all collapsed to one sample after recomputing spans.

### 后续注意

这次修复解决的是 `trajectory_retokenized` live run 没起作用的问题，不直接解决训练显存。修复后 trajectory 会被真正拼回去，很多样本会恢复到 64k 以上，例如抽样中有 90k、117k、122k token 的单 trajectory。

因此后续重跑前需要配合：

1. context parallelism；
2. 或训练侧 cap/trim，且保证 `loss_mask` 与 `rollout_log_probs` 长度仍严格等于 `response_length`；
3. monitor 继续检查 `harbor_trajectory_retokenized_prefix_turns > 0`，否则说明 retokenized fallback 又退化了。

## 2026-06-08 32GPU 9B Daytona 2train/2rollout trajectory_retokenized smoke：rollout 后静态 batch 形状断言

来源：

```text
server: self_improving_terminal_agent_RL_32GPU_slime_v4
instance: 8B1D80739E921BCD019EA0CF5BD51796
run tag: d60p30_5g_0608_0200
script: examples/harbor_terminal_rl/start_full89_32gpu_9b_daytona_2train2rollout_60x_retokenized_smoke_tmux.sh
config: examples/harbor_terminal_rl/harbor_terminal_bench2_full89_32gpu_9b_daytona_retokenized_2train2rollout_60x_smoke.yaml
```

### 现象

这次不是 Daytona smoke 卡在 sandbox 创建或 trial 执行阶段。运行过程中已经验证到：

```text
Daytona/SGLang/train/update_weights 路径可以启动
rollout 侧产出 result.json 约 69 条
rollout_dumps 产出 1 个 dump
没有看到 CUDA OOM 或 Daytona 300GiB quota fatal
```

随后 launcher 日志里 fatal：

```text
AssertionError: static path: num_mbs (65) is not a multiple of
dp_size * mb_group (2); got step_size=65, micro_batch_size=1,
dp_size=2, mb_group=1
```

失败后主训练进程已经退出，后续表现为实例空闲；不是训练还在正常推进。后来该 Taiji instance 已变成 completed，无法再 exec 进节点读取 `/tmp` 原始日志。

### 直接原因

断言来自 `slime/utils/dp_schedule.py` 的静态 microbatch 切分逻辑。训练侧配置是 2 个 actor/train 节点，因此 DP=2；静态路径 `micro_batch_size=1` 下，每一步 microbatch 数必须能被 `dp_size * mb_group = 2` 整除，保证每个 DP rank 拿到相同 microbatch 数。

这次实际进入某一步的是 65 个 sample/microbatch：

```text
step_size=65
micro_batch_size=1
dp_size=2
mb_group=1
```

65 是奇数，无法均分到 2 个 DP rank，所以训练侧主动 fail fast。这个 fail fast 是合理的；如果静默继续，会造成 rank 间 microbatch 数不一致，后续 PP/DP 同步更危险。

### 为什么看起来像卡住

fatal 发生在 rollout 之后、actor train 开始前后的 batch schedule 阶段。Ray job 退出后没有真实训练进程继续消耗 GPU；如果没有 keeper，Taiji 看到的是低 GPU 利用率，容易触发查杀告警。所以外观看起来像“smoke 卡住/不动了”，本质是主 job 已经失败，只剩实例还活着。

### 暴露的问题

这次主要暴露的是 **fully-async rollout 输出到 train batch 的计数 contract 还不稳**：

1. 配置层面期望是 `GLOBAL_BATCH_SIZE=64`、2 个 rollout group、每组 32 条 trajectory。
2. 训练侧实际看到 `step_size=65`，说明进入该 step 的 sample 数不是偶数。
3. 可能来源有两类，需要下次保留 dump 后复核：
   - 某个 trajectory/group 在 `trajectory_retokenized` 转换后仍产生了 2 个 train samples/chunks；
   - fully-async 收集阶段把 late/extra result 一起送进了训练 batch，导致实际 sample 数超过预期。

无论是哪一种，训练侧静态 path 都要求最终 step sample/microbatch 数能被 DP 整除。Daytona 本身不是这次 fatal 的根因；它只是替代了 Apptainer env，已经证明能跑出 rollout 结果。

### 后续建议

1. 下一次 smoke 把 launcher 日志同步到共享盘，避免 Taiji instance completed 后 `/tmp` 日志丢失。
2. 在 fully-async 汇总到 train 前打印并校验：
   - result count；
   - converted sample count；
   - group_id -> sample count 分布；
   - `step_size % (dp_size * micro_batch_size * mb_group)`。
3. 对 smoke 可先设置 `USE_DYNAMIC_BATCH_SIZE=1` 规避奇偶问题，但正式修复仍应保证 group/sample 计数符合训练 contract。
4. 如果坚持静态 batch，收集/转换后要么裁掉多余 sample，要么补齐到 DP 对齐边界；不能让 65 这种 batch 静默进入 actor train。
5. Daytona cleanup 已补 REST delete-all 路径，失败后已确认远端 sandbox 清到 `remote candidates=0 disk_gib=0`。

### 已加保护

`slime/ray/rollout.py` 现在会在 reward post-process 之前检查静态 DP microbatch 对齐。如果某个 step 会触发 `num_mbs % (dp_size * mb_group) != 0`，会优先模拟删除一个能让该 step 对齐的 zero-reward trajectory/group；找不到可修复的 zero-reward group 时继续 assert，并打印 step、sample count、microbatch count 和 zero-reward group sizes。对应纯逻辑在 `slime/utils/dp_schedule.py::select_zero_reward_groups_to_drop_for_static_alignment`，覆盖了 Daytona smoke 的 `64 groups -> 65 samples` 场景。

## 2026-06-09 32GPU Daytona partial cutoff smoke：3 小时低 GPU 利用率查杀

来源：

```text
server: self_improving_terminal_agent_RL_32GPU_slime_v4
instance: 8B1D81849E921BCB019EA7E742D61FA8
run tag: daytona_partial26_groupserial32_clean_20260609_1103
script: examples/harbor_terminal_rl/start_full89_32gpu_9b_daytona_2train2rollout_60x_retokenized_smoke_tmux.sh
config: examples/harbor_terminal_rl/harbor_terminal_bench2_full89_32gpu_9b_daytona_retokenized_2train2rollout_60x_smoke.yaml
```

### 现象

Taiji 在第 3 次告警后执行查杀并释放实例：

```text
GPU utilization over last 3 hours: avg 14.29%, max 50.69%, min 0.00%
rule: GPU avg < 30% and runtime > 4h
action: instance released
```

这次不是 CUDA OOM，也不是训练进程主动退出；实例是被平台的低利用率规则释放。查杀前我们已经临时在 train/rollout 节点补了 keeper，但补得太晚，3 小时滑动平均已经被前面长时间低利用率拉低，短时间把即时利用率抬到 30%-50% 也救不回 3 小时平均。

### 当时在测试什么

这次 smoke 不是测模型效果，而是在测 Daytona + partial group cutoff 工程链路：

```yaml
harbor_num_concurrent_trials: 32
harbor_fully_async_max_concurrent_groups: 1
harbor_partial_group_min_units: 26
harbor_trials_per_node: 16
```

目标是每次只启动 1 个 rollout group：组内最多 32 个 Daytona sandbox，收够 26 个 trainable trajectory 后触发 partial cutoff，取消长尾 sandbox，再启动下一个 group。这样可以验证 group 内 oversampling，同时避免两个 group 同时展开成 64 个 sandbox 撞 Daytona 300GiB quota。

查杀前 clean run 的状态：

```text
update_weights: completed, about 14.8s
rollout: Harbor fully-async rollout 1 target=2 max_concurrent_groups=1
group_0 trajectory.json: 32
group_0 result.json: 20
rollout_dumps: 0
Daytona remote: 12 sandboxes / 60GiB
new quota errors: none
OOM/Traceback: none
```

也就是说 clean run 已经证明了：

1. Ray 4 节点和 SGLang/update_weights 能启动；
2. 单 group 并发下 Daytona 正常创建 32 个 sandbox；
3. 旧 quota error 消失；
4. group_0 还没到 `result.json >= 26`，所以 partial cutoff 尚未触发，训练还没有拿到 rollout dump。

### 直接原因

这次被杀的直接原因是 **平台按 3 小时窗口计算低 GPU 利用率，keeper 启动太晚**。

在 fully-async rollout 等待 terminal-bench/Daytona 长尾时，train 节点会长时间处于 `train_wait`，真实训练 GPU 利用率接近 0。即使 rollout 节点的 SGLang scheduler 有部分卡在跑，整体 32 卡的平均仍可能低于 30%。当平台已经发到第 3 次告警时，再临时启动 keeper 只能改善即时利用率，无法快速抬高过去 3 小时平均。

### Daytona cleanup 踩坑

更早的一次重启还被 Daytona quota 卡住。关键坑点是：远端 Daytona sandbox 的 `name` 实际是 UUID，不是 Harbor trial 名 `slime-r...`。

错误假设：

```text
按 name-prefix=slime-r1- 查询/删除，就能清理当前 rollout 的 sandbox
```

实际情况：

```text
remote candidates=60 disk_gib=300 prefix=None
name=<uuid>
labels={"code-toolbox-language": "python"}
createdAt around 2026-06-09T02:27Z / 02:38Z / 02:54Z
```

因此按 `slime-r...` 前缀清理会显示 0，但真实远端已经有 60 个 5GiB sandbox 占满 300GiB。后续即使把 `harbor_num_concurrent_trials` 降到 32，也会继续报：

```text
Failed to create sandbox: Total disk limit exceeded. Maximum allowed: 300GiB.
```

已补的工具改动：

1. `examples/harbor_terminal_rl/daytona_cleanup_slime_sandboxes.py` 的 `*-all-rest` 路径会打印 `created/updated/labels`；
2. 新增 `--created-before`，可以只清理某个时间点之前的 UUID sandbox；
3. 这次用 `--created-before 2026-06-09T02:50:00Z` 清掉 59 个旧 sandbox，释放 295GiB；
4. 再清掉损坏 run 剩余 1 个 sandbox 后，确认 `remote candidates=0 disk_gib=0`，才启动 clean run。

### 另一个误判点

不要把 `trajectory.json` 数当成 rollout group 完成数。

这次 clean run 中 `group_0 trajectory.json=32` 很早就齐了，但 `result.json` 当时只有 20，rollout manager 仍然：

```text
Harbor fully-async rollout 1: collected 0/2, queue=0, completed=0
```

说明 `trajectory.json` 只是 agent 侧写出的轨迹文件，不代表 Trial 已完成、reward/result 已落盘，也不代表 adapter 已转换出 trainable sample。partial cutoff 的有效观察指标应优先看：

1. `result.json` 数；
2. converted/trainable sample 数；
3. rollout manager 是否出现 accepted/partial group 日志；
4. rollout dump 是否写出。

### 下次重拉机器前的硬性 checklist

1. **先清 Daytona 远端 quota**：

```bash
python3 examples/harbor_terminal_rl/daytona_cleanup_slime_sandboxes.py \
  --action delete-all-rest \
  --dry-run \
  --request-timeout 20
```

必须确认 `remote candidates=0 disk_gib=0`，或者只剩明确属于当前 live run 的 sandbox。不能只查 `--name-prefix slime-r...`。

2. **不要在已有 3/3 告警的旧实例上做 smoke**：

3 小时平均已经被污染时，最好直接换新实例；如果必须继续用，keeper 必须立即启动并持续跑够一个窗口，不要等平台第 3 次告警。

3. **启动前先决定 keeper 策略**：

Daytona/terminal-bench rollout 阶段 train nodes 会 idle。为了避免训练任务查杀规则，smoke 阶段需要二选一：

- 使用非训练/调试类型的资源跑纯 rollout smoke；
- 或在 train-wait 阶段主动开 keeper，等正式 train step 开始后再关。

4. **单 group smoke 配置优先用保守值**：

```yaml
harbor_num_concurrent_trials: 32
harbor_fully_async_max_concurrent_groups: 1
harbor_partial_group_min_units: 26
harbor_trials_per_node: 16
```

不要再用 `56/2` 这种接近 300GiB quota 的配置做首轮验证。`56/2` 会让两个 group 竞争创建 64 个 5GiB sandbox，在存在任何残留或服务开销时都会撞 quota。

5. **观察 progress 时看 result，不看 trajectory**：

最低限度每 2-3 分钟记录：

```text
result.json by group
trajectory.json by group
Daytona remote candidates/disk
rollout manager collected/completed
rollout_dumps count
GPU util by node
```

只有 `result.json >= harbor_partial_group_min_units` 后，才应该期待 partial cutoff；只有 rollout dump 出现后，才算进入训练侧验证。

### 后续待做

1. 给 Daytona sandbox 增加可搜索的 Harbor trial metadata/name，避免只能按 UUID 和时间过滤清理。
2. fully-async rollout 日志里增加 per-group `result/trainable/trajectory` 计数，减少人工误判。
3. 在 smoke launcher 里加 preflight：如果 Daytona remote disk 已接近 300GiB，直接 fail fast，不启动训练。
4. 为 Taiji 训练任务加一个受控的 train-wait keeper 开关，并在第一步真实 train 开始后自动关闭。

### 2026-06-09 12:20 clean smoke 失败：host runtime 缺 Harbor 依赖

new instance:

```text
self_improving_terminal_agent_RL_32GPU_slime_v4
instance: 8B1D81E79E920C67019EAA8E8D842253
run_tag: daytona_partial26_groupserial32_clean_20260609_1220
launcher log: /tmp/slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_daytona_partial26_groupserial32_clean_20260609_1220_launcher.log
```

这次启动链路本身走通了：

```text
ray cluster has 4/4 nodes
Timer update_weights end (elapsed: 14.9s)
starting Harbor fully-async rollout worker
Harbor fully-async rollout 1: target=2 queue_warm=0 max_concurrent_groups=1
```

但 rollout trial 还没有创建 Daytona sandbox 就全部失败。异常栈的第一原因是：

```text
File "/path/to/workspace/code/harbor/src/harbor/trial/trial.py", line 15, in <module>
  from tenacity import (
ModuleNotFoundError: No module named 'tenacity'

ImportError: Harbor is required for slime_plugins.harbor rollout. Add harbor/src to PYTHONPATH or install Harbor in the slime runtime environment.
```

随后 fully-async 层看到所有 trial 都失败，按预期 fail fast：

```text
Dropping incomplete Harbor fully-async group with 0/32 trajectory units.
RuntimeError: Harbor fully-async rollout could not produce enough complete trainable groups: 0/2 groups after 16 completed group tasks.
```

判断：

1. 这不是 Daytona quota 问题。启动前和失败后 `delete-all-rest --dry-run` 都是 `remote candidates=0 disk_gib=0`。
2. 这不是 partial group cutoff 逻辑问题。trial 在 import Harbor 依赖阶段就失败了，还没进入 sandbox 创建、agent 执行或 result 转换。
3. `PYTHONPATH` 已经能找到 `harbor/src`，缺的是 Harbor source runtime 的三方包。

修复：

1. 复用已有 `examples/harbor_terminal_rl/install_harbor_runtime_deps.sh`，里面包含 `tenacity`、`shortuuid`、`litellm`、`dirhash`、`claude-agent-sdk`、`supabase`、`openai==2.6.1`。
2. 在通用 fully-async tmux 启动脚本里新增 `INSTALL_HARBOR_RUNTIME_DEPS` 开关。默认 `0`，不影响非 Daytona/旧流程。
3. Daytona smoke wrapper 默认设置：

```bash
INSTALL_HARBOR_RUNTIME_DEPS=1
INSTALL_DAYTONA=1
```

4. 依赖安装在 launcher/worker 进入训练 wrapper 前执行，monitor 不执行。这样 Ray `HarborTrialWorker` 不管落在哪个节点，都能 import Harbor runtime deps。

后续 checklist 增加一项：

```bash
python3 - <<'PY'
from slime_plugins.harbor.import_utils import ensure_harbor_distribution_metadata
ensure_harbor_distribution_metadata()
import tenacity
import daytona
from harbor.trial.trial import Trial
print("harbor runtime deps ok")
PY
```

必须在每个 train/rollout 节点的实际运行 Python 环境里通过，再启动 Daytona smoke。

修复后同一实例重启 smoke：

```text
run_tag: daytona_partial26_groupserial32_depsfix_20260609_1243
launcher log: /tmp/slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_daytona_partial26_groupserial32_depsfix_20260609_1243_launcher.log
```

已确认：

```text
ray cluster has 4/4 nodes after 6s
Timer update_weights end (elapsed: 13.6s)
starting Harbor fully-async rollout worker
Harbor fully-async rollout 1: target=2 queue_warm=0 max_concurrent_groups=1
group_0 trajectory.json: 32
group_0 result.json at 12:58:32: 2
Daytona remote at 12:58:32: 30 sandboxes / 150GiB
rollout_dumps: 0
```

这说明 `tenacity`/Harbor dependency 问题已经绕过，Daytona trial 能进入执行并 finalize result。下一步继续等 `group_0 result.json >= 26`，验证 partial group cutoff 是否接受前 26 个 trainable trajectory 并清理剩余长尾 sandbox。

## 2026-06-09 32GPU Daytona full89 partial26 64k：训练侧 CUDA OOM

run tag：

```text
daytona_full89_partial26_64k_progress_20260609_1336
```

关键路径：

```text
launcher log: /tmp/slime_full89_32gpu_9b_a1r3_b3n32_trajnorm_fa_daytona_full89_partial26_64k_progress_20260609_1336_launcher.log
base dir: /path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_daytona_2train2rollout_60x_retokenized_full_daytona_full89_partial26_64k_progress_20260609_1336
trial dir: /path/to/workspace/slime_harbor_trials/tb2_full89_32gpu_9b_daytona_2train2rollout_60x_retokenized_full_daytona_full89_partial26_64k_progress_20260609_1336
```

现象：

```text
ray cluster has 4/4 nodes after 6s
Timer update_weights end
Harbor fully-async rollout started
rollout_dumps/1.pt 38M
rollout_dumps/2.pt 21M
```

partial group cutoff 是生效的：

```text
group 0/1 accepted 26/32 for rollout 1
group 2/3 accepted 26/32 for rollout 2
```

fatal 发生在 train，不是 Daytona sandbox 创建，也不是 rollout：

```text
RayTaskError(OutOfMemoryError)
ray::MegatronTrainRayActor.train()
torch.OutOfMemoryError: Tried to allocate 56.96 GiB.
GPU 6 total 95.00 GiB, free 49.40 GiB, process already 45.57 GiB
```

直接原因是 64k 上限允许单个 train sample 太长。rollout 1 进入训练时已经能看到平均 response/total length 非常高：

```text
rollout/response_lengths: 54480.301886792455
rollout/total_lengths: 55921.92452830189
```

这解释了为什么 dynamic batch size 没救下来：dynamic batch 可以减少同一 microbatch 里的样本数，但不能把一个 55k token 的单样本拆到 batch size 1 以下。也就是说，`MAX_TOKENS_PER_GPU=32768` 这类 token batch cap 只能约束 batch packing，不能自动截短一个已经超过 cap 的 sample。

处理：

1. Daytona 32GPU smoke/full wrapper 的 `HARBOR_MAX_TRAIN_TOTAL_TOKENS` 默认从 `65536` 改成 `32768`。
2. 下一次 full run 先验证 32768 retokenized train sample 不再触发同类 Megatron forward OOM。
3. 如果仍要训练 64k 级别样本，需要单独启 context parallel 或做 sample-level chunking；不能只依赖 dynamic batch。

## 2026-06-10 Daytona new key quota probe：实际上限是 30GiB disk / 10 CPU

背景：

用户提供了新的 Daytona key，希望验证是否能同时 host 64 个 sandbox。为避免用 Harbor/Ray/SGLang 全链路探 quota，新增轻量探测脚本：

```text
examples/harbor_terminal_rl/daytona_capacity_probe.py
```

脚本行为：

1. 用 Daytona SDK 直接创建轻量 `ubuntu:22.04` sandbox。
2. 每个 probe sandbox 都带 `slime-capacity-*` name prefix 和 `slime_probe=daytona_capacity` label。
3. 每轮创建后用 REST 列远端状态。
4. 默认退出时只删除自己创建的 probe sandbox。

5GiB/1CPU 探测：

```text
python3 examples/harbor_terminal_rl/daytona_capacity_probe.py \
  --targets 64 \
  --disk-gib 5 \
  --create-parallelism 16 \
  --cleanup-first
```

结果：

```text
target_result target=64 requested=64 created=6 live_tracked=6 failed=58
DaytonaValidationError: Failed to create sandbox: Total disk limit exceeded. Maximum allowed: 30GiB.
remote_probe_state target=64 remote_count=6 remote_disk_gib=30
cleanup_final count=6 errors=0
```

1GiB/1CPU 补充探测：

```text
python3 examples/harbor_terminal_rl/daytona_capacity_probe.py \
  --targets 30,31 \
  --disk-gib 1 \
  --create-parallelism 16 \
  --cleanup-first
```

结果：

```text
target_result target=30 requested=30 created=10 live_tracked=10 failed=20
DaytonaValidationError: Failed to create sandbox: Total CPU limit exceeded. Maximum allowed: 10.
remote_probe_state target=30 remote_count=10 remote_disk_gib=10
cleanup_final count=10 errors=0
```

最后确认远端没有残留：

```text
remote candidates=0 disk_gib=0 prefix=None
summary candidates=0 cleaned=0 errors=0
```

结论：

1. 这个新 key 不是之前假设的 300GiB org quota，而是 `30GiB disk` 和 `10 CPU`。
2. 当前 Harbor Daytona 配置是 `override_storage_mb: 5120`，也就是每个 sandbox 5GiB；在这个 key 下最多只能同时跑 `min(30/5, 10/1)=6` 个 sandbox。
3. 这个 key 不能用于 60/64 sandbox 的 full89 32GPU 训练，否则 rollout 侧会立刻 quota failure，训练 GPU 会长期 idle 并再次触发低利用率查杀。
4. 要继续跑 60/64 并发，需要换回有约 300GiB quota 的 Daytona key/org，或者升级这个 key 所属 org 的 limits；降低到 6 并发虽然能跑通功能，但不适合 32GPU 利用率。

## 2026-06-15 96GPU 27B TerminalWorld full1530：Daytona sandbox 泄漏撞 RAM 配额 → rollout 凑不齐 group 硬崩

实验：`run_terminalworld_full1530_96gpu_27b_daytona_n20_partial16_fully_async.sh`，RUN_TAG=`tw1530_20260614_160257`。
数据：TerminalWorld 1530 自包含 Dockerfile 任务。配置：n=20、partial=16、`rollout_batch_size=16` → 每步 256 条轨迹、320 并发、fully_async、binary reward、entropy off。

### 现象

训练正常推进到 **step 20**（两个 checkpoint `iter_0000009` / `iter_0000019` 已落盘），随后卡住、日志增长放缓，最终 launcher 写 `.stop_workers` marker、Ray job 退出。fatal：

```text
RuntimeError: Harbor fully-async rollout could not produce enough complete trainable groups:
  15/16 groups after 160 completed group tasks.
  Increase `harbor_rollout_group_attempt_multiplier`, reduce failing tasks, or relax
  `harbor_require_full_rollout_groups`.
  (slime_plugins/harbor/fully_async_rollout.py:362)
```

崩溃前日志里大量：

```text
trial.py:1315 - Trial slime-r21-g605-s0-i12100 failed: Failed to create sandbox:
  Total memory limit exceeded. Maximum allowed: 1000GiB.  [repeated 15x across cluster]
```

### 直接原因

不是 CUDA OOM、不是训练侧 batch 形状、不是 Daytona 镜像/scoring 问题。是 **Daytona sandbox 泄漏累积撞 RAM 配额**，再被 `require_full_rollout_groups=true` 放大成硬崩。三个缺陷叠加（全部在 `harbor/src/harbor/environments/daytona.py`）：

1. **创建竞态丢引用**（`_create_sandbox`，旧 `timeout=30`）。`asyncio.shield(create_task)` 被取消时（oversampling 砍尾部 trial 触发 `CancelledError`），用 `wait_for(create_task, timeout=30)` 兜底捕获 sandbox 引用。但 S3 build-context 拉取慢（日志实测重试 56–60s）> 30s → `wait_for` 超时 → `self._sandbox` 留 `None`，而 Daytona 服务端最终仍把 sandbox 建出来了。

2. **`stop()` 跳过删除**（`_DaytonaDirect.stop` / `_DaytonaDinD.stop`）。`if not env._sandbox:` 分支只打 `logger.warning("Sandbox not found. Please build the environment first.")`（旧 `daytona.py:327` / `daytona.py:673`）然后 `return`，不做任何删除。日志里这个 warning 出现 **1815 次** = 1815 次漏删。

3. **`auto_delete_interval_mins=0` 救不了**。其语义是“stop 时立即删”，不是“禁用自动回收”。泄漏的 sandbox 从未收到 stop 信号，所以 Daytona 也不会自动回收它们。

`harbor_num_concurrent_trials: 320` 的 `asyncio.Semaphore` 只限制**并发协程数**，不限制 Daytona 上**实际存活的 sandbox 数**。漏删的 sandbox 不占协程但占 Daytona RAM → 无上限累积。清理脚本 dry-run 实测崩溃后残留 **538** 个孤儿（远超 320）。538 撞 RAM 1000GiB（注意：是内存配额，不是我们一直盯的 disk 5000GiB 配额）→ rollout 21 某个 group 反复建不出 sandbox，160 次尝试后仍 15/16 → `require_full_rollout_groups=true` 要求 16/16 → 抛 `RuntimeError`。

诊断佐证（崩溃日志 grep）：

| 签名 | 次数 | 含义 |
| --- | --- | --- |
| `daytona.py:327/673 Sandbox not found`（stop 跳过删除） | 1815 | 真泄漏源 |
| `Keeping sandbox alive delete=False` | 0 | `delete: true` 配置正确，不是这条路径 |
| `Error stopping sandbox`（删除 API 调用失败） | 3 | 删除 API 本身可靠，不是 API 问题 |
| 清理脚本 dry-run 残留孤儿 | 538 | 实际存活远超 320 并发上限 |

### 已采取修复

治本（改 `harbor/src/harbor/environments/daytona.py` 源码）+ 防御（yaml）+ 安全网（cron），并发保持 320 不降。

1. **创建时给 sandbox 打可查询 label**（`_DaytonaDirect.start` 3 处 + `_DaytonaDinD.start` 2 处 create-params 都加 `labels=sandbox_labels`）。新增 `_harbor_sandbox_labels()` 返回 `{"harbor_managed": "1", "harbor_session": <trial_name>}`；`session_id == config.trial_name == slime-r{rollout}-g{group}-s{sample}-i{idx}`，每个 trial 唯一，作为按 label 查删的锚点。

2. **`stop()` 兜底按 label 查删孤儿**。两个 `stop()` 的 `if not env._sandbox:` 分支改为调用新增的 `_delete_orphan_sandboxes_by_label()`：拿 client → `list(labels={"harbor_session": ...})` → 逐个 `delete()`。即使创建竞态丢了 `self._sandbox` 引用，也能按 session label 找回删除。新增 `_list_sandboxes_by_label()` 包装器，**版本无关**地处理两种 SDK 形态（见下方“关键插曲”）。

3. **创建竞态超时 30s → 90s**（`_create_sandbox` 的 `wait_for(create_task, timeout=90)`），覆盖实测的 56–60s S3 重试，减少丢引用概率。

4. **yaml `auto_stop_interval_mins` + `auto_delete_interval_mins`**（`environment.kwargs`，经 `factory.py:239 **config.kwargs` 进 ctor，纯 yaml 无需改源码）。⚠️ **更正（2026-06-21）：原方案只设 `auto_delete_interval_mins: 60` 是无效的纵深防御**。源码 `daytona.py:881-922` 证实 `auto_delete_interval_mins` 的语义是「**stop 之后**多少分钟删」，而 `auto_stop_interval_mins` 默认 **0 = 永不自动 stop**。泄漏的 sandbox（`stop()` 从未跑成功）永远停在 `state=started` → 永不 auto-stop → auto_delete 计时器永不启动 → 永久存活。tb2of_raw_20260620_062054 跑完后正是这样残留 308 个孤儿（224 个本 run r27 收尾尾巴 `autoStopInterval=0`+84 个 16h 古老的 `autoDeleteInterval=0`），全靠手删。**正确做法：两个都设** —— `auto_stop_interval_mins` 设到 > agent wall（`override_timeout_sec`，30min→45 / 60min→90，保证运行中 trial 不被误停）+ `auto_delete_interval_mins: 10`（stop 后短宽限）。健康 trial 仍由 `stop()` 立即删；这两个旋钮只兜泄漏。已修 4 个 yaml。

   **另一个独立坑（同日发现）：Daytona list API 是 cursor 分页、每页上限 100、最新优先**。`GET /api/sandbox`（或单次 curl）最多返回 100 + 一个 `nextCursor`，会静默截断。整个 tb2 run 期间只查第 1 页的巡检 Bash 一直报「TOTAL=100, ZOMBIES=0」，而 200+ 个老 sandbox 全在后续页里没被看到 → 泄漏被完全掩盖。**必须用 `nextCursor` 翻到空页**。已加只读全分页普查：`daytona_cleanup_slime_sandboxes.py --action audit`（exit code = zombie 数 = started 且 idle≥15min），单测 `tests/test_daytona_cleanup_audit.py`（5 passed）。

5. **yaml `harbor_require_full_rollout_groups: false`**（原 true）+ **`harbor_rollout_group_attempt_multiplier: 8 → 12`**。凑不齐 16/16 时用部分批（≥`partial_group_min_units`）训练这一步，而不是抛 `RuntimeError` 崩溃 —— 直接消除这次的崩溃点。

6. **周期清扫 cron（安全网）**。训练期间每 20 分钟后台跑 `daytona_cleanup_slime_sandboxes.py --action delete-all-rest --created-before <now-45min>`，只删超过 45min（> 30min trial timeout）的孤儿，绝不误删运行中的 trial。

### 关键插曲：SDK 版本差异（差点二次踩坑）

`stop()` 兜底最初按 harbor venv（daytona 0.168.0）的 `list()` 写成 `await daytona.list(labels=...)` + `.items`。但**训练实际运行用的是 system site-packages 的 daytona 0.187.0**，其 `list()` 是 **async generator**（`async for sandbox in daytona.list(ListSandboxesQuery(labels=...))`），不是 awaitable，也没有 `.items`。直接跑会抛 `TypeError: object async_generator can't be used in 'await' expression`。

修正为版本无关的 `_list_sandboxes_by_label()`：用 `inspect.isasyncgenfunction` 判别，async-gen 路径用 `ListSandboxesQuery` + `async for`，旧路径用 `await` + `.items`，都附加客户端侧 label 过滤兜底。

教训：改 Harbor sandbox 生命周期前，务必确认**训练运行时实际加载的 daytona 版本**（system 0.187，不是 harbor venv 0.168），并用独立 SDK 脚本（`validate_daytona_label_fix.py`）对真实 API 验证 create-with-label → list-by-label → delete → confirm-gone 全链路，再重训。

### 验证与续训

- SDK 层验证（`validate_daytona_label_fix.py`）：create 带 label → list 查到 → delete → 复查 0，**PASS**。
- 从 `iter_0000019` 续训（RUN_TAG=`tw1530resume_20260615_100330`）：`--load` 自动指向旧 SAVE_DIR，日志确认 `successfully loaded checkpoint ... at iteration 19`；`--ref-load` 仍是原始基座（KL 锚不变）。
- 续训实测：step 20→21→22 持续前进，`global_batch_size` 为 206/236/244（`require_full=false` 防御按设计生效，partial 批训练，不再崩溃）；`could not produce enough` 复发 = 0；旧泄漏签名 `daytona.py:327/673` = 0；新兜底 `Sandbox reference missing on stop`（`daytona.py:335`）正常 firing；清理脚本实测总存活 sandbox 稳定在 **216–219 < 320**，不再累积。
- 仍偶发 `Total memory limit exceeded`（每轮 11–25 次），但被 oversampling + `require_full=false` 吸收，step 持续前进、0 崩溃。判断：部分 TerminalWorld 重型任务单 sandbox 内存需求高，并发瞬时峰值触顶；非泄漏（总数稳定）。下一轮若持续高频，考虑给高内存任务限流或微降并发。

### 涉及文件

- `harbor/src/harbor/environments/daytona.py`（治本：label 创建 + `stop()` 兜底查删 + 超时 30→90 + `_harbor_sandbox_labels` / `_delete_orphan_sandboxes_by_label` / `_list_sandboxes_by_label` 三个新方法）
- `examples/harbor_terminal_rl/harbor_terminalworld_full1530_96gpu_27b_daytona_n20_partial16_fully_async.yaml`（`auto_delete_interval_mins: 60`、`harbor_require_full_rollout_groups: false`、`harbor_rollout_group_attempt_multiplier: 12`；`harbor_num_concurrent_trials: 320` 不变）
- `examples/harbor_terminal_rl/validate_daytona_label_fix.py`（新增 SDK 验证脚本）
