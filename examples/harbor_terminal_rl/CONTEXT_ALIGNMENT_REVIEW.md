# Rollout / Train Context Alignment Review

本文记录 2026-05-30 对 Harbor Terminus2 + slime 训练链路中 context length 对齐问题的代码检查。

## 结论

当前代码里确实存在两种配置状态：

- 32GPU 9B turn_grouped：rollout/SGLang 是 128k，但 actor train 默认仍是 32k。
- 64GPU 27B turn_grouped：rollout/SGLang/train 都是 128k，当前入口已经对齐。

这不是 slime 把 `rollout_max_context_len` 偷偷改成 32k，而是 32GPU wrapper 继承了 base 脚本里的
`SEQ_LENGTH=32768` 和 `HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768`。

## 参数链路

### Rollout 侧

入口脚本把：

```bash
ROLLOUT_MAX_CONTEXT_LEN=131072
SGLANG_CONTEXT_LENGTH=131072
SGLANG_MAX_PREFILL_TOKENS=131072
```

传给 slime CLI：

```bash
--rollout-max-context-len
--sglang-context-length
--sglang-max-prefill-tokens
```

Harbor rollout adapter 在 `slime_plugins/harbor/rollout.py` 里把
`args.rollout_max_context_len` 注入 Terminus2 LLM：

```python
llm_kwargs.setdefault("max_context_tokens", max_context_tokens)
```

`slime_plugins/harbor/llm.py` 的 `SlimeSGLangLLM.get_model_context_limit()` 会返回这个值。
因此 Terminus2 认为模型 context limit 是 128k。

### Train 侧

训练侧有两个独立 cap：

```bash
SEQ_LENGTH
HARBOR_MAX_TRAIN_TOTAL_TOKENS
```

`SEQ_LENGTH` 最终进入 Megatron：

```bash
--seq-length
--max-position-embeddings
```

`HARBOR_MAX_TRAIN_TOTAL_TOKENS` 被传入 Ray runtime env，然后
`slime_plugins/harbor/sample_postprocess.py` 读取：

```python
value = getattr(args, "harbor_max_train_total_tokens", None)
if value in (None, ""):
    value = os.environ.get("HARBOR_MAX_TRAIN_TOTAL_TOKENS")
```

sample hook 会在 Harbor rollout 返回后、转 train data 前裁剪样本：

```python
sample.tokens = list(tokens[-keep:])
```

因此 rollout 看到的 prompt history 可以是 128k，而 train data 可能只保留最后 32k。

## 当前入口状态

### 32GPU 9B turn_grouped

入口：

```text
examples/harbor_terminal_rl/run_full89_grpo_32gpu_max100_n16_5epoch_high_concurrency_turn_grouped.sh
```

该 wrapper 设置了：

```bash
ROLLOUT_MAX_RESPONSE_LEN=4096
USE_ROLLOUT_LOGPROBS=1
CUSTOM_REWARD_POST_PROCESS_PATH=slime_plugins.harbor.partial_reward.trajectory_grouped_progress_rewards
```

但没有设置：

```bash
SEQ_LENGTH
HARBOR_MAX_TRAIN_TOTAL_TOKENS
```

所以它继承 base 脚本：

```text
examples/harbor_terminal_rl/run_full89_grpo_32gpu_max100_n16_5epoch.sh
```

里的默认：

```bash
ROLLOUT_MAX_CONTEXT_LEN=131072
SGLANG_CONTEXT_LENGTH=131072
SGLANG_MAX_PREFILL_TOKENS=131072
SEQ_LENGTH=32768
HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768
ROLLOUT_ALL_SAMPLES_PROCESS_PATH=slime_plugins.harbor.sample_postprocess.cap_total_tokens_in_samples
ROLLOUT_DATA_POSTPROCESS_PATH=slime_plugins.harbor.sample_postprocess.cap_total_tokens_in_rollout_data
```

结论：32GPU 9B turn_grouped 当前是 `rollout 128k / train 32k`。

### 64GPU 27B turn_grouped

入口：

```text
examples/harbor_terminal_rl/run_qwen35_27b_full89_64gpu_max100_n16_5epoch_high_concurrency_turn_grouped.sh
```

该 wrapper 显式覆盖：

```bash
ROLLOUT_MAX_CONTEXT_LEN=131072
SGLANG_CONTEXT_LENGTH=131072
SGLANG_MAX_PREFILL_TOKENS=131072
SEQ_LENGTH=131072
HARBOR_MAX_TRAIN_TOTAL_TOKENS=131072
CP_SIZE=2
LOG_PROBS_CHUNK_SIZE=32
CALCULATE_PER_TOKEN_LOSS=0
```

它会覆盖 27B base 脚本里的 32k 默认值。

结论：64GPU 27B turn_grouped 当前是 `rollout 128k / train 128k`。

## 为什么 128k rollout + 32k train 有问题

如果一个 turn 的 sample 是：

```text
tokens = prompt_ids + completion_ids
response_length = len(completion_ids)
```

rollout 阶段 completion 是在完整 prompt history 下采样的。例如 prompt 接近 80k。

如果训练前被裁到 32k：

```python
sample.tokens = tokens[-32768:]
```

response suffix 仍然保留，但 prompt prefix 变了。于是：

- rollout logprobs 来自 128k prompt context；
- train forward/logprob 来自 32k prompt context；
- reward 来自完整 terminal trajectory；
- PPO/GRPO loss 看到的是被截短后的 context。

这会破坏 on-policy 假设，尤其在 `--use-rollout-logprobs` 打开时更隐蔽：

- `old_log_probs` 直接用 `rollout_log_probs`；
- 这些 logprobs 是 128k context 下算的；
- 当前 actor 的 `log_probs` 是 32k context 下算的；
- `train_rollout_logprob_abs_diff` 在某些路径下可能变成 0，但不代表语义对齐。

因此真正的对齐不是“打开 `--use-rollout-logprobs`”，而是 rollout 生成时的 context 和 train
forward 时的 context 必须一致，或者不要训练被截断过 context 的 turn。

## 可选修法

### 方案 A：真 128k 训练

把 32GPU 9B 也改成：

```bash
SEQ_LENGTH=131072
HARBOR_MAX_TRAIN_TOTAL_TOKENS=131072
```

但 32GPU 当前只有 1 个 actor node，base 设置是 PP=8/CP=1。9B 128k train 很可能显著变慢或 OOM。
如果要在 32GPU 上做真 128k，建议改成 2 个 actor nodes：

```bash
ACTOR_NUM_NODES=2
ROLLOUT_NUM_GPUS=16
PIPELINE_MODEL_PARALLEL_SIZE=8
CP_SIZE=2
SEQUENCE_PARALLEL=1
SEQ_LENGTH=131072
HARBOR_MAX_TRAIN_TOTAL_TOKENS=131072
LOG_PROBS_CHUNK_SIZE=32
CALCULATE_PER_TOKEN_LOSS=0
```

代价是 rollout GPU 从 24 降到 16，rollout 吞吐会下降，但训练语义对齐。

### 方案 B：32k 对齐训练，只训练短 context turn

保留 rollout 128k，让 agent 仍能解决长任务；但对于 train cap 之外的 turn，不做 policy update：

```text
if original_total_length > HARBOR_MAX_TRAIN_TOTAL_TOKENS:
    remove_sample=True 或 loss_mask 全 0
```

这避免用 32k context 去训练 128k context 下采样出来的 response。优点是显存安全；缺点是后期长
context turn 不会被训练，只保留早期/短上下文 turn 的信号。

这个方案需要新增一个 Harbor sample postprocess hook，不能继续用当前
`cap_total_tokens_in_samples` 直接裁 prompt 后训练。

### 方案 C：rollout 也限制到 32k

把 rollout 和 SGLang 也设成：

```bash
ROLLOUT_MAX_CONTEXT_LEN=32768
SGLANG_CONTEXT_LENGTH=32768
```

这会语义对齐，但对 max_step=100 的 terminal agent 不友好。当前 Terminus2 配置里
`enable_summarize=false`，history 很容易超过 32k；一旦超过，trial 会更早失败。

如果走这个方案，需要重新打开 summarization 或实现 deterministic history trimming。Harbor 注释里也提示：
summarization 会让 rollout details 不完整，因此这不是当前最干净的训练路径。

## 建议

短期：

1. 64GPU 27B turn_grouped 保持当前 128k/128k，对齐是正确的。
2. 32GPU 9B 不要继续静默使用 `rollout 128k / train 32k`。
3. 如果要用 32GPU 做语义正确的训练，优先试 2 actor nodes + CP=2 的 128k train。
4. 如果 32GPU 显存/速度扛不住 128k，再实现“over-cap turn 零 loss / remove_sample”的 hook，而不是裁到 32k 继续训练。

需要重点看 W&B：

```text
harbor/train_context_trimmed_sample_ratio
harbor/train_context_max_original_total_length
harbor/train_context_max_trimmed_total_length
harbor/train_response_trimmed_sample_ratio
train/train_rollout_logprob_abs_diff
rollout/ppo_kl 或 train/ppo_kl
```

如果 `train_context_trimmed_sample_ratio > 0` 且这些样本仍参与 loss，就说明仍存在 context mismatch。
