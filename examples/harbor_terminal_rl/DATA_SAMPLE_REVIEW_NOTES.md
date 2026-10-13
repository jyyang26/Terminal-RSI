# Harbor/Slime Rollout Data 人工 Review 记录

本文记录人工阅读 `rollout_dumps/*.pt` 转出的 trajectory JSON 后发现的问题、判断依据和后续修改建议。

## 2026-05-30：Trial-level timeout 被误读为 turn-level truncated

### 样本

- dump: `rollout_dumps/1.pt`
- 导出文件: `rollout_dumps/trajectories_10_full_text.jsonl`
- trial: `slime-r1-g3-s10-i58`
- task: `terminal-bench/build-cython-ext`
- sample mode: `turn_grouped`
- reward: `0.0`
- turn 数: 29

### 人工观察

第 0 个 turn：

- `status = truncated`
- `finish_reason = stop`
- `response_length = 217`
- response 是完整 JSON action，并正常以 `<|im_end|>` 结束。

第 28 个 turn：

- `status = truncated`
- `finish_reason = stop`
- `response_length = 187`
- terminal output 显示 pytest 已经 `17 passed, 1 failed`。
- agent 继续提出合理下一步：检查 `pyknotid/spacecurves/` 下实际模块名。

人工判断：这两个 turn 自身都不是生成截断。`finish_reason=stop` 说明当前 LLM call 正常结束。

### 原始 trial 结尾

查看原始 trial 目录：

```text
/path/to/workspace/slime_harbor_trials/
  tb2_full89_32gpu_max100_5epoch_resp4096_ctx128k_rb16_os32_conc384_pn128_turngrouped/
  slime-r1-g3-s10-i58/
```

目录里存在：

```text
agent/episode-29/prompt.txt
exception.txt
result.json
trial.log
```

`exception.txt` 关键报错：

```text
harbor.trial.trial.AgentTimeoutError: Agent execution timed out after 900.0 seconds
```

也就是说，这条 trajectory 并不是第 0 或第 28 个 turn 被截断，而是在准备继续第 29 个 turn 时，整条 agent execution 撞到了 900 秒 wall-clock timeout。

### 根因

当前 adapter 里 `sample.status` 是 trial-level final status：

```python
if exception_info is not None:
    if "Timeout" in exception_type:
        return Sample.Status.TRUNCATED
```

在 `turn_grouped` 模式里，同一条 trajectory 展开的每个 turn sample 都继承这个 trial-level status：

```python
sample.status = status
```

因此，一个 trajectory 只要最终因为 agent timeout 失败，它前面所有正常 `finish_reason=stop` 的 turn 也都会显示成 `status=truncated`。这会导致：

- 人工读样本时误以为每个 turn 都被截断。
- W&B 的 `rollout/truncated_ratio` 被显著放大。
- `rollout/truncated_ratio` 不再等价于单步 LLM response hitting max tokens。

### `max_turns=100` 和 `900s timeout` 的关系

这条样本只有 29 个 turn，远低于我们配置的：

```yaml
max_turns: 100
```

但它先撞到了 Harbor trial 的 agent execution wall-clock timeout：

```text
Agent execution timed out after 900.0 seconds
```

这说明 `max_turns=100` 只是 step 数上限；900s 是另一条独立的总时间上限。对于 terminal-bench 这类任务，一个 step 可能包含：

- LLM 请求和 JSON parsing。
- tmux/Apptainer terminal command 执行。
- pip install / build / pytest 等长命令。
- terminal output dump 和下一轮 prompt 构造。

因此 29 个 turn 用完 900s 是合理可能的。`max_turns=100` 没有机会真正生效。

### 为什么这是有价值的 near-solve 样本

第 28 个 turn 的 terminal output 显示：

```text
17 passed, 1 failed
ModuleNotFoundError: No module named 'pyknotid.spacecurves.complex128ity'
```

模型下一步计划是检查实际 module 名字，属于合理 debugging 路径。人工判断这类样本可能“差一点就能解出来”，但被 900s timeout 提前 kill 了。

### 修改建议

#### 1. 提高 agent execution timeout

对 `max_turns=100` 的实验，900s 太紧。建议下一轮显式设置 agent timeout，先试 1800s 或 2400s：

```yaml
harbor_trial_defaults:
  agent:
    override_timeout_sec: 2400
```

或者：

```yaml
harbor_trial_defaults:
  agent:
    max_timeout_sec: 2400
```

更推荐 `override_timeout_sec`，语义更明确：为了 100 turn 的 terminal agent rollout，整条 agent execution 给固定 wall-clock budget。

当前 `harbor_trial_remote_timeout_sec=21600`，不是这条样本的瓶颈。

#### 2. 拆分 trial-level status 和 turn-level generation status

导出 JSON 时已经增加了两个字段：

```text
trial_status
turn_generation_status
```

推荐把 adapter 训练数据语义也改成类似结构：

- `sample.status`: 当前 turn 的 generation status。`finish_reason == "length"` 才标 `TRUNCATED`，`finish_reason == "stop"` 标 `COMPLETED`。
- `sample.metadata["harbor_trial_status"]`: 整条 trial 最终状态，比如 `completed` / `truncated` / `failed`。
- `sample.metadata["harbor_trial_exception_type"]`: 记录 `AgentTimeoutError` 等异常类型。

这样 W&B 的 `rollout/truncated_ratio` 才能重新接近“生成被 max_new_tokens 截断”的含义。

#### 3. 单独记录 timeout 指标

保留或新增 Harbor-specific metrics：

```text
harbor/trial_timeout_ratio
harbor/trial_timeout_trajectory_ratio
harbor/agent_timeout_ratio
harbor/turn_generation_truncated_ratio
```

其中：

- `trial_timeout_ratio`: trial-level timeout。
- `turn_generation_truncated_ratio`: 单个 LLM turn 的 `finish_reason=length`。

不要再只依赖 `rollout/truncated_ratio` 判断截断问题。

### 风险和代价

提高 agent timeout 会加重 rollout 长尾：

- 单个 bad trajectory 占用 TrialWorker/Apptainer 更久。
- 首个 rollout batch latency 可能上升。
- GPU 利用率可能更依赖 oversampling、FIRST_COMPLETED、tail cancel、dynamic filter。

但人工样本显示，900s 会砍掉 near-solve trajectory。后续应该用更长 timeout 配合 async/oversampling，而不是用 900s 直接截断 100-turn agent。

## 2026-05-30：`prefix_mismatch_turns` 是导出重构问题，不是 turn_grouped 训练问题

### 现象

人工查看：

```text
/path/to/workspace/code/slime/examples/harbor_terminal_rl/trajectories_10_full_text.jsonl
```

第一个 example 里：

```json
"prefix_mismatch_turns": [1, 2, 3, ..., 28]
```

看起来像每个 turn 都有 mismatch。

### 字段来源

这个字段不是 slime 训练数据字段，而是 `export_rollout_trajectories.py` 为了人工可读导出临时加的诊断字段。

导出脚本尝试用下面的假设重构整条 trajectory：

```text
next_prompt_tokens == previous_prompt_tokens + previous_completion_tokens + observation_delta_tokens
```

如果这个 token-level prefix 假设不成立，就把对应 turn 记进 `prefix_mismatch_turns`。

### 为什么会 mismatch

Terminus2 每一轮不是简单把上轮 prompt 和 completion 原样拼接成下一轮 prompt。它会重新构造 message history，再通过 tokenizer/chat template 重新编码：

- 插入 role/chat-template token。
- 插入新的 terminal observation。
- 插入 parser warning，例如：

  ```text
  Previous response had warnings:
  WARNINGS: - Extra text detected before JSON object
  - Extra text detected after JSON object
  ```

- 可能存储的是 parser 清洗后的 assistant message，而不是模型 raw completion token 的逐字节副本。
- `<think>`、`<|im_start|>`、`<|im_end|>` 等 chat-template 细节也可能导致 token 序列不再是严格前缀关系。

因此，第 N 轮 prompt 在语义上包含了历史，但 token 序列通常不是第 N-1 轮 `prompt + response` 的严格 prefix extension。

### 是否影响训练

对当前 `turn_grouped` 训练：**不直接影响**。

`turn_grouped` adapter 的训练样本构造是每个 turn 独立：

```python
sample.tokens = prompt_ids + completion_ids
sample.response_length = len(completion_ids)
sample.loss_mask = [1] * sample.response_length
```

它不要求下一轮 prompt 是上一轮 prompt+completion 的前缀追加。因此 `prefix_mismatch_turns` 不会破坏当前 turn-level 训练。

真正会受影响的是 strict `trajectory` 模式。当前 `trajectory` adapter 会检查：

```python
next_prompt.startswith(previous_tokens)
```

一旦不满足就报错：

```text
Cannot build whole-trajectory slime sample because the next Harbor prompt is not a prefix extension
```

这也是我们之前 trajectory-level 样本构造不稳定、后来切到 `turn_grouped` 的核心原因之一。

### 如何阅读导出的 JSON

当前导出的字段可靠性：

- `turns[].response_text`：可靠，来自该 turn 的 completion token suffix。
- `turns[].prompt_length` / `response_length` / `total_length`：可靠，来自 slime Sample。
- `turns[].finish_reason`：可靠，来自 SGLang meta_info。
- `turns[].observation_delta_text`：只适合人工阅读，是导出脚本根据 prompt 差异做的启发式近似；如果 `prefix_mismatch_turns` 非空，不应把它当成严格 token-level delta。
- `trajectory_text`：只适合人工快速 review，不应当成 actor 实际训练 token 序列。

如果要做完全可靠的人工 review，建议后续导出时加：

```bash
--include-turn-prompts
```

这样每个 turn 都会包含完整的 `prompt_text`，虽然文件会更大，但不会依赖 prefix delta 重构。

### 修改建议

1. 把导出 JSON 中的字段改名为更明确的：

   ```text
   reconstruction_prefix_mismatch_turns
   ```

   避免误解成训练数据 mismatch。

2. 如果未来要重新做 trajectory-level update，需要从 Terminus2/Harbor 侧收集结构化 message history 或 full chat-template tokenization，并生成 role-aware loss mask；不能靠 `prompt_token_ids/completion_token_ids` 做简单前缀拼接。

3. 当前 `turn_grouped` 路线可以继续使用，但要把 advantage/reward 语义保持在 trajectory group 级别。

## 2026-05-30：多文件 trajectory review 暴露的训练数据信号问题

### 数据来源

人工 review 文件：

```text
trajectories_mixed_80_full_text.jsonl
trajectories_completed_40_full_text.jsonl
trajectories_truncated_40_full_text.jsonl
trajectories_failed_40_full_text.jsonl
```

完整 rollout dump 组级统计脚本：

```text
analyze_rollout_group_signals.py
```

统计范围：

```text
rollout_dumps/1.pt ... rollout_dumps/5.pt
turn_samples = 24226
GRPO groups = 80
group size = 16 trajectories
```

### 关键统计

完整 80 个 GRPO group 的 reward 方差：

```text
all_zero groups = 42
all_one groups  = 3
mixed groups    = 35
```

也就是说，`45/80 = 56.25%` 的 group 是零 reward 方差。对于 GRPO group-normalized reward 来说，这些 group 基本没有有效 advantage 信号，却消耗了大量 rollout 时间。

status pattern 也显示大量 trial-level timeout：

```text
truncated:16 groups = 21
completed:1,truncated:15 groups = 9
completed:2,truncated:14 groups = 7
```

trajectory 长度：

```text
trajectory_turns min/mean/max = 4 / 18.93 / 60
trajectory_max_total_length min/mean/max = 1859 / 18366 / 32768
```

### 从导出 JSON 看到的样本质量问题

#### 1. 零方差 group 太多

`mixed_80` 中前 5 个 group：

```text
build-cython-ext:              16/16 reward=0
adaptive-rejection-sampler:    16/16 reward=0
chess-best-move:               16/16 reward=0
caffe-cifar-10:                16/16 reward=0
configure-git-webserver:       1/16 reward=1, 15/16 reward=0
```

这说明很多 task 在当前模型+timeout+env 设置下 pass rate 接近 0。这样的 task 对 GRPO 来说几乎只产生无效 batch。

#### 2. 二值终局 reward 过稀疏，near-solve 全部当 0

例如 `build-cython-ext` 多条样本已经能跑到 pytest `17 passed, 1 failed`，但 reward 仍然是 0。所有早期正确操作和最后失败操作拿同一个 0 reward。

这会带来 credit assignment 问题：

- near-solve 和完全失败没有区别。
- turn_grouped 下，trajectory reward 会复制到每个 turn，前面很多正确 turn 也被当作负样本。
- 对长任务来说，二值终局 reward 太稀疏。

建议引入 partial reward：

```text
pytest passed / total
visible verifier score
task-specific progress signal
format/parser penalty
timeout penalty
```

#### 3. `completed`/`failed`/`truncated` 与 reward 不一致

导出数据中出现：

```text
completed + reward=0
failed    + reward=1
truncated + reward=1
```

例子：

- `completed_40`: 20 条 reward=0，20 条 reward=1。
- `failed_40`: 32 条里有 4 条 reward=1。
- group 统计里 `build-pmars` 有 2 条 reward=1 但 group status 全是 `truncated`。
- `cobol-modernization` 有 13 条 reward=1，但大部分 status 是 `truncated`。

因此 status 不能直接当成功/失败监督，也不能直接用于过滤训练样本。当前最可信的成功信号仍然是 verifier reward。

#### 4. parser/format 噪声很重

`mixed_80`：

```text
parser warning observations = 290 / 1767 turns
responses starting with code fence = 197
responses with prose before JSON = 140
parse json_error = 22
parse unbalanced = 14
```

这些 turn 虽然很多被 parser recovery 继续执行，但会污染训练：

- 模型会学习到 code fence / prose + JSON 的输出格式。
- parser warning 会进入下一轮 prompt，占用 context。
- 格式错误会消耗额外 turn，增加 timeout 概率。

建议：

```text
parser warning penalty
pure JSON format reward
prompt 中明确禁止 code fence 和 JSON 外文本
记录 harbor/parser_warning_turn_ratio
```

#### 5. 训练 context cap 与 rollout context 不一致

导出的 train samples 中很多 `total_length` 达到 `32768`。这说明样本在进入 actor train 前被 `HARBOR_MAX_TRAIN_TOTAL_TOKENS=32768` cap 过。

如果 rollout 生成时使用的是更长上下文，而 actor train 只看到截短后的 32k prompt，就会出现：

```text
rollout logprob: full rollout context 下的 response 概率
train logprob:   clipped train context 下的 response 概率
```

这会破坏 on-policy 假设，并可能解释 `train/train_rollout_logprob_abs_diff` 持续升高。

建议：

- rollout context 和 train context 尽量对齐。
- 如果要训练 128k context，就把 actor seq_length / train cap 提到 128k。
- 如果 actor 只能训 32k，就 rollout 时也使用同样的 history truncation/summarization 策略。

#### 6. command/duration 造成 timeout 长尾

`mixed_80` 中：

```text
commands per turn mean ~= 1.87
commands per turn p95 = 4
commands per turn max = 9
declared command duration sum = 17843s / 80 trajectories
duration p95 = 30s
duration max = 120s
```

长命令、重复 pytest、pip/build、sed 修补失败，会快速吃掉 900s agent timeout。提高 timeout 是必要的，但还需要控制长尾：

```text
long command / repeated test penalty
pytest/test output partial reward
async rollout + zero-variance group filtering
```

### 当前训练效果不佳的最可能原因排序

1. **GRPO 有效 batch 比例低**：56% group 零 reward 方差，基本没有学习信号。
2. **终局二值 reward 太稀疏**：near-solve 轨迹全部 reward=0，turn_grouped credit assignment 很粗。
3. **900s timeout 太紧**：许多 18-30 turn 的 near-solve 被提前 kill，`max_turns=100` 没有真正发挥。
4. **train/rollout context 不一致**：32k train cap 可能导致 logprob mismatch 和 off-policy。
5. **格式噪声污染训练**：code fence、JSON 外文本、parser warnings 比例不低。
6. **status 语义误导**：trial-level timeout 被复制到每个 turn，`rollout/truncated` 不代表 generation truncation。

### 优先修改建议

1. 开启/实现 group-level dynamic sampling filter：丢弃 all-zero/all-one reward group，或 backfill 到 mixed group。
2. 提高 agent timeout 到 `1800s`/`2400s`，配合 async/oversampling 控制长尾。
3. 引入 partial reward，至少对 pytest/visible pass ratio 做 reward shaping。
4. 拆分 `trial_status` 和 `turn_generation_status`，修正 `rollout/truncated` 语义。
5. 对齐 train/rollout context，避免 128k rollout + 32k train cap 的 logprob mismatch。
6. 增加格式 reward/penalty，减少 parser recovery 样本进入训练时的噪声。
