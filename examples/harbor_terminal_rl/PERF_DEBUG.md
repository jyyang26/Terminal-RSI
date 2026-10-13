# Harbor 终端 Agent RL — 性能 Debug 文档

记录 96-GPU Qwen3.5-27B fully-async / trajectory_retokenized 训练的性能诊断结论与可复用的排错经验。

---

## Case 1 — reward 持续下降:长度坍缩 / 熵坍缩

- **Run**: `run96retok2`（96 GPU，12 节点；TP2×PP8×CP2=32 train + 64 rollout；fully-async；`trajectory_retokenized`；ctx 64k；`num_rollout=24`）
- **现象**: `harbor/reward_mean` 在 W&B 上持续下降
- **结论**: 真实的 reward 退化，根因是**长度/熵坍缩**（KL=0 + entropy-coef=0，策略收敛到"提前收手"的低熵捷径解）。训练动力学本身没崩。
- **最终结局**: run 正常跑满 24 个 step 自然收尾（`Job ... succeeded`，`latest_checkpointed_iteration=23`，checkpoint iter_5/7/.../23 全部落盘）。问题是**算法配置**层面，不是工程故障。

### 先排除一个会误导的"假阳性"：`rollout/rewards` ≈ 0 是正常的

不要盯着 `rollout/rewards`（它一直在 0 附近正负跳）——那是 **GRPO 组内归一化后的 advantage**，减完组均值后 mean 必然 ≈ 0（日志里 `advantages ≈ 1e-8`、`kl = 0`）。这条恒为 0 是设计使然，**不是病**。

真正反映"任务做得好不好"的是 **`harbor/reward_mean`** 和 **`rollout/raw_reward`**（原始任务得分）。这两条确实在跌。

### reward 真实下降的数据

> **⚠️ 口径修正（2026-06-08 实测复核）**：本表早期版本写的 `response_length ~24.8k→13.2k` 是凭印象记串的数字，**以 launcher log 实测为准**。长度 metric 的正确名字是 **`response_len/mean`**（单位 token，trajectory_retokenized 模式下是整条轨迹的 response token 总和），实测是 **~12k → ~9k（约 -25%）**，不是腰斩。完整逐 step 实测数据见下方 **Case 2**。

| 阶段 | harbor/reward_mean | response_len/mean | trial_timeout_ratio |
|------|--------------------|-------------------|---------------------|
| rollout 1–7（早） | 0.05–0.37 | ~11.4k–12.7k | 0.01–0.12 |
| rollout 17–23（近） | 0–0.08 | ~8.5k–10.9k | 0–0.05 |

`harbor/reward_mean` 明显下滑（后期多次为 0），`response_len/mean` 同步从 ~12k 降到 ~9k——方向一致，是长度/熵坍缩的初步线索。但**真正最刺眼的信号是单轮 4096 截断比例飙升**，详见 Case 2。

### 诊断：典型的长度坍缩 / 熵坍缩 → reward 退化

看 run 脚本的损失配置（`run_qwen35_9b_terminal_bench2_64h20.sh:433-437`）：

```
--advantage-estimator grpo
--kl-loss-coef 0.00      # 无 KL 约束，不拉回参考模型
--kl-coef 0.00
--entropy-coef 0.00      # 无熵正则
```

**KL=0 + 熵正则=0**，意味着没有任何机制阻止策略熵收缩。在 terminal-agent 任务上，模型正在学到一条"捷径"：单轮输出越来越长（频繁撞 `ROLLOUT_MAX_RESPONSE_LEN=4096` 上限被截断，见 Case 2），同时整条轨迹的有效产出在退化，`raw_reward` 就掉。这是 agentic RL 里很经典的失败模式——**策略不是在变强，而是在往"单轮越写越长、撞上限被截断"的低熵退化方向漂**。

`trial_timeout_ratio`（任务 3600s wall-clock 超时比例）后期反而下降，说明不是"想做但被超时掐死"，问题出在策略本身的输出行为。

### 哪些指标健康 / 不健康

**健康（训练动力学本身没崩）：**
- `grad_norm` 稳定 0.10–0.24，无尖刺
- `train/loss`（pg_loss）0.005–0.01，小而稳
- `kl = 0`（本来就关了）
- Ray / GPU / dump 全程正常，无 OOM、无 node 死

**不健康（问题所在）：**
- ⚠️ `harbor/reward_mean` 持续下降（任务成功率在退化），后期多次为 0
- ⚠️ `harbor/any_step_length_finish_sample_ratio`（单轮撞 4096 上限的样本占比）从 ~5% 飙到后期 ~57%–68%（**最刺眼的信号，详见 Case 2**）
- ⚠️ `response_len/mean` 从 ~12k 降到 ~9k（长度/熵漂移）
- ⚠️ 多个 rollout 出现 `reward_mean = 0.0`（整批 8 组无一成功）

### 建议（配置层面，需重启生效）

按"最可能见效"排序：

1. **加熵正则**：`--entropy-coef 0.001 ~ 0.01`，对抗"在解不动的难任务上塌缩成复读"的退化模式（治本，见 Case 3）。
2. **加一点 KL 约束**：`--kl-loss-coef 0.001`，把策略拉住、别离参考模型太远。
3. **加复读/超长惩罚**：对被 4096 截断（`finish_reason=length`）的样本给负 reward，或检测高 `dup_line_frac` 直接惩罚——直接打掉复读捷径。
4. **降学习率**：当前可能偏大，导致过快坍到捷径解。
5. **⚠️ 不要单纯调高 `ROLLOUT_MAX_RESPONSE_LEN`（4096）**：Case 3 实测证明后期撞 4096 的内容是**冗余复读垃圾**而非有用产出，放宽上限只会让模型复读更多、烧更多算力，不解决 reward。4096 截断是**症状不是病根**。
6. **64k 训练 cap 保持不动**：实测裁剪率 ~0.17%、从不碰 response（见 Case 2），不是瓶颈。

> 复现续训：以 `iter_23` 为起点，加 `--entropy-coef` + 小 `--kl-loss-coef`（+ 可选复读惩罚）开一版对比。**不要**动 `ROLLOUT_MAX_RESPONSE_LEN`。

---

## Case 2 — 上下文/截断调研：64k 训练 cap 不是瓶颈，单轮 4096 才是

起因是一个很合理的怀疑：「rollout 设了 64k context，为什么训练样本还有超过 64k 的？`truncated_ratio` 不是 0，会不会把题目截掉污染训练？」这条把 Harbor 里**三层不同的长度限制**和**两种不同的"截断"**全核了一遍，结论对续训配置有直接影响。

### 三层长度限制（务必分清，它们管的不是一回事）

| 层级 | 配置项（27B run 实测值） | 作用域 | 命中后果 |
|------|--------------------------|--------|----------|
| 单轮生成 | `ROLLOUT_MAX_RESPONSE_LEN` = **4096** → `llm_kwargs.max_new_tokens` | **一次** LLM 调用最多生成的 token | `finish_reason="length"`，该轮输出被掐断 |
| 单轮 context | `ROLLOUT_MAX_CONTEXT_LEN` / `SGLANG_CONTEXT_LENGTH` = **65536** | **一次** LLM 调用 prompt+生成的总上限（`rollout.py:869`→`llm_kwargs.max_context_tokens`；`arguments.py:1912` 推出 `rollout_max_prompt_len = ctx-1`） | 推理引擎拒绝/截断该轮 |
| 训练样本总长 | `HARBOR_MAX_TRAIN_TOTAL_TOKENS` = **65536** | **整条轨迹**拼起来后、进优化器前的硬 cap | 从最旧 token 开始裁，见下 |

外加 agent 轮数上限 yaml `max_turns: 120`，且 Terminus2 用 `linear_history: true` + `enable_summarize: false`（yaml 第 38/41 行）——**没有滑窗、没有压缩**，全历史线性堆进每轮 prompt。所以"为什么训练样本会超 64k"的答案是：

- 64k 是**单轮** context 闸，不是整条轨迹的闸；
- `trajectory_retokenized` 的一个训练样本 = 整条多轮轨迹拼接（turn1.prompt + turn1.resp + 轮间新增 context + turn2.resp + …），几十轮堆下来**天然可能 > 单轮 64k**；
- 所以才需要 train 侧那道独立的 `HARBOR_MAX_TRAIN_TOTAL_TOKENS=65536` 闸把整条轨迹裁回 ≤64k（27B wrapper 第 74/92/93 行挂了 `cap_total_tokens_allow_response_trim_in_samples` + `..._in_rollout_data`）。

### 两种"截断"别混（口径陷阱）

- **`truncated_ratio` / `harbor/trial_timeout_ratio`** = trial 撞 **3600s wall-clock 超时**，**跟 64k 上下文毫无关系**。代码：`atif_adapter.py:39-46` —— 只有 `exception_type` 含 `"Timeout"` 才置 `Sample.Status.TRUNCATED`，否则 COMPLETED/FAILED。`rollout.py:443-445` 把它同义透出为 `trial_timeout_*`。
- **真正的"长度命中"**是另一组指标：
  - `harbor/any_step_length_finish_sample_ratio` —— 轨迹里**至少有一轮**撞 4096（`finish_reason=length`）的样本占比；
  - `harbor/train_context_trimmed_sample_ratio` —— 被 train 侧 64k 硬 cap 实际裁到的样本占比；
  - `harbor/train_response_trimmed_sample_ratio` —— 裁到**连 response 都被砍**的样本占比（最严重）。

### 裁剪是从哪头裁的（决定"题目会不会被截"）

硬裁逻辑 `_hard_trim_plan`（`sample_postprocess.py:241`）：`keep = min(total, cap)`，`tokens = tokens[-keep:]` —— **保留尾部、从最旧 token 开始砍**。terminal-bench 的题面/instruction 在 system + 首轮 prompt（轨迹最前面），所以**理论上一旦整条轨迹 > 64k 触发裁剪，最先被砍的确实是任务描述**。这个担心方向上成立，要不要慌取决于实测裁剪量。

### 实测数据（run96retok2，launcher log 全程 23 步）

数据来源：`launcher_run96retok2.log` 里每个 rollout 打的 `rollout.py:1455 - perf N: {...}` 指标 dict（过滤 `HarborTrialWorker`/`tmux_session` 容器噪声后解析）。

| perf | reward_mean | response_len/mean | response_len/max | trim_ratio | resp_trim_ratio | max_orig_len | timeout_ratio | **anylen_finish** |
|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| 1  | 0.115 | 11.5k | 43515 | 0 | 0 | 0 | 0.109 | 0.082 |
| 2  | 0.251 | 12.7k | 37319 | 0 | 0 | 0 | 0.115 | 0.059 |
| 5  | 0.131 | 11.7k | 43142 | 0 | 0 | 0 | 0.074 | 0.017 |
| 6  | 0.370 | 9.2k  | 39412 | 0 | 0 | 0 | 0.049 | 0.086 |
| 10 | 0.007 | 9.9k  | 42940 | 0 | 0 | 0 | 0.161 | 0.137 |
| 12 | 0.066 | 10.0k | 40891 | 0 | 0 | 0 | 0.002 | 0.272 |
| 15 | 0.075 | 10.2k | 59026 | 0 | 0 | 0 | 0.011 | 0.428 |
| 17 | 0.081 | 9.9k  | 45613 | 0 | 0 | 0 | 0     | 0.410 |
| 18 | 0.000 | 9.6k  | 54754 | 0 | 0 | 0 | 0.032 | 0.446 |
| 19 | 0.000 | 10.3k | 47438 | 0 | 0 | 0 | 0     | 0.326 |
| 20 | 0.048 | 10.9k | 59358 | **0.00166** | 0 | **65652** | 0.012 | 0.544 |
| 21 | 0.000 | 9.6k  | 55005 | **0.00184** | 0 | **65694** | 0.048 | 0.510 |
| 22 | 0.049 | 10.6k | 54057 | 0 | 0 | 0 | 0.049 | 0.677 |
| 23 | 0.007 | 9.1k  | 52413 | **0.00172** | 0 | **65542** | 0.002 | 0.570 |

（log 里同时有三条 INFO 实锤：`sample_postprocess.py:146 - Harbor hard sample token cap trimmed 1/603 ... max_total_before=65652, max_total_after=65536` 等，分别对应 perf 20/21/23。）

### 结论：把怀疑逐条拍板

1. **❌ 64k 训练截断不是瓶颈。** 全程 23 步只有 3 步出现裁剪，每次 **1/约600 ≈ 0.17%**；裁前最长才 65542~65694，**只超 64k 几十~一百多 token**；裁后精确 = 65536。绝大多数轨迹根本到不了 64k。
2. **❌ "题目被截"在本 run 不成立。** `train_response_trimmed_sample_ratio` **全程 = 0**（从没砍到 response）；被砍的只是 prompt 最前面那几十个 token，而题面在前几千 token 内，这点裁剪够不着。方向上的隐患成立，但实测量级可忽略。
3. **❌ `truncated_ratio≠0` 不是长度问题。** 它是 3600s 任务超时（前期 ~11%、后期降到 0~5%），与 64k 无关。
4. **✅ 单轮 4096 是后期最显眼的截断点（但它是症状，不是病根，见 Case 3）。** `any_step_length_finish_sample_ratio` 从前期 ~5% 一路飙到后期 **54%–68%**——后期超过一半的轨迹里至少有一轮被 4096 掐断，与 reward 后期普遍掉到 0~0.05 同步。**Case 3 进一步解码这些被截断的内容，发现是冗余复读垃圾，所以"放宽 4096"是错的方向。**
5. **✅ 印证"64k 对 response overkill"的直觉。** response 总和才 ~9–12k（其余 64k 容量被 prompt/终端回显 `raw_content:true` 占掉），训练 cap 对 response 远没用满；后期撞 4096 的也不是有用内容，所以 64k 整条 cap 更不是瓶颈。

> 一句话：**之前把瓶颈找错了位置。** 不是 64k 训练截断、不是题目被截、不是任务超时。后期一多半轨迹撞 4096 是真的，但 **Case 3 证明撞上限的内容是复读垃圾**——所以根因是 Case 1 的熵漂移（KL=0+entropy=0 让策略在难任务上塌缩成复读），4096 截断只是这个塌缩的可观测症状。三个 Case 是同一套动力学。

### 相关 metric 速记（trajectory_retokenized 下）

- `response_len/mean` / `response_len/max`：**整条轨迹**的 response token（不是单轮）。前缀 `response_len/` 来自 `slime/ray/rollout.py:1512`。
- `harbor/any_step_length_finish_sample_ratio` / `step_length_finish_*`：单轮撞 4096（`rollout.py:491-494`）。
- `harbor/train_context_trimmed_sample_ratio` / `train_response_trimmed_sample_ratio` / `train_context_max_original_total_length` / `..._max_trimmed_total_length`：64k 硬 cap 的裁剪统计（`rollout.py:498-504`）。
- `harbor/trial_timeout_ratio` ≡ `truncated_ratio`：3600s wall-clock 超时（`atif_adapter.py:39-46`）。

---

## Case 3 — 后期变长的 response 是「真干活」还是「冗余复读」？（解码实证）

承接 Case 2 的疑问：后期 `response_len` 提升、撞 4096 比例飙升，那变长的内容到底是有用的命令/推理，还是冗余重复？把 token 解码成文字直接看。

### 工具

`examples/harbor_terminal_rl/inspect_rollout_dump.py`（用 `AutoTokenizer` 把 dump 里 `tokens[-response_length:]` 解码成文字；`--filter-finish-reason length` 可只看被 4096 截断的样本）。本次额外按 `loss_mask` 把 response 拆成 **assistant 真实生成（mask=1）** 与 **终端回显 observation（mask=0）**，并用 gzip 压缩率 + 行级重复率量化冗余。

```bash
# 解码后期某 dump 里被 4096 截断的样本，看 response 文本
python3 inspect_rollout_dump.py \
  /path/to/shared_storage/.../rollout_dumps/22.pt \
  --hf-checkpoint /path/to/models/Qwen3.5-27B \
  --filter-finish-reason length --max-samples 6 --include-prompt
```

### 实测：被 4096 截断的内容 = 机械复读直到撞上限

抽 run96retok2 rollout 22 的 `length` 样本逐条解码，形态高度一致，**全是 reward=0 的复读**：

| 样本 | 任务 | reward | 复读形态（tail 实录） |
|---|---|---|---|
| #0 | circuit-fibsqrt | 0.0 | `out199=0\nout200=0\n…out238=0` 一行行往下数，41083 字符 |
| #2 | circuit-fibsqrt | 0.0 | 同样 `out213=0…out270=0` 复读，68500 字符 |
| #3 | circuit-fibsqrt | 0.0 | 自然语言打转：`The /app/gates.txt file 208 377 and 20000… has 32000 lines and 208 377 and…` 同句换序重复 |
| #5 | circuit-fibsqrt | 0.0 | 退化成纯噪声：`0 000 0 00 0 0000 0 0000…` |

复读发生在 **assistant 自己生成的 token** 里（不是终端回显堆积），且这些样本都拿 0 分。

### 量化指标（早期 rollout 2/6 → 后期 15/22）

| 指标 | 含义 | 2 | 6 | 15 | 22 |
|---|---|---|---|---|---|
| `length` 样本数 | 被 4096 截断 | 4 | 3 | 30 | **56** |
| `asst_frac`(stop 样本) | response 里 assistant 占比 | 0.48 | 0.48 | 0.65 | **0.78** |
| `dup_line_frac` | 重复行占比 | 0.4–0.6 | 0.4–0.6 | 0.4–0.6 | 0.4–0.6 |
| `max_line_repeat` | 同一行最多重复次数 | 45–87 | 6–45 | 65–94 | 60–62 |

两个关键读数：
- **`asst_frac` 0.48 → 0.78**：后期变长**主要是模型自己在生成**，不是 observation 堆积——排除了"终端输出太多撑大 response"的可能。
- **`length` 样本数 4 → 56（14×）**：被复读撑爆 4096 的样本量级暴涨。

### 一个诚实的修正

原本预期"后期复读率显著高于早期"，但 `dup_line_frac` / `compress_ratio` **早期后期差别不大**——早期 `stop`（正常收尾）样本本身也有不低重复行（`max_line_repeat~87`），因为这类任务（写 gates.txt、JSON 命令块）天然含结构化重复。所以准确表述是：

> 问题**不是**"重复率从低涨到高"，而是 **后期撞 4096 被截断的绝对数量暴涨（4→56），且这些被截样本几乎全是 reward=0 的退化复读**。

### 任务分布：是难任务上的局部塌缩，不是全局崩坏

rollout 22 的 56 个 length 截断里：`circuit-fibsqrt` 19/92、`dna-assembly` 15/74 两个难任务占了大半（都要生成超大文件 / 海量结构化输出）。说明这是**特定难任务上，策略解不动时塌缩成复读**，而非整体复读崩坏。

### 结论 + 对续训配置的影响

1. **后期变长 = 冗余复读垃圾，不是有用产出。** 模型在 circuit-fibsqrt 这类难任务上不"放弃"，而是逐行复读 / 自然语言打转 → 撞 4096 被截 → 命令残缺 → reward=0。
2. **这与 Case 1 的熵漂移是同一根因**：KL=0 + entropy=0，策略在解不动的任务上熵塌缩进复读模式。
3. **⚠️ 因此「放宽 `ROLLOUT_MAX_RESPONSE_LEN` 4096→8192」是错的**（修正 Case 2 早期写法）：让模型复读更多垃圾、烧更多算力，不解决 reward。4096 截断是**症状不是病根**。
4. **该做的是治本**：加 `--entropy-coef` / 小 `--kl-loss-coef` 抑制复读塌缩；可选对 `finish_reason=length` 或高 `dup_line_frac` 的样本加复读/超长惩罚，直接打掉复读捷径。

---

## 排错速查表（可复用经验）

### W&B step 轴口径（容易看错）

- `train/loss` 绑 `train/step`（= 真实 optimizer step；`define_metric` 在 `slime/utils/wandb_utils.py:224-232`）。
- `harbor/*`（含 `harbor/reward_mean`）**没有绑任何 step_metric** → W&B 默认画在隐式全局 `_step` 上，每次 `wandb.log()` 调用 +1，会被 ~3× 放大。所以"看起来过了 24 个 step"其实可能只有 8 个真实 optimizer step。
- 想对齐：图里把 `harbor/*` 的 x 轴手动切成 `rollout/step`，或给 run 加 `--wandb-always-use-train-step`（只改显示，不动训练）。

### step / batch 语义

- 真实 optimizer step 数：log 里数 `train/step` 的去重值（过滤容器噪声 `HarborTrialWorker` / `tmux_session`）。
- 一次 rollout 批 = `rollout_batch_size(8) × n_samples_per_prompt(32)` = **256 条轨迹** = `global_batch_size(256)` = **1 个 optimizer step**。
- step 数公式：`train_iters = num_rollout × rollout_batch_size × n_samples_per_prompt // global_batch_size`（`slime/backends/megatron_utils/model.py:160`）。
- 注意 `--use-dynamic-global-batch-size` 在 slime 里**不是有效 arg**（无 add_argument，无消费点）；真正起作用的是 `--global-batch-size` + `--use-dynamic-batch-size`（按 `max_tokens_per_gpu` 把 256 条轨迹重切成可变数量的 token-packed microbatch，即进度条里的"NNN microbatch"，≠ 样本数）。

### 从 launcher log 抓真实指标（绕开容器噪声）

容器内 agent 解题会往 log 里灌大量 `HarborTrialWorker` / `tmux_session.py` 行（含 `entropy=`、`loss=`、`kl=10.0` 等假命中），必须过滤：

```bash
LOG=.../logs/launcher_run96retok2.log
# 完整 rollout 指标 dict（data.py:206）
grep -aE "rollout [0-9]+: \{" "$LOG" | grep -avE "HarborTrialWorker|tmux_session"
# reward 序列
grep -aE "reward_mean'" "$LOG" | grep -avE "HarborTrialWorker|tmux_session" | grep -aoE "reward_mean': [0-9.]+"
# 真实训练 step
grep -aE "train/step'" "$LOG" | grep -avE "HarborTrialWorker|tmux_session" | grep -aoE "train/step': [0-9]+" | sort -u
```

关键字段：`rollout/raw_reward`（任务原始得分）、`rollout/response_lengths`、`rollout/truncated`、`harbor/reward_mean`、`harbor/num_groups`（应=8）、`harbor/num_failed_trials`。

### 长度/截断口径（最容易看错，详见 Case 2）

- **三层长度限制各管各的**：`ROLLOUT_MAX_RESPONSE_LEN`（单轮生成，27B=4096）< `ROLLOUT_MAX_CONTEXT_LEN`（单轮 prompt+生成，64k）< 整条轨迹（多轮拼接，可超 64k）→ `HARBOR_MAX_TRAIN_TOTAL_TOKENS`（train 侧硬裁回 ≤64k）。"训练样本超 64k"是裁剪**前**的统计，正常。
- **`truncated_ratio` ≠ 长度截断**：它 ≡ `harbor/trial_timeout_ratio` = 3600s wall-clock 超时（`atif_adapter.py:39-46`）。真正的长度命中看 `harbor/any_step_length_finish_sample_ratio`（单轮撞 4096）和 `harbor/train_context_trimmed_sample_ratio`（64k 硬裁）。
- **硬裁从最旧 token 砍**（`tokens[-keep:]`，`sample_postprocess.py:241`）→ 理论上会先砍题面；但实测裁剪率 ~0.17%、`train_response_trimmed_sample_ratio` 恒 0，量级可忽略。先查实测再下结论。

### 判断「response 变长是有用还是复读」（详见 Case 3）

光看 `response_len` 涨不能下结论，必须解码看内容：

```bash
# 1) 解码被 4096 截断的样本，肉眼看 tail 是不是复读
python3 inspect_rollout_dump.py <dump.pt> --hf-checkpoint <HF_27B> \
  --filter-finish-reason length --max-samples 6
```

- 用 `loss_mask` 拆 **assistant 生成（mask=1）** vs **终端回显（mask=0）**：`asst_frac` 高 → 变长是模型自己写的（要警惕复读）；低 → 是 observation 堆积（另一类问题）。
- 客观冗余指标：`dup_line_frac`（重复行占比）、`max_line_repeat`（同行最多重复次数）、gzip `compress_ratio`（越低越冗余）。
- ⚠️ 别只比"早期 vs 后期重复率"——结构化任务（写大文件/JSON）天然高重复。关键看 **`finish_reason=length` 的样本数是否暴涨 + 这些样本是否几乎全 reward=0**。
- 结论若是复读：**不要放宽 `ROLLOUT_MAX_RESPONSE_LEN`**（喂更多复读垃圾），要从熵正则/KL/复读惩罚治本。

### 监控误判：run 正常收尾 vs 崩溃

`monitor_full89_...health.sh` 在 Ray 集群正常拆除后会报 `verdict=DEAD / ray-head-not-running`，这**可能是 run 跑满 `num_rollout` 后的正常退出**，不是崩溃。区分方法——tail launcher log 找：

- 正常收尾：`Job '<id>' succeeded` + `wrote worker stop marker`；`latest_checkpointed_iteration` 达到末值；（`BrokenPipeError` 来自 wandb teardown、`unexpected EOF` 来自收尾脚本，都是无害噪声）。
- 真崩溃：`could not produce enough complete trainable groups`（insufficient-trainable-groups）、CUDA OOM、ActorDied / 心跳丢失。

run 结束后记得删掉对应的健康检查 cron，否则会对已完成的 run 每半小时误报一次 DEAD。

---

## Case 4 — context-exceeded（prompt 撑爆 64k 被 SGLang 直接拒绝）+ 从头重训方案

### 新发现：一个独立于"单轮 4096"的失败模式

在另一台机器上独立跑 27B inference（ctx 64k、无 summary）观察到失败分布：**context-exceeded 31/37 (84%)** 是主因。回到训练 launcher log 核对，**训练侧同样存在且量级很大**：

```
[http_server] Error: Requested token count exceeds the model's maximum context length
of 65536 tokens. You requested a total of 66132 tokens: 62036 from input + 4096 completion.
```

| 指标 | 实测（run96retok2 全程） |
|---|---|
| SGLang `http_server` context-exceeded 拒绝次数 | **3066 次** |
| 被拒 prompt 的 input token min/mean/max | **61440 / 62213 / 65468**（全部擦着 64k 上限） |

这与 Case 2/3 的"单轮 4096 截断"是**两条不同的路径**：
- **Case 3（单轮 4096）**：`finish_reason=length`，模型一轮生成撞 `ROLLOUT_MAX_RESPONSE_LEN`。
- **Case 4（整 prompt 64k）**：Terminus2 `linear_history: true` + `enable_summarize: false`，全历史线性堆积，prompt 本身撑爆 64k → SGLang 在 prefill 前直接 400 拒绝整轮。训练侧靠 `harbor_rollout_group_attempt_multiplier: 8` 重采没丢 trial，但长任务后期系统性撞墙、大量浪费。

### Fix：开 summary（数据安全性已验证）

`enable_summarize: false → true`：Terminus2 在快撑爆时自动 LLM-summarize 中间结果，把历史压回去。代价是每次 summary 多一次 LLM 调用。

**与 `trajectory_retokenized` 兼容性已验证安全**：summary 会重写历史（turn 间 prefix 对齐失败 → `strategy=="split"`），`atif_adapter.py:707-721` 会 `finalize_chunk` 收尾旧 chunk、从 summary 渲染后的新 prompt **开全新 chunk**（清空 response/loss_mask），不会把 policy 没 condition 过的 token 喂进 loss。审计文档 `NORMALIZED_TRAJECTORY_STITCHING_AUDIT.md:189` 确认这是正确行为。

### 关于 128k 与显存（为何最终决定保持 64k）

曾考虑把 ctx 抬到 128k（模型原生 `text_config.max_position_embeddings=262144` 支持）。但 train 侧有 OOM 红线：
- slime dynamic-batch 的 `first_fit_pack`（`seqlen_balancing.py:180-198`）**不切分超长样本**——一条 131072 token 的轨迹会**单独占一个 bin，bin 大小 = 131072**。
- 约束 `bin ≤ max_tokens_per_gpu × cp_size`（`dp_schedule.py:213`）要求 `MAX_TOKENS_PER_GPU × CP(2) ≥ 131072` → `MAX_TOKENS_PER_GPU` 须从 32768 翻到 **65536**。27B 在 65536 tokens/GPU 前向激活很可能 OOM。
- 出路要么 CP=2→4（把 train GPU 从 32 翻到 64，打乱 96 卡 placement，过于折腾），要么保持 64k。

**决定：全程保持 64k。** 既然 summary 已从根上治了 context-exceeded，64k 足够，且**完全规避 OOM 风险与 CP 改动**。

### 从头重训方案（v2）—— 不续训，从 pretrained 重新开始

> 续训会带着已塌缩的策略，entropy/KL 难以拉回；从头训才能让正则从一开始就生效。

**关键决策一览：**

| 项 | 决定 | 理由 |
|---|---|---|
| summary | **开**（`enable_summarize: true`） | 治 context-exceeded 主因（Case 4） |
| context | **保持 64k**（rollout/sglang/train cap/seq 全不动） | 规避 128k 的 OOM + CP 改动 |
| single-step max token | **保持 4096** | Case 3：撞 4096 的是复读垃圾，放宽只会喂更多垃圾 |
| 熵正则 | `--entropy-coef 0.01` | 对抗复读/熵塌缩（Case 1/3 根因） |
| KL loss | `--use-kl-loss --kl-loss-coef 0.001`（`kl-coef` 保持 0） | 轻微锚定参考模型；治塌缩 |
| 学习率 | `1e-6 → 5e-7` constant | 降低过快坍到捷径解的风险 |
| epoch | 2 epoch（`NUM_ROLLOUT=24` 不变） | 同原计划 |

**⚠️ 易踩的坑（核代码确认）：**

1. **`--kl-loss-coef 0.001` 单独设无效**：KL 项由 `args.use_kl_loss` 门控（`train_loss.py:140`、`loss.py:972`），**不会被 `kl_loss_coef>0` 自动打开**。必须同时加 `--use-kl-loss`。
2. **开 KL 强制需要 reference model**：`--use-kl-loss` 触发 `arguments.py:1682` 检查 `ref_load`。已确认 `Qwen3.5-27B_torch_dist_pp8_timeout60/latest_checkpointed_iteration.txt` 存在 → 可用。代价：多一份 ref 模型显存 + 每步一次 ref 前向。
3. **`kl-coef` 与 `kl-loss-coef` 不能同时非零**（`arguments.py:1757` assert）。保持 `kl-coef 0.00`。
4. **entropy/KL/lr 在 base script 里是硬编码**（`run_qwen35_9b_terminal_bench2_64h20.sh:434-437,452`），不是 env 驱动 → 必须直接改脚本那几行。
5. **`USE_CUSTOM_NO_ENTROPY_LOSS=1` 的 custom loss 不用换**：`policy_loss_no_entropy_when_disabled` 会自动检测 `entropy_coef>0` 启用 entropy 计算（`train_loss.py:25`）。
6. **summary 不是 env 驱动**：base script 只把整个 yaml 经 `--custom-config-path` 传入，不 patch agent kwargs → 必须直接改 yaml `enable_summarize: true`。

**具体改动（3 个文件）：**

```
# 文件 1: examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh
#   line 434:  --kl-loss-coef 0.00      →  --kl-loss-coef 0.001
#   line 437:  --entropy-coef 0.00      →  --entropy-coef 0.01
#   GRPO_ARGS 内新增一行:               +  --use-kl-loss
#   line 452:  --lr 1e-6                →  --lr 5e-7
#   （kl-coef 0.00 保持不变）

# 文件 2: examples/harbor_terminal_rl/harbor_terminal_bench2_full89_96gpu_27b_..._fully_async.yaml
#   line 38:   enable_summarize: false  →  enable_summarize: true

# 文件 3: examples/harbor_terminal_rl/start_full89_96gpu_27b_fully_async_tmux.sh
#   确认 CONFIG_TEMPLATE 指向上面那个被改过的 yaml（默认就是它）
#   ctx/seq/cap/MAX_TOKENS_PER_GPU 全部保持 64k 不动
```

---

## Case 5 — 开了 summary，为什么还有 context-exceeded？（slime backend 异常类型 bug + 修复）

v2 run（v2r4，summary ON）跑起来后仍观察到大量 SGLang `exceeds the model's maximum context length` 400（首个 rollout ~70min 内就累积 600+，train step1 时 887）。开了 summary 不该如此。逐代码核查后定位为**一个真 bug + 一个次因**。

### Terminus2 的两道 summary 防线

1. **主动 summary（proactive）**：`_check_proactive_summarization`（`harbor/.../terminus_2.py:964`）每轮算 `free_tokens = get_model_context_limit() - current_tokens`，`< proactive_summarization_threshold(8000)` 时主动 summarize。slime backend 的 `get_model_context_limit()` 返回 `max_context_tokens`=64k（`slime_plugins/harbor/llm.py:132`）。
2. **被动 fallback**：`_query_llm`（`terminus_2.py:1022`）`except ContextLengthExceededError:` → unwind + summarize 重试。

### 根因（真 bug）：slime backend 抛错类型不对

`SlimeSGLangLLM.call()`（`slime_plugins/harbor/llm.py`）发请求撞 SGLang 400 时，**把它包成通用 `RuntimeError`**，而不是 `ContextLengthExceededError`。Terminus2 的 fallback（防线2）只 catch `ContextLengthExceededError` → **接不住** → 整个 turn 失败、完全绕过 summary 补救。这就是 context-exceeded 的主要来源。

### 次因：proactive 的 token 估算偏低

防线1 的 `current_tokens = _count_total_tokens(chat)` 用 Terminus2 自己的估算（litellm/tokenizer），与 SGLang 服务端真实 chat-template 编码有偏差（尤其 `raw_content: true` 终端回显 + 特殊 token + 模板开销）。当估算 < 真实时，以为还有 >8000 free tokens 没触发 summary，但真实 prompt 已超 64k → SGLang 400。8000 的 margin 不足以吸收偏差。

### 修复（`slime_plugins/harbor/llm.py`，只动插件，不碰 core）

两层，都抛 `ContextLengthExceededError` 让 fallback summary 能接住：

1. **主动检查**（发请求前，用与 SGLang 相同的真实 token 数）：
   ```python
   if self._max_context_tokens is not None:
       max_new = int(sampling_params["max_new_tokens"] or 0)
       if len(prompt_token_ids) + max_new > int(self._max_context_tokens):
           raise ContextLengthExceededError(...)
   ```
   抄 `harbor/llms/tinker.py:204` 的 proactive-raise 模式——根本不发那个注定失败的请求。
2. **400 兜底**（万一估算与服务端漂移）：捕获 SGLang 400 且 body 含 `"maximum context length"` 时，翻译成 `ContextLengthExceededError`（而非 RuntimeError）。

修复后 `import` 行：`from harbor.llms.base import ContextLengthExceededError, LLMResponse`（两类同在 `harbor/llms/base.py`，line 40 / 17）。

### 验证点（重起后看）

- `grep -ac "exceeds the model's maximum context length" <launcher.log>` 应**大幅下降**（撞墙的请求改为触发 summary 而非失败）。
- 可选：看 Terminus2 是否多了 summarization 子调用（`grep summariz`）——说明 fallback 真的接住了。
- ⚠️ 注意权衡：summary 多 → 每次多一次 LLM 调用，rollout 略慢；但不再丢 turn。

### 可选叠加（未做，按需）

把 yaml `proactive_summarization_threshold` 从 8000 调高到 12000–16000，让防线1 更早触发、少依赖兜底。本次先只修异常类型（最小、最对症）。

---

## Case 6：v2r5 完整跑完后的最终复盘（RUN_TAG=v2r5_130222）

**结论先行**：两个修复都端到端验证生效，没有走 run96retok2 的塌缩老路；但 **epoch2 相对 epoch1 没有提升（基本持平、略降）**，且 **训练末段 entropy 冲高到 ~1.0 与 reward 软化是同一现象的两面**。最优 checkpoint 在**中段（iter 15–17）**，不是末尾 iter 23。

### Run 配置回顾

summary ON + `entropy-coef 0.01` + `use-kl-loss` / `kl-loss-coef 0.001` + 常数 `lr 5e-7`，context 64k，single-step max token 4096，2 epoch / NUM_ROLLOUT=24，trajectory_retokenized，T=1.0 / n=32。叠加 Case 5 的 `llm.py` context-exceeded 异常类型修复。

### 完成状态（正常结束，非 DEAD）

- `Job 'raysubmit_L88qwmLAgPtKQBvQ' succeeded` + `wrote worker stop marker`（@12:21:48）。
- `latest_checkpointed_iteration.txt = 23`，最终 `train/step=23`，checkpoint 存到 `iter_0000023`。
- 0 fatal markers 全程。

### 修复验证（两个都生效）

- **context-exceeded 全程仅 87 次**（对比 pre-fix v2r4：单单 train step 1 就 887 次）。撞墙请求被路由到 fallback summary 而非失败，`grep summariz` 24k+ 命中。
- **没有熵/长度塌缩**：`any_step_length_finish_sample_ratio` 全程在 0.05–0.36 震荡，**从未进入 run96retok2 的塌缩带 0.54–0.68**。entropy 不降反升（见下），证明熵正则把策略推向探索、没退化成"重复刷到 4096"。

### Epoch 对比（按 rollout 顺序，前 12 / 后 11；末轮 teardown 未落盘 reward）

`harbor/reward_mean` 是 T=1.0 / n=32 探索分布的 **shaped-reward 均值**，不是 pass@1；此处只做轮间相对对比（口径一致可比）。

| | 均值 | 中位数 | 区间 |
|---|---|---|---|
| **Epoch 1**（r1–12） | **0.119** | 0.116 | 0.0 – 0.221 |
| **Epoch 2**（r13–23） | **0.111** | 0.114 | 0.002 – 0.251 |
| 差值 | **−0.008（−6.5%）** | −0.002（持平） | — |

→ **第二轮无显著提升**，差异在单 rollout 抖动幅度（±0.1）以内。reward 全程高点出现在 epoch 交界（r14–17：0.19/0.16/0.11/**0.25**），随后 epoch2 后半段塌到低带（r20–23：0.002/0.055/0.008/0.083）。**末尾 checkpoint（iter 21/23）正落在软化尾巴上，不是最优点。**

### entropy_loss 轨迹：单调上行，末段 ~1.0

| 阶段 | step | entropy_loss |
|---|---|---|
| 早期 | 1–9 | 0.45–0.60（均值 ~0.51） |
| 中段 | 10–17 | 0.59→0.77→回落 0.61–0.66 |
| 后段 | 18–20 | 0.72–0.74 |
| **末段** | **21–23** | **1.01 / 0.99 / 0.97** |

关键不是"到了 0.7"，而是**它单调上行**：0.51（早）→ 0.7（中后）→ ~1.0（末），末段比早期高 ~90%。

### 核心关联：熵冲高 ↔ reward 软化（同一现象两面）

| rollout | r17 | r18 | r19 | r20 | r21 | r22 | r23 |
|---|---|---|---|---|---|---|---|
| reward | **0.25** | 0.065 | 0.138 | 0.002 | 0.055 | 0.008 | 0.083 |
| entropy | 0.66 | 0.74 | 0.72 | 0.73 | **1.01** | 0.99 | 0.97 |

熵爬到 1.0 的同一段，正是 reward 塌到低带那几轮。解读：常数 lr 5e-7 跑到后期，**熵正则的探索压力盖过了 reward 信号的收敛压力**，策略变得更随机/发散 → 采样质量下降 → reward 软化。这与"epoch2 没提升"是**同一回事**：`entropy-coef 0.01` 对 89 题 / 2 epoch 这个小数据集**略偏强**，末段把熵顶到 ~1.0、挤掉了 reward 收敛空间。

判定：**不是 bug，是"到顶了"而非"练坏了"**。熵上行 + 不塌缩 = 正则按设计生效；只是后期探索过度。

### 行动项

1. **选 checkpoint**：优先 **iter_15 / iter_17**（reward 峰 + 熵适中 ~0.6 区间），避开 iter_23（熵已冲高、reward 在低带）。
2. **下次若再训**，二选一收敛后期：(a) `entropy-coef` 0.01→**0.005**；或 (b) lr 加**末段衰减**（cosine/linear，而非常数 5e-7），让后期从"探索"切回"收敛"。
3. **验证"提升没提升"的正确标尺**：训练曲线只能看相对趋势；要和 40% pass@1 参考对齐，需跑**对齐口径的低温 eval**（T≈0、n=1、TB2 全集，只看 terminal 成功率），对比 iter_15 vs iter_23 的真实成功率。
