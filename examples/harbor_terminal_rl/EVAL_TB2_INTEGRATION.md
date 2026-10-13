# 周期性 TB2 Eval 接入说明（Harbor RL）

> 给 Harbor 终端 agent RL 训练加上「每 N 个 rollout step 在 held-out 测试集（Terminal-Bench 2, 89 题）上跑一次 eval，pass rate 画到 wandb `eval/*`」的能力。
>
> 实现日期：2026-06-17 · slime core **0 改动**，全部走 plugin hook + 配置。

---

## 1. 动机（为什么做）

之前的 GSPO + 长度reward + 课程学习的 27B run 只在**训练任务**（`by_length` 课程集）上有 reward 曲线，**看不到模型在 held-out 测试集上的泛化能力**——无法判断训练是真的在提升泛化，还是只是过拟合课程集。

目标：每训练 **10 个 rollout step**，在 **Terminal-Bench 2（TB2，89 个任务）** 上跑一次 eval，把 pass rate 画到 wandb `eval/tb2`，并在 step 0 先测一个 baseline 作为 t=0 参照点。

---

## 2. 调研结论（实测确认的事实）

| 事实 | 出处 |
|---|---|
| slime core 已内置周期 eval：`should_run_periodic_action(rollout_id, eval_interval, ...)` 触发 `rollout_manager.eval()` | `train_async.py:85`、`slime/utils/misc.py:75` |
| `--eval-interval 10` = 每 10 个 rollout step 触发一次（`(rollout_id+1) % 10 == 0`） | `slime/utils/misc.py:75` |
| Harbor `generate_rollout` 已原生支持 `evaluation=True` | `slime_plugins/harbor/rollout.py:43` |
| eval 结果自动 log 到 wandb `eval/<dataset>`（= 平均 reward = pass rate） | `slime/ray/rollout.py:1472-1500` |
| 设 `--eval-interval` 时 core 会 `assert args.eval_datasets` 非空 | `slime/utils/arguments.py:1751` |
| `_resolve_eval_datasets` **不检查 eval path 是否存在**（dummy path 可用） | `slime/utils/arguments.py:1629` |
| TB2 数据：`/path/to/shared_storage/.../terminal-bench-2-main`，89 个 `*/task.toml`，与训练集同构 | 实测 `find` |

### 两个必须解决的真问题
1. **`HarborTaskDataSource` 没有独立 eval 池**（`data_source.py:33`）：`get_samples()` 永远从训练 `self.entries` 取，且和训练共用 `sample_offset` 指针 → eval 会测训练任务、且把课程指针往前推**污染课程学习**。
2. **Harbor eval 复用训练的 `_generate_rollout_async`**，内含 oversampling + 长尾 kill + `target_groups=rollout_batch_size`（=16）逻辑（`rollout.py:53-54`）→ 直接用于 eval 会只测 16 题、且把慢任务中途 cancel → **pass rate 严重失真**。

### ⚠️ 关键架构坑（读源码才发现）
训练用的 **fully_async** rollout 函数 `generate_rollout_fully_async` **对 `evaluation=True` 直接 raise**（`slime_plugins/harbor/fully_async_rollout.py:433`，设计如此）。而 `--eval-function-path` 默认回退到 `--rollout-function-path`（= fully_async）→ **直接开 eval 会崩**。
→ 解法：显式把 `--eval-function-path` 指向标准 `generate_rollout`（它路由到新写的、不碰 fully-async 机制的 `_generate_eval_async`）。eval 与 fully-async 训练路径**完全解耦**。

---

## 3. 设计方案

| 改动 | 文件 | 作用 |
|---|---|---|
| 1 | `slime_plugins/harbor/data_source.py` | 加独立 eval 池 + `get_samples(evaluation=True)` 分支 |
| 2 | `slime_plugins/harbor/rollout.py` | 新增 `_generate_eval_async`：全量、关 oversample、关 tail-kill、不跑训练 reward hook |
| 3 | `tests/test_harbor_eval_data_source.py` | 5 个 CPU 单测锁住 eval 行为 |
| 4 | `run_qwen35_9b_terminal_bench2_64h20.sh`（共享 base） | eval flag 改成 `EVAL_INTERVAL` 驱动 + 向后兼容 |
| 5 | gspo wrapper + curriculum yaml | 接 eval flags + TB2 eval 池路径 |

**用户拍板的参数**：沿用训练采样温度、每题采 1 次（`n_samples_per_eval_prompt=1`）、step 0 先测 baseline（`SKIP_EVAL_BEFORE_TRAIN=0`）。

---

## 4. 具体改了什么

### 4.1 `slime_plugins/harbor/data_source.py`

`__init__` 末尾新增 eval 池加载（独立计数器）：
```python
# Held-out evaluation pool. ... 不动训练 sample_offset。
self.eval_entries = _load_eval_task_entries(args)
self.eval_sample_group_index = 0
self.eval_sample_index = 0
```

`get_samples` 加 `evaluation` 参数 + eval 分支：
```python
def get_samples(self, num_samples: int, evaluation: bool = False) -> list[list[Sample]]:
    if evaluation:
        return self._get_eval_samples()
    ...  # 训练逻辑完全不变
```

新增 `_get_eval_samples()`：遍历**全部** `eval_entries`，每题 `n_samples_per_eval_prompt`（默认回退 `n_samples_per_prompt`）个 sample，用 `eval_sample_*` 独立计数器，**不走 requeue、不碰训练指针**。

新增模块级 `_load_eval_task_entries(args)`：复用现有 `_discover_task_entries` + `_normalize_entry`，读 `harbor_eval_tasks_dir` / `harbor_eval_tasks_glob` / `harbor_eval_tasks_limit` / `harbor_eval_tasks_offset`；**未配置则返回 `[]`（不 raise）**，保证没开 eval 的 run 行为完全不变。

### 4.2 `slime_plugins/harbor/rollout.py`

`generate_rollout` 的 eval 分支改为走专用协程：
```python
def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        output = run(_generate_eval_async(args, rollout_id, data_source))
        return RolloutFnEvalOutput(data={"harbor": output["eval"]}, metrics=output["metrics"])
    output = run(_generate_rollout_async(args, rollout_id, data_source))
    return RolloutFnTrainOutput(samples=output["samples"], metrics=output["metrics"])
```

新增 `_generate_eval_async(args, rollout_id, data_source)`：
- `data_source.get_samples(0, evaluation=True)` 取**全部** eval 任务（一题一 group）；
- `asyncio.gather` **等所有任务跑完**（无长尾 cancel——慢任务也是真实数据点）；
- 复用 `_run_one_group(..., retry_missing_samples=0)` 做实际 trial（agent/sandbox/转换路径与训练完全一致）；
- **不调 `_apply_rollout_sample_hooks`**（不跑长度reward shaping / dynamic filter）→ `eval/harbor` 保持干净的平均任务 reward；
- 返回 `{"eval": {"rewards","truncated","samples"}, "metrics": {...}}`，metrics 用 `harbor/eval_*` 前缀。

同时简化了训练用的 `_generate_rollout_async`：去掉签名里的 `*, evaluation` 参数和那段现在永不执行的 eval 分支（死代码）。相应更新 `tests/test_harbor_rollout_async.py`（去掉 4 处 `evaluation=False` 入参）。

### 4.3 `examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh`（共享 base 脚本）

把原本硬编码的 `--skip-eval-before-train` 改成 **`EVAL_INTERVAL` 驱动 + 向后兼容**：
```bash
if [ -n "${EVAL_INTERVAL:-}" ]; then
   HARBOR_ROLLOUT_ARGS+=(
      --eval-interval "${EVAL_INTERVAL}"
      --eval-function-path slime_plugins.harbor.rollout.generate_rollout  # 绕过 fully_async 的 eval-raise
      --eval-prompt-data "${EVAL_DATASET_NAME:-tb2}" /dev/null            # dummy 占位，只为过 assert
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
   )
   if [ "${SKIP_EVAL_BEFORE_TRAIN:-0}" = "1" ]; then
      HARBOR_ROLLOUT_ARGS+=(--skip-eval-before-train)
   fi
else
   HARBOR_ROLLOUT_ARGS+=(--skip-eval-before-train)   # 不设 EVAL_INTERVAL → 旧行为完全不变
fi
```

### 4.4 gspo wrapper（`run_terminalworld_full1530_..._gspo_lenreward_fully_async.sh`）

```bash
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-tb2}"
export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"   # 0 = step0 先测 baseline
```

### 4.5 curriculum yaml（`harbor_terminalworld_full1530_..._gspo_lenreward_curriculum.yaml`）

```yaml
harbor_eval_tasks_dir: /path/to/workspace/code/terminal-bench-2-main
harbor_eval_tasks_glob: "*/task.toml"
```

---

## 5. 验证结果

全部通过：

```bash
# 新 eval 单测（5/5）+ 全 harbor CPU 套件（81/81）
python -m pytest tests/test_harbor_eval_data_source.py -v       # 5 passed
python -m pytest tests/test_harbor_*.py -q                      # 81 passed

# bash / yaml 语法
bash -n run_qwen35_9b_terminal_bench2_64h20.sh                  # OK
bash -n run_terminalworld_full1530_..._gspo_lenreward_..._async.sh  # OK
python -c "import yaml; yaml.safe_load(open('...curriculum.yaml'))"  # OK

# dummy eval-prompt-data 满足 assert args.eval_datasets，/dev/null 不被存在性校验  # PASS
# 端到端：真实 TB2 加载 89 题、全量 eval pass、训练 sample_offset 零污染          # PASS
```

新单测覆盖：(a) eval 返回全量、不动训练指针；(b) `n_samples_per_eval_prompt` 生效；(c) 回退到 `n_samples_per_prompt`；(d) 没配 eval_dir 时 `eval_entries==[]` 且训练路径不变、eval 报 ValueError；(e) eval 计数器跨 pass 独立递增。

---

## 6. 下次训练会发生什么

- **step 0**：先在 TB2（89 题）测 baseline → wandb `eval/tb2`（= pass rate）出 t=0 点；
- 之后**每 10 个 rollout step** 测一次（step 10/20/.../90，约 10 次）；
- 每次 89 个 sandbox（每题 1 次），≈ 一个训练 step rollout 的开销；
- 训练**课程顺序、reward 曲线完全不受 eval 影响**（独立 eval 池 + 独立计数器）；
- eval 走同一套 label-based sandbox 清理，不泄漏。

### wandb 上会出现的 key
- `eval/tb2`：TB2 平均 reward（= pass rate，主指标）
- `eval/tb2-truncated_ratio`：截断比例
- `eval/tb2/*`：样本级 metrics（`compute_metrics_from_samples`）
- `harbor/eval_*`：eval 内部统计（任务数、转换数、失败数等）

---

## 7. 开关与可调项

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `EVAL_INTERVAL` | `10` | 每几个 rollout step 测一次；**设为空 = 完全关闭 eval**（回到旧行为） |
| `N_SAMPLES_PER_EVAL_PROMPT` | `1` | 每题采样次数（pass@k 时调大，但开销×k） |
| `SKIP_EVAL_BEFORE_TRAIN` | `0` | `1` = 跳过 step0 baseline |
| `EVAL_DATASET_NAME` | `tb2` | wandb 里 eval 曲线的名字（`eval/<name>`） |
| yaml `harbor_eval_tasks_dir/glob` | TB2 路径 | eval 任务池；改它即可换 eval 集 |

### 已知项 / 未来可选
- **eval 采样温度**：当前沿用训练温度（Harbor 路径不消费 `--eval-temperature`）。若要 greedy eval（temp=0），需在 `rollout.py` 把 eval 采样参数透传进 agent `llm_kwargs`——本次未做，列为可选后续项。
- **fully_async 不支持 eval**：这是按设计的限制；eval 走标准 `generate_rollout` 是正解，无需改 fully_async。

---

## 8. 涉及文件清单

**改动（plugin）**
- `slime_plugins/harbor/data_source.py`
- `slime_plugins/harbor/rollout.py`

**新增（测试）**
- `tests/test_harbor_eval_data_source.py`

**改动（测试，签名同步）**
- `tests/test_harbor_rollout_async.py`

**改动（配置）**
- `examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh`
- `examples/harbor_terminal_rl/run_terminalworld_full1530_96gpu_27b_daytona_n20_partial16_gspo_lenreward_fully_async.sh`
- `examples/harbor_terminal_rl/harbor_terminalworld_full1530_96gpu_27b_daytona_n20_partial16_gspo_lenreward_curriculum.yaml`

**未改**：slime core（`slime/**`）全程 0 改动。
