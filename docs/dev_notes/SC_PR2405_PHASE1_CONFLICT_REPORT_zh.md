# PR #2405 (Score Centering) 合入 —— 阶段一：冲突排查与隔离架构设计

**日期**：2026-09-25
**PR**：https://github.com/THUDM/slime/pull/2405 —— 28 文件，+1966 / −40
**基线**：slime-terminal-RL @ `2cf4f30`（worktree of slime v0.3.0，分支 `harbor/v030-port`）
**状态**：仅分析。**未修改任何代码，未创建任何补丁。**

---

## 0. 先说结论（TL;DR）

1. **上游与本地已结构性分叉。** PR 基于 upstream main（v0.3.x 之后若干个月），它依赖 4 个本地**根本不存在**的模块 / 基础设施：`slime/observability/`（整个包）、`slime/utils/tensor_store.py`（`DiskTensorRef`）、`slime/utils/routed_experts.py`、以及 `Sample.append_response_tokens()`。**这不是 cherry-pick，是移植（port）。**

2. **好消息：两个基线脚本的损失函数根本不走 PR 修改的那个函数。** 两脚本解析后都带 `--loss-type custom_loss --custom-loss-function-path slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled`。PR 对 `slime/backends/megatron_utils/loss.py::policy_loss_function` 的全部 +123 行改动，**在你的两个 run 里是死代码**。这是最强的天然隔离。

3. **坏消息：有 4 处"无开关的无条件改动"会落在基线路径上**，必须显式处理（详见 §3 的 C-1 ~ C-4）。其中 **C-1（sglang patch 改写 `output_top_logprobs` 语义）是唯一真正的破坏性风险**，且它在 Python 层之外。

4. **SC 本身在你的集群上目前跑不起来**，有 3 个硬阻断（§5）。这不影响"合入 + 向后兼容"这个目标，但你需要知道合进来的是一个暂时无法启用的特性。

---

## 1. 两个启动脚本的调用链（已实读确认）

```
run_27b_dense_128gpu_ppo_256k.sh            (541 行，仅 export env)
    └─ bash run_tmax15k_122b_a10b_..._partial_reward.sh   ← 你的第 2 个基线脚本
            └─ exec run_tmax15k_122b_a10b_..._strongest.sh
                    └─ exec run_qwen35_9b_terminal_bench2_64h20.sh   ← 真正拼 TRAIN_CMD
                            └─ ray job submit -- python3 train_async.py <args>
```

注意：**脚本 1 的父脚本就是脚本 2**。所以两者共享 95% 的 arg 解析路径，差异只在 env 覆盖（拓扑 / 上下文 / EP / ckpt / 数据池）。这是好事——只需保护一条链。

### 1.1 解析后与本次合入相关的实际参数值

| 参数 | 27B/256k | 122B/84k | 来源 |
|---|---|---|---|
| `--loss-type` | **`custom_loss`** | **`custom_loss`** | `strongest.sh:505` `USE_CUSTOM_NO_ENTROPY_LOSS=1` |
| `--custom-loss-function-path` | `slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled` | 同 | `64h20.sh:112` |
| `--advantage-estimator` | `ppo` | `ppo` | `strongest.sh:503` |
| `--use-rollout-logprobs` | **不传**（`USE_ROLLOUT_LOGPROBS=0`） | 同 | `strongest.sh:510` |
| `--use-tis` | **不传** | 同 | 无 env 钩子 |
| `--get-mismatch-metrics` | **不传** | 同 | `GET_MISMATCH_METRICS` 未设 |
| `--rollout-function-path` | `slime_plugins.harbor.rollout.generate_rollout` | 同 | `64h20.sh:372` |
| `--rollout-temperature` | `1.0` | `1.0` | `64h20.sh:379` 默认 |
| `--rollout-top-p` / `--rollout-top-k` | `1.0` / `-1` | 同 | `64h20.sh:380-381` |
| `--qkv-format` | `thd` | `thd` | `strongest.sh:443` |
| `--allgather-cp` | **不传**（默认 False，走 zigzag CP） | 同 | `arguments.py:261` |
| `--log-probs-chunk-size` | `1024` | 未设 → `-1` | 27B 脚本 `:237` |
| `--use-rollout-routing-replay` | **不传**（27B dense，`USE_ROUTING_REPLAY=0`） | **传**（`USE_ROUTING_REPLAY=1`） | 27B `:277` / parent `:289` |
| `harbor_sample_mode` | `trajectory_retokenized` | `trajectory_retokenized` | yaml |
| CP / TP / PP | 4 / 4 / 2 | 4 / … | 脚本 |

**关键推论**：两个 run 走的是 **`custom_loss` → Harbor 的 `train_loss.py` fork**，而不是 slime core 的 `policy_loss_function`。`train_loss.py` 是 `policy_loss_function` 在 v0.3.0 时刻的一份**手工 fork**（235 行，逐行对照几乎一致，仅加了 entropy 惰性计算 + `group_mask_sums` 修复）。

---

## 2. 受影响核心文件清单（按风险分级）

图例：🔴 落在基线执行路径上（必须加开关） · 🟡 落在路径上但只读/惰性（低风险） · 🟢 基线路径不经过 · ⬜ 本地缺失，需新建 · ❌ 本地缺失且非必需

| # | PR 文件 | 本地状态 | 落在基线路径? | 风险 |
|---|---|---|---|---|
| 1 | `slime/utils/score_centering.py` | ⬜ 新建 235L | 否（纯新模块） | 🟢 |
| 2 | `slime/utils/ppo_utils.py` | 存在 681L，+126 | 是（被 import） | 🟡 `importance_weights` 重写了 TIS 公式 |
| 3 | `slime/utils/types.py` | 存在 245L，+41/−2 | 是 | 🟡 `append_response_tokens` 本地**不存在**，该 hunk 无处可打 |
| 4 | `slime/utils/arguments.py` | 存在 1971L，+34/−2 | **是** | 🔴 `slime_validate_args` 无条件调用 2 个新校验 |
| 5 | `slime/utils/http_utils.py` | 存在 301L，+35/−7 | **是** | 🔴 `post()` / `_post()` / `do_post()` 三处签名变更 |
| 6 | `slime/ray/rollout.py` | 存在 1615L，+18/−6 | **是** | 🔴 `generate()` 无条件访问 2 个新 `Sample` 字段 |
| 7 | `slime/backends/megatron_utils/loss.py` | 存在 1357L，+123/−10 | **否**（custom_loss 绕过） | 🟢 但 `vanilla_tis_function`/`icepop_function` 被本地 fork 引用 |
| 8 | `slime/backends/megatron_utils/actor.py` | 存在 716L，+15 | 是 | 🟡 已 `if key not in rollout_data: continue`，天然安全 |
| 9 | `slime/backends/megatron_utils/model.py` | 存在 967L，+2 | 是 | 🟡 只往 `forward_step` keys 列表加 2 项 |
| 10 | `slime/utils/tensor_store.py` | ⬜ **本地完全缺失** | — | 🔴 PR 假设其存在（`DiskTensorRef`） |
| 11 | `slime/utils/routed_experts.py` | ⬜ **本地完全缺失** | — | 🟢 R3 走 `slime_plugins/harbor/routing_replay.py`，架构不同 |
| 12 | `slime/observability/{rollout,train}_data_utils.py`<br>`slime/observability/train_metric_utils.py` | ⬜ **整个包缺失** | — | 🟢 本地对应物在 `slime/utils/train_dump_utils.py`(22L) / `data.py::log_rollout_data` |
| 13 | `docker/patch/latest/sglang-top_p.patch` | 存在 935L（**未应用到运行时**） | **是（运行时语义！）** | 🔴🔴 见 C-1 |
| 14 | `docker/version.txt` | 存在，本地已改 | 否 | ⬜ 不合 |
| 15 | `.github/workflows/pr-test.yml{,.j2}` | 存在 | 否 | ⬜ 可选 |
| 16 | `tests/test_score_centering*.py` ×4 | ⬜ 新建 1045L | 否 | 🟢 建议全带上 |
| 17 | `docs/{en,zh}/get_started/*.md` | 存在 | 否 | 🟢 无风险 |

---

## 3. 逐一冲突点 + 隔离方案

### 🔴 C-1 — `sglang-top_p.patch`：**唯一真正的破坏性变更**（且在 Python 之外）

**PR 做了什么**：把 `tokenizer_manager.py` 的 top-logprobs 返回逻辑从"无条件 detokenize 并填 `meta_info["output_top_logprobs"]`"改成 `if not return_text_in_logprobs:` 分支——**不再填 `output_top_logprobs`**，改填 3 个新的二进制字段 `output_topk_{token_ids,log_probs,shape}`。

**为什么这是真风险**：这是**无条件的语义替换**，不受任何 slime 侧 `args` 控制。任何 `return_logprob=True` + `top_logprobs_num>0` + `return_text_in_logprobs=False` 的请求，其 `output_top_logprobs` 会**静默变空**。

**当前实测状态（已验证，这是转危为安的关键）**：
```
运行时 sglang 0.5.12.post1 @ /sgl-workspace/sglang/python/sglang
  output_topk_token_ids  : False   ← top_p.patch 未应用
  top_p_token_ids        : False   ← 同上
  routed_experts         : True    ← sglang.patch 已应用（R3 依赖，正常）
```
→ **`sglang-top_p.patch` 当前就没有生效**。它只存在于 `docker/patch/latest/`，Dockerfile:109 的 patch 循环里，但你们这套环境的 sglang 不是从那个 Dockerfile 层装的。

**并且**：Harbor 的 LLM 客户端（`slime_plugins/harbor/llm.py:119-124`）请求体里**从不带 `top_logprobs_num`**，只有 `return_logprob: True`。`llm.py:150` 也只读 `meta_info["output_token_logprobs"]`（sampled-token logprob，**不是** `output_top_logprobs`）。

**结论 + 方案**：
- 双重安全：(a) patch 未应用；(b) 即使应用，Harbor 不请求 top-logprobs 也不读 `output_top_logprobs`。
- **方案**：把 patch 文件更新合入（保持与上游同步），但**明确不重建镜像 / 不 re-apply patch**。在 `docker/version.txt` 附近或 patch 文件头加一段注释记录"本集群运行时未应用此 patch"。
- **硬要求写进合并说明**：*未来任何人若重建镜像并应用 `sglang-top_p.patch`，必须先确认没有代码依赖 `meta_info["output_top_logprobs"]`。* 我会在阶段三加一个 CPU 断言测试锁死这条契约。

---

### 🔴 C-2 — `arguments.py::slime_validate_args` 无条件调用新校验

**PR 改动**：
```python
def slime_validate_args(args):
    from slime.utils.ppo_utils import get_pg_loss_type
    from slime.utils.score_centering import validate_score_centering_args
    get_pg_loss_type(args)              # ← 无条件
    validate_score_centering_args(args) # ← 无条件
```
以及：
```python
-    if args.use_rollout_logprobs:
+    if args.use_rollout_logprobs and get_pg_loss_type(args) != "reinforce":
         assert not args.use_tis, ...
```

**风险分析（逐行验算）**：
- `validate_score_centering_args`：第一行 `if not getattr(args, "use_score_centering", False): return` → 默认 `False`，**立即返回，零副作用**。安全。
- `get_pg_loss_type(args)`：`configured=None`（`--pg-loss-type` 默认 None）、`sc=False`、`estimator="ppo"` → `objective = None or ("reinforce" if False else "cispo" if "ppo"=="cispo" else "ppo")` = `"ppo"`。两个 `if` 都不触发 → **返回 `"ppo"`，无异常**。安全。
- 修改后的 `use_rollout_logprobs` assert：基线 `use_rollout_logprobs=False`，短路，**assert 从来不执行**（改前改后都一样）。安全。

**但仍需加固的理由**：`get_pg_loss_type` 里有 `raise ValueError`。如果将来有人加了 `--advantage-estimator gspo` 或别的组合，这个无条件调用可能在**与 SC 完全无关的配置上**抛错。

**隔离方案**：
```python
def slime_validate_args(args):
    # SC gate: 完全不涉及 SC 的启动不进入任何 SC 代码路径。
    if getattr(args, "use_score_centering", False) or getattr(args, "pg_loss_type", None) is not None:
        from slime.utils.ppo_utils import get_pg_loss_type
        from slime.utils.score_centering import validate_score_centering_args
        get_pg_loss_type(args)
        validate_score_centering_args(args)
    ...
```
并且 `use_rollout_logprobs` 那条 assert **原样保留不改**（因为改动只在 `pg_loss_type=="reinforce"` 时才有意义，而那只能由 SC 开启）：
```python
    if args.use_rollout_logprobs and _pg_loss_type_or_ppo(args) != "reinforce":
```
其中 `_pg_loss_type_or_ppo` 是本地新增的零成本 helper：SC 关闭时直接 `return "ppo"`，不 import `score_centering`。

---

### 🔴 C-3 — `http_utils.py`：`post()` 三处签名变更

**PR 改动**：`post()` / `_post()` / `_DistributedPostActor.do_post()` 全部新增 `score_centering_top_k=None`。

**风险分析**：
- 纯新增 keyword-only 默认参数 → 现有调用点全部兼容。
- `_post` 里 `if score_centering_top_k is not None: ... else: output = json.loads(content)` → 默认走 `else`，**与原逻辑逐字节相同**。
- **真实风险在分布式 Ray actor 上**：`do_post.remote(..., score_centering_top_k=...)` 需要 actor 端的类定义也有该参数。若 **actor 进程用旧代码、driver 用新代码**（滚动升级 / stale `__pycache__`），Ray 会抛 `TypeError: got an unexpected keyword argument`。本地 `slime/ray/utils.py` 有 `_distributed_post_enabled` 路径，**且 `__pycache__` 是 checked-in 的**（CLAUDE.md 明确提到）。

**隔离方案**：
1. Driver 侧只在 SC 开启时才传该 kwarg（**默认完全不出现在 `.remote()` 调用里**）：
   ```python
   extra = {"score_centering_top_k": k} if k is not None else {}
   obj_ref = actor.do_post.remote(url, payload, max_retries, headers=headers, timeout=timeout, **extra)
   ```
   这样基线的 `.remote()` 调用签名与合并前**完全一致**，跨版本 actor 不可能报错。
2. Actor 侧签名用 `**kwargs` 兜底吸收未知参数，配合显式取值。
3. 合并后清理 `slime/**/__pycache__/*.pyc`（`PYTHONPYCACHEPREFIX=/tmp/...` 重新编译）。

**注意**：Harbor 的生成请求**根本不走 `http_utils.post`** —— `llm.py:127-128` 自己 `httpx.AsyncClient().post()`。所以两个基线脚本的 rollout 生成路径**完全不碰这个文件的改动**。`post()` 在基线里只用于 SGLang 的控制面调用（flush_cache / update_weights 等）。

---

### 🔴 C-4 — `slime/ray/rollout.py::generate()` 无条件访问新 Sample 字段

**PR 改动**：
```python
disk_routes = [
    value
    for sample in data
    for value in (sample.rollout_routed_experts, sample.rollout_topk_token_ids, sample.rollout_topk_log_probs)
    if isinstance(value, DiskTensorRef)
]
```

**风险分析**：
- 这里是**属性直接访问**（`sample.rollout_topk_token_ids`），不是 `.get()`。
- `Sample` 是 `@dataclass`，PR 同时在 `types.py` 加了这两个字段并给了 `= None` 默认值 → 属性存在，访问安全。
- **但**：本地 `types.py:130-135` 有自定义 `__getattribute__` / `__setattr__`（用于 deprecation warning），需确认不会对新字段有额外行为。
- **更大的问题**：这段 PR 代码引用了 `DiskTensorRef`，而 `slime/utils/tensor_store.py` **本地不存在**（见 C-5）。本地对应的 `generate()` 现在根本没有 `disk_routes` 这段逻辑（R3 走 `slime_plugins/harbor/routing_replay.py` 的 numpy 内存路径）。

**隔离方案**：**不移植这个 hunk。** 本地无 `DiskTensorRef` / 无 R3 spill-to-disk 架构，该 hunk 的前提不成立。SC 数据在本地走内存 numpy 路径（与本地 R3 一致）。

同时 `_convert_samples_to_train_data` 的 hunk 本身已是正确的 gating 写法（`if getattr(self.args, "use_score_centering", False):`），**照搬**。`_split_train_data_by_dp` 只往 key 列表加 2 项，且循环体有 `if key not in data: continue` → 安全，照搬。

---

### ⬜ C-5 — `slime/utils/tensor_store.py` / `DiskTensorRef` 本地缺失

**PR 假设**：sampler top-k 头可以 spill 到磁盘（safetensors），通过 `DiskTensorRef` 传引用而非数据，并支持 `__getitem__(slice)` 按行读取。PR 有 +48 行是**扩展**这个已存在文件（`__len__`、`__getitem__`、`retain_debug_tensor_refs`）。

**本地情况**：文件不存在。本地 R3 架构完全不同 —— `slime_plugins/harbor/routing_replay.py` 用 in-memory `np.ndarray`，`actor.py:297-300` 用 `torch.from_numpy(r)` 直接转。`rollout_routed_experts_store_dir` 这个 arg 本地也不存在（grep = 0）。

**方案（二选一，需你定）**：
- **方案 A（推荐，低风险）**：**不移植 `tensor_store.py`**。SC 的 sampler head 走纯内存 numpy 路径，对齐本地 R3。删掉 `score_centering.py` 里所有 `DiskTensorRef` 分支（`validate_sampler_topk` 的 disk 分支、`spill_sampler_topk` 整个函数）。
  - 代价：**显存/主机内存压力**。实测算：k=128 时每 token 1024 B（int32 ids + fp32 logps）。27B/256k 一个 rollout batch（~718 samples × 平均 40k resp tokens）≈ **27 GiB 主机 RAM**，全部经 Ray object store 传到 trainer。这是 SC 在 256k 上的真实成本。
  - 但因为 SC 默认关闭，**这个成本在基线上是 0**。
- **方案 B（完整但工作量大）**：新建 `slime/utils/tensor_store.py`（需从 upstream 取完整文件，不只是 diff），再新增 `--rollout-routed-experts-store-dir` arg。牵连面大，且与本地 Harbor R3 架构双轨并存。

我建议 **A**，并在文档里写明"本 port 的 SC 不支持 disk spill；在 256k 长上下文上启用 SC 前需评估 ~27 GiB 主机内存开销"。

---

### 🟡 C-6 — `ppo_utils.py::importance_weights` 重写 TIS 公式

**PR 改动**：把 `loss.py` 里两处 inline 的 clamp / where 换成新的公共 `importance_weights()`。

**数值等价性验算**：
| 原式 | 新式 | 等价？ |
|---|---|---|
| `torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)` | `importance_weights(tis, "tis", low=tis_clip_low, high=tis_clip)` → `ratio.clamp(min=low, max=high)`（low/high 非 None） | ✅ 逐字节相同 |
| `torch.where((r>=low)&(r<=high), r, zeros_like(r))` | `importance_weights(r, "mis", low, high)` → `torch.where((r>=low)&(r<=high), r, 0.0)` | ✅ 语义相同（`0.0` 标量广播 vs `zeros_like`，dtype/device 由 `r` 决定，结果一致） |

**但**：新版的 `mode="tis"` 在 `low is None` 时默认 `0.0`、`high is None` 时默认 `2.0`；`mode="mis"` 默认 `0.5/5.0`。**调用点始终显式传 low/high**，所以默认值不会生效。安全。

**额外关键点**：本地 `slime_plugins/harbor/train_loss.py:26` **import 了 `vanilla_tis_function`**。所以对它的修改**会**影响基线路径——但只在 `args.use_tis or args.get_mismatch_metrics` 为真时才被调用，**两个基线脚本均为 False**，该分支不执行。

**隔离方案**：新增 `importance_weights` 作为纯新增函数（无风险），但 **`vanilla_tis_function` / `icepop_function` 的函数体保持原样不改**。理由：改动是纯重构、零收益、却触及基线 import 的符号。SC 自己的代码直接调 `importance_weights`，不需要这两个函数被重构。→ **减少 4 行 diff，换取零风险。**

---

### 🟡 C-7 — `types.py`：`append_response_tokens` 本地不存在

**PR 改动**：在 `Sample.append_response_tokens()` 里加 +35 行 SC head 拼接逻辑，并在 `_validate_response_metadata_lengths()` 里加长度校验。

**本地情况**：两个方法**都不存在**（grep = 0）。本地 `Sample` 只有 `update_from_meta_info` / `get_reward_value` / `effective_response_length` 等。Harbor 自己在 `atif_adapter.py` 里手工装配 sample（`sample.tokens = ...`、`sample.rollout_log_probs = list(...)`、`sample.response_length = ...`），走的是完全不同的路径。

**方案**：
- **只移植 2 个 dataclass 字段**（带 `= None` 默认值）到 `Sample`：`rollout_topk_token_ids`、`rollout_topk_log_probs`。类型注解去掉 `DiskTensorRef`（方案 A 下不存在该类）。
- **不移植** `append_response_tokens` 的 hunk（无处可打）。
- **不移植** `_validate_response_metadata_lengths` 的 hunk（方法不存在）。改为在 `slime/ray/rollout.py::_convert_samples_to_train_data` 的 SC gate 里做等价长度校验（PR 那里本来就有 `validate_sampler_topk`）。
- 需验证本地 `types.py:130-135` 的自定义 `__getattribute__`/`__setattr__` 对新字段无副作用（阶段三第一步做）。

---

### 🟢 C-8 — `loss.py::policy_loss_function` +123 行：基线不执行，但需处理 Harbor fork 的分叉

**PR 改动**：`policy_loss_function` 加了 `pg_loss_type` 分派（`reinforce` 分支）、`get_score_centering_terms()` 新函数（+70L）、`_slice_allgather_response_rows()` 辅助、SC 指标上报。

**基线路径**：`--loss-type custom_loss` → `train_loss.py::policy_loss_no_entropy_when_disabled`。**core `policy_loss_function` 从不被调用。**（已确认：`64h20.sh:586` 在 `USE_CUSTOM_NO_ENTROPY_LOSS=1` 时无条件加 `--loss-type custom_loss`，而 `strongest.sh:505` 把它默认为 1。）

**这既是保护也是问题**：SC 移植进 core 后，**你的两个 run 即使加了 `--use-score-centering` 也不会生效**，因为它们走 fork。

**方案**：
- **core `loss.py`**：完整移植 SC 逻辑，全部包在 `if use_score_centering:` / `if pg_loss_type == "reinforce":` 分支里。`pg_loss_type` 默认解析为 `"ppo"` → 走原 `compute_policy_loss` 分支，**与合并前逐字节一致**。
- **`train_loss.py`（Harbor fork）**：**本阶段不改**。保持基线 100% 冻结。若你后续想在 Harbor run 上用 SC，那是一次独立的、需要单独评审的改动（我会在报告末尾的"后续"里标注）。
- `_allgather_cp_redistribute` 调用签名：PR 调用时**漏传 `args=`**（PR 写 `_allgather_cp_redistribute(res, logits_local_len=..., total_lengths=..., response_lengths=...)`，但本地 `loss.py:147-155` 该函数签名要求 `args` 且 `slice_log_prob_with_cp` 需要 `args.qkv_format`）。**上游该函数签名已变**。移植时必须补 `args=args, max_seq_lens=max_seq_lens`，否则 `TypeError`。← 这是一个必须修的移植 bug。

---

### 🟡 C-9 — `actor.py` / `model.py`：天然安全

**`actor.py` +15L**：
```python
for key, dtype in (("rollout_topk_token_ids", torch.int32), ("rollout_topk_log_probs", torch.float32)):
    if key not in rollout_data:       # ← 已有防御
        continue
```
`if key not in rollout_data: continue` 已经是你要求的"数据字典契约安全"写法。SC 关闭时 key 不存在 → 直接 continue，**零影响**。照搬，但需去掉 `DiskTensorRef` 分支（方案 A）。

**`model.py` +2L**：只往 `forward_step` 的 keys 列表加 2 个字符串。`DataIterator.get_next()`（`data.py:251-256`）对缺失 key 返回 `None`：
```python
vals = self.rollout_data.get(key, None)
if vals is None: batch[key] = None
```
→ SC 关闭时 `batch["rollout_topk_token_ids"] = None`，而 loss 里对应 `batch.get(...)` 也只在 SC 分支才读。**安全，照搬。**

**`data.py::log_rollout_data`**（本地对应 PR 的 `train_metric_utils.py`）：需把 2 个新 key 加进 skip 列表（`data.py:306-318`），否则 SC 开启时会尝试对 `[N,k]` 的 int32 tensor 做 `sum_of_sample_mean` → shape 错误。SC 关闭时 key 不存在 → 无影响。

---

### 🟢 C-10 — `slime/observability/*` 与 `routed_experts.py`：不移植

| PR 文件 | 本地对应物 | 决定 |
|---|---|---|
| `observability/rollout_data_utils.py` | 无（本地在 `slime/ray/rollout.py` 内联） | 不移植 |
| `observability/train_data_utils.py` (+67) | `slime/utils/train_dump_utils.py`（22 行，无 CP restore 机制） | 不移植 |
| `observability/train_metric_utils.py` (+2) | `slime/backends/megatron_utils/data.py::log_rollout_data` | **等价改动移植**（加 2 个 skip key） |
| `utils/routed_experts.py` (+6) | `slime_plugins/harbor/routing_replay.py`（架构不同，无 spill hook） | 不移植（方案 A 下 `spill_sampler_topk` 不存在） |

这些都只影响 debug dump / 指标，**不影响训练数值**。基线路径不经过（`save_debug_train_data` 只在 `args.save_debug_train_data is not None` 时触发，基线未设）。

---

## 4. 超参数开关设计（完整规格）

### 4.1 新增 CLI 参数

添加位置：`slime/utils/arguments.py` 的 `add_algo_arguments(parser)`（约 `:780` 处，紧跟 `--advantage-estimator` 之后，与 PR 一致）。

| # | 参数名 | 类型 | 默认值 | 默认行为 | 触发逻辑 |
|---|---|---|---|---|---|
| 1 | `--use-score-centering` | `store_true` | **`False`** | 关闭，全部 SC 代码路径不进入 | 唯一总闸。所有 SC 代码必须在 `if getattr(args, "use_score_centering", False):` 之内 |
| 2 | `--pg-loss-type` | `str`, `choices=["ppo","reinforce","cispo"]` | **`None`** | `None` → `get_pg_loss_type()` 推导为 `"ppo"`（因 `advantage_estimator="ppo"`） | 显式选择 PG 目标；`reinforce` 才走 `-adv*logp` |
| 3 | `--score-centering-top-k` | `int` | `128` | **不生效**（仅 SC 开启时读） | sampler head 宽度 k |

**为什么不加 `--advantage-type` / `--algorithm-type`**：你的要求里提到这几个候选名。我建议**严格对齐 PR 的 `--use-score-centering` + `--pg-loss-type`**，理由：(a) 未来与上游同步时零重命名成本；(b) `--advantage-estimator` 已存在且语义是"优势估计器"，`--pg-loss-type` 语义是"策略梯度目标"，两者正交，PR 的切分是对的；(c) 少发明一个名字，少一处漂移。若你偏好本地命名，我可以加 alias。

### 4.2 默认值等价性证明（逐参数）

对基线的两个 run（`advantage_estimator="ppo"`、未传任何新 flag）：

```
args.use_score_centering   = False   (store_true 默认)
args.pg_loss_type          = None
args.score_centering_top_k = 128     (未被读取)

get_pg_loss_type(args):
    configured = None
    sc         = False
    estimator  = "ppo"
    objective  = None or ("reinforce" if False else "cispo" if "ppo"=="cispo" else "ppo")
               = "ppo"
    if sc and objective != "reinforce":          → False, 不抛
    if configured is not None and estimator in ("gspo","cispo"):  → False, 不抛
    return "ppo"                                  ✅

policy_loss_function:
    pg_loss_type = "ppo"
    use_score_centering = False
    train_log_probs_for_tis:  if pg_loss_type=="reinforce" or not tis  → "ppo"≠"reinforce",
                              退化为原式 `if not train_log_probs_for_tis`  ✅
    分派:  if "ppo"=="reinforce" → F
           elif "ppo"=="cispo"   → F
           else → compute_policy_loss(...)         ✅ 与原版同一行
    sc_terms = {}                                  ✅ 不进 get_score_centering_terms
    log_probs_to_compare = log_probs if (use_rollout_logprobs=False or use_sc=False)
                                     → False → old_log_probs   ✅ 与原式相同
    for key,value in {}.items(): pass              ✅ 无新指标

slime_validate_args:  SC gate 为 False → 不 import score_centering, 不调 get_pg_loss_type  ✅

ray/rollout.py::_convert_samples_to_train_data:
    if getattr(args,"use_score_centering",False): → F, 跳过     ✅
    → train_data 不含 rollout_topk_*                            ✅

actor.py::_get_rollout_data:
    for key in (...): if key not in rollout_data: continue → 两次 continue  ✅

model.py::forward_step:
    keys 多 2 项 → DataIterator.get_next 对缺失 key 返回 None
    → batch["rollout_topk_*"] = None，loss 侧不读               ✅

http_utils.post(...):
    score_centering_top_k=None → else: json.loads(content)      ✅ 原逻辑
    .remote() 调用不含新 kwarg（方案 C-3.1）                     ✅

data.py::log_rollout_data:
    skip 列表多 2 项，但 rollout_data 里没有这 2 个 key → 无影响  ✅

vanilla_tis_function / icepop_function:  函数体不改              ✅
train_loss.py (Harbor fork):             完全不改                ✅
sglang-top_p.patch:                      文件更新，不 re-apply    ✅
```

**结论：两个基线脚本的执行逻辑 / 数据流 / 损失计算与合并前 100% 一致。**

### 4.3 数据字典契约安全清单

| 位置 | 新字段读取写法 | 安全性 |
|---|---|---|
| `Sample` dataclass | 2 个新字段，`= None` 默认 | dataclass 保证属性存在 |
| `ray/rollout.py` train_data 写入 | 整块包在 `if getattr(args,"use_score_centering",False):` 内 | SC 关闭时 key 根本不产生 |
| `ray/rollout.py::_split_train_data_by_dp` | key 列表 + `if key not in data: continue` | ✅ |
| `actor.py::_get_rollout_data` | `if key not in rollout_data: continue` | ✅ |
| `model.py::forward_step` keys | `DataIterator.get_next` → `.get(key, None)` | ✅ |
| `loss.py::get_score_centering_terms` | `batch.get("rollout_topk_token_ids")` + 显式 `raise` | ✅ 仅 SC 分支进入 |
| `data.py::log_rollout_data` | skip 列表（`for key,val in rollout_data.items()` 遍历实际存在的 key） | ✅ |
| Ray actor `do_post` | 方案 C-3.1：默认不传该 kwarg + `**kwargs` 兜底 | ✅ 跨版本安全 |

**所有跨 Ray 边界的新字段，SC 关闭时在 dict 里根本不存在** —— 比 `dict.get(k, None)` 更强的保证：老训练流的数据结构 **byte-identical**。

---

## 5. SC 在本集群目前无法启用的 3 个硬阻断（不影响合入，但你需要知道）

即使合入完成，**你现在也无法真正开启 `--use-score-centering`**：

1. **SGLang 侧不产出数据**（阻断级）。SC 需要 `meta_info["output_topk_{token_ids,log_probs,shape}"]`，这来自 `sglang-top_p.patch` 的新 hunk。实测运行时 sglang 0.5.12.post1 **没有应用该 patch**（`output_topk_token_ids` 不在 `tokenizer_manager.py` 里）。→ 需重建镜像，且重建会触发 C-1 的语义变更风险。

2. **Harbor rollout 不请求也不透传 head**（阻断级）。`slime_plugins/harbor/llm.py` 自建 httpx 请求，不带 `top_logprobs_num`，`llm.py:150` 只读 `output_token_logprobs`。要让 SC 在 Harbor 上工作，需改 `llm.py`（加 `score_centering_request`）+ `atif_adapter.py`（把 head 沿 `trajectory_retokenized` 的切分/重分词逻辑一路搬运，含 masked tool token 的 dummy 分布）。**这是一个独立的、工作量不小的 Harbor 适配项。**

3. **基线走 Harbor `train_loss.py` fork**，core 的 SC 分支不执行（C-8）。

**另外两个"能跑但需评估"的点**：
- `trajectory_retokenized` 会丢弃 / 重切 prompt 与 response token（`atif_adapter.py:840-870` 的 `prompt_suffix_dropped_tokens` / `response_prefix_dropped_tokens`），sampler head 必须逐 token 同步这些丢弃，否则 head 与 token 错位 → SC 的 correction 完全错误且**不会报错**。
- 内存：27B/256k 下 ~27 GiB/rollout 的主机 RAM（方案 A 无 disk spill）。

→ **我的建议**：本次合入的目标定位为 **"基础设施就位 + 零回归"**，把 SC 做成一个已合入、经 CPU 单测验证、但在 Harbor 路径上尚未接通的特性。Harbor 适配（上述 1 & 2）单独立项。**这完全满足你的阶段一/三要求，且风险最低。**

---

## 6. 阶段三实施顺序（待你确认后执行）

| 步 | 内容 | 涉及文件 | 验证手段 |
|---|---|---|---|
| 3.0 | 建立回归基线：dump 两个脚本的 `--dry-run` 完整 resolved TRAIN_CMD + `parse_args()` 后的 `vars(args)` 快照 | — | 合并后逐字节 diff，必须为空（除新增 3 个 key 及其默认值） |
| 3.1 | 参数定义与配置层解耦：3 个新 arg + `slime_validate_args` 的 SC gate + `_pg_loss_type_or_ppo` helper | `slime/utils/arguments.py` | 重跑 3.0 快照 diff |
| 3.2 | 独立纯函数移植：新建 `slime/utils/score_centering.py`（去 `DiskTensorRef`），`ppo_utils.py` 新增 `get_pg_loss_type` / `importance_weights` / `calculate_topk_log_probs` + 2 个缺失的 vocab-parallel helper（`_maybe_all_reduce`、`_get_vocab_parallel_rank_size`，本地 grep=0，需一并补） | `slime/utils/{score_centering,ppo_utils}.py` | `tests/test_score_centering{,_binary}.py` CPU 通过 |
| 3.3 | Rollout 与通信层兼容适配：`types.py` 2 字段、`ray/rollout.py` SC gate、`http_utils.py`（方案 C-3.1）、`data.py` skip 列表 | 4 文件 | `tests/test_score_centering_transport.py` + 重跑 3.0 |
| 3.4 | Trainer / Loss 主循环分支调度：`loss.py` SC 分支（含 `_allgather_cp_redistribute` 签名修正）、`actor.py`、`model.py` | 3 文件 | 全量 `python -m pytest tests -m unit`；对比 SC-off 下 `policy_loss_function` 的 AST/字节码等价性 |
| 3.5 | 收尾：`sglang-top_p.patch` 文件更新 + "未 re-apply" 注释 + 契约断言测试 + 文档（`docs/` 中英 + 本报告转正） | patch / docs / tests | — |

**每一步结束后我都会跑 3.0 的快照 diff**，一旦出现非预期差异立即停下报告。

---

## 7. 需要你决策的 3 个点

1. **C-5 内存策略**：方案 A（纯内存 numpy，不移植 `tensor_store.py`，SC 在 256k 上 ~27 GiB/rollout 主机内存）vs 方案 B（完整移植 disk spill 基础设施）。**我推荐 A。**
2. **参数命名**：严格对齐 PR（`--use-score-centering` / `--pg-loss-type`）vs 使用你提到的本地名（`--advantage-type` 等）。**我推荐对齐 PR。**
3. **Harbor 适配范围**：本次只做 core 合入（SC 在 Harbor 路径上暂不可用，见 §5）vs 本次一并改 `llm.py` + `atif_adapter.py` 打通 Harbor。**我推荐前者**，后者作为独立立项——因为它要动 `atif_adapter.py` 的 `trajectory_retokenized` 切分逻辑，而那正是两个基线脚本的**热路径**，改动风险与"零回归"要求直接冲突。

---

**阶段一结束。等待你的逐项审查与"方案确认，可以执行合并"指令。未修改任何代码。**
