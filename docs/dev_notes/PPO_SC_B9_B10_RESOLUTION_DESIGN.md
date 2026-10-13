# B-9 / B-10 彻底解决方案设计

**日期**：2026-09-26
**前置**：`PPO_SC_INTEGRATION_BLOCK.md`（B-9 修正、B-11 ~ B-17）、`PPO_SC_INTEGRATION_PLAN.md`
**状态**：**设计文档，未写任何实现代码**。等你确认后进入实现。

> **2026-09-26 更新**：已复核 `SC_PR2405_PHASE1_CONFLICT_REPORT_zh.md` 并把其可用部分融合进来，
> 见 **附录 A**。三点要看：(1) 该报告 §4.3 的 8 项数据契约经逐行复核成立，**直接作为 S2 的验收清单**；
> (2) 它的 C-1（SGLang 风险定性）**已被推翻**，与 B-9 是同一误判；
> (3) 它补上了本设计的一个真实漏洞 —— `--allgather-cp` 的 head 布局（**B-17**），修复方案见 **§A.6**。
> §2.4 的文件清单与 §3.2 的路线已按 **§A.5 / §A.7** 修订。

---

## 0. 排查结论先行：三点改变了方案走向

排查过程产出了三条硬证据，它们**否决了我原先的部分判断**，请先看这三条再看方案。

### 0.1 B-9 的"语义风险"不成立 —— 严重度 High → Low

patch 修改的那段代码**整体位于 `if top_logprobs_num > 0:` 之内**（运行时 `tokenizer_manager.py:1940`，patch 未改动该守卫），而 `io_struct.py` 把省略的 `top_logprobs_num` 归一化为 `0`。

**因此："不传 `top_logprobs_num`" 本身就是完备的隔离。** 不需要任何运行时开关去保护基线 —— 基线的请求根本进不到那个分支。这直接回答了你的问题 2 中"是否存在仅在开启 SC 时才生效的安全传参方式"：**有，而且就是不传参数**。

### 0.2 真正的阻断是 patch 文件本身（B-11），不是"没打 patch"

- 文件**被截断**：1175 行，末尾 hunk 少一行 context 且缺尾换行 → `git apply` 报 `corrupt patch at line 1176`。
- 补上换行后，**18 个文件里 11 个打不上**，含 SC 关键的 `tokenizer_manager.py`（3 hunk 挂 1）。
- `--3way` 不可用：`repository lacks the necessary blob`。

**并更正我先前的一处口误**：我曾说 "`tokenizer_manager.py` 在隔离下可干净应用"，那是 `--include` 过滤导致的空输出误读；正确结论是它**也失败**。

### 0.3 环境不可整体重建（B-15）

`/sgl-workspace/sglang` 有 **1480 行未提交的本地修改**（41 文件，含 delta 权重同步的 sglang 侧改动），无 commit / 无 stash / 无备份。**任何"重建干净 checkout"的方案都会丢失它们。**

→ 方案必须是**最小追加、绝不重建**。

---

## 一、B-9 解决方案：最小 patch + job 级环境切换

### 1.1 为什么不是"打上现有 patch"

现有 `sglang-top_p.patch` 做了两件不相关的事：**(a)** top-p nucleus replay（`rollout_top_p_token_ids`，本仓库 grep=0，**没有任何代码在用**）；**(b)** SC 需要的 top-k 二进制输出。

SC 只需要 (b)。而 (a) 牵连 `sampler.py` / `schedule_batch.py` / `sampling_batch_info.py` / 3 个 speculative worker —— **采样与调度热路径**，且这些正是打不上的文件。

**结论**：提取最小 hunk，面积从 18 文件 / 1175 行降到 **1 文件 / ~25 行**。

### 1.2 新 patch 的确切内容

**新文件**：`docker/patch/latest/sglang-score-centering-topk.patch`

目标：`python/sglang/srt/managers/tokenizer_manager.py`，两处改动。

```python
# 改动 1：import（文件头）
+import numpy as np

# 改动 2：在 `if top_logprobs_num > 0:` 块内（当前运行时 :1949-1959）
#         把两行赋值替换为二选一分支
     if top_logprobs_num > 0:                                    # ← 守卫不动
         if len(state.input_top_logprobs_val) > len(state.input_top_logprobs):
             ...                                                  # ← 不动
         meta_info["input_top_logprobs"] = state.input_top_logprobs

+        # Score Centering needs the sampler's *numeric* head, not detokenized
+        # text. Emit it as binary only for requests that (a) asked for top
+        # logprobs, (b) did not ask for text, and (c) are finished (or are
+        # streaming). Every other request keeps the original behaviour via the
+        # else branch below, so this cannot change any existing consumer.
+        if not return_text_in_logprobs and (
+            meta_info.get("finish_reason") is not None or state.obj.stream
+        ):
+            ids = np.asarray(state.output_top_logprobs_idx, dtype="<i4").reshape(-1, top_logprobs_num)
+            logps = np.asarray(state.output_top_logprobs_val, dtype="<f4").reshape(-1, top_logprobs_num)
+            if ids.shape != logps.shape:
+                raise ValueError("Sampler top-k ids/logprobs have different shapes")
+            meta_info["output_topk_token_ids"] = pybase64.b64encode(ids.tobytes()).decode("ascii")
+            meta_info["output_topk_log_probs"] = pybase64.b64encode(logps.tobytes()).decode("ascii")
+            meta_info["output_topk_shape"] = list(ids.shape)
+        else:
             if len(state.output_top_logprobs_val) > len(state.output_top_logprobs):
                 state.output_top_logprobs.extend(
                     self.detokenize_top_logprobs_tokens(...)      # ← 原逻辑，整体搬进 else
                 )
             meta_info["output_top_logprobs"] = state.output_top_logprobs
```

**比上游 PR 更保守的两点**：
1. `else` 分支**完整保留原逻辑** → 任何请求 top-logprobs 且 `return_text_in_logprobs=True` 的第三方消费者行为**逐字节不变**。
2. 不引入 `array` / `_b64_encode_int32` / `_encode_top_p_token_ids` 等 top-p replay 的辅助函数（`pybase64` 运行时已 import，实测 `pybase64 in tokenizer_manager.py = True`）。

**三重安全性论证**（为什么基线永不受影响）：

| 层 | 机制 | 证据 |
|---|---|---|
| 1 | 外层守卫 `if top_logprobs_num > 0` 未改动 | patch diff 只在块内 |
| 2 | 省略该参数 → 归一化为 0 | `io_struct.py`: `if self.top_logprobs_num is None: self.top_logprobs_num = 0` |
| 3 | Harbor 基线请求不含该参数，也不读 `output_top_logprobs` | `llm.py:119-124`, `:150` |

### 1.3 Job 级环境切换设计

**为什么只能是 job 级**（B-12）：SGLang 以 `multiprocessing.Process` 在 Ray actor **进程内** fork（`sglang_engine.py:50-64`），子进程继承父进程已 import 的模块对象。因此不能"只给 rollout actor 换 PYTHONPATH"。

**为什么 PYTHONPATH 可行**（B-16 实测）：stdlib `PathFinder` 在 `sys.meta_path` 中排在 editable install 的 `_EditableFinder` **之前**，实测整包影子可成功遮蔽。

**目录布局**

```
${SLIME_ENV_ROOT}/sglang_sc/                 # 影子根，一次性构建，所有节点共享（apdcephfs）
  └── sglang/                                # 完整副本（75 MB）+ 已打最小 patch
${SLIME_ENV_ROOT}/sglang_sc/.provenance      # 记录源 HEAD + 本地修改 hash + patch hash
```

**构建脚本**（新文件 `scripts/build_sglang_sc_shadow.sh`，**手动执行一次，不在训练路径上**）

```bash
#!/usr/bin/env bash
# Build the score-centering sglang shadow tree. Run ONCE, by hand.
# Never invoked from a training launch: a training run must never mutate its
# own runtime.
set -euo pipefail

SRC=${SRC:-/sgl-workspace/sglang/python/sglang}
DST=${DST:?set DST to the shared shadow root, e.g. .../sglang_sc}
PATCH=${PATCH:?path to sglang-score-centering-topk.patch}

# 0. MANDATORY: back up the 1480 lines of uncommitted local modifications
#    before touching anything (see PPO_SC_INTEGRATION_BLOCK.md B-15).
git -C "$(dirname "$(dirname "${SRC}")")" diff HEAD > "${DST}.local_mods_backup.patch"

# 1. Full copy. A PARTIAL shadow is forbidden: B-16 measured that a shadow
#    holding only the top-level __init__.py silently falls back to the real
#    tree for submodules, producing a mixed-provenance import graph.
mkdir -p "${DST}"
cp -a "${SRC}" "${DST}/sglang"

# 2. Apply the minimal patch to the COPY only. The live tree is never touched.
patch -p1 -d "${DST}" --forward --no-backup-if-mismatch < "${PATCH}"

# 3. Prove the patched copy actually emits the new fields, and that the
#    original path is untouched.
python3 - "${DST}" "${SRC}" <<'PY'
import sys, pathlib
dst, src = (pathlib.Path(p) for p in sys.argv[1:3])
patched = (dst / "sglang/srt/managers/tokenizer_manager.py").read_text()
live    = (src / "srt/managers/tokenizer_manager.py").read_text()
assert "output_topk_token_ids" in patched, "shadow did not get the patch"
assert "output_topk_token_ids" not in live, "live tree was mutated -- ABORT"
assert "if top_logprobs_num > 0:" in patched, "outer guard lost"
print("shadow verified")
PY

# 4. Record provenance so a later run can prove which tree it used.
{ echo "src=${SRC}"
  echo "src_head=$(git -C "$(dirname "$(dirname "${SRC}")")" rev-parse HEAD)"
  echo "patch_sha256=$(sha256sum "${PATCH}" | cut -d' ' -f1)"
  echo "built_at=$(date -Is)"
} > "${DST}/.provenance"
```

**切换点：唯一一处，在 `RUNTIME_ENV_JSON` 的 `PYTHONPATH` 前面**

`run_qwen35_9b_terminal_bench2_64h20.sh:777` 现状：
```
"PYTHONPATH": "/root/Megatron-LM:${HARBOR_SRC}:${SLIME_DIR}:${PYTHONPATH:-}",
```
改为（**唯一改动**）：
```
"PYTHONPATH": "${SC_SGLANG_PREFIX}/root/Megatron-LM:${HARBOR_SRC}:${SLIME_DIR}:${PYTHONPATH:-}",
```
其中在脚本上方解析：
```bash
# Score-centering runtime selection. SC_SGLANG_PREFIX is EMPTY unless score
# centering is explicitly enabled, so the baseline PYTHONPATH string is
# byte-for-byte what it is today.
SC_SGLANG_PREFIX=""
if [ "${ENABLE_SCORE_CENTERING:-0}" = "1" ]; then
  : "${SC_SGLANG_SHADOW:?ENABLE_SCORE_CENTERING=1 requires SC_SGLANG_SHADOW}"
  [ -f "${SC_SGLANG_SHADOW}/.provenance" ] \
    || _die "SC_SGLANG_SHADOW has no .provenance -- run scripts/build_sglang_sc_shadow.sh"
  grep -q 'output_topk_token_ids' \
    "${SC_SGLANG_SHADOW}/sglang/srt/managers/tokenizer_manager.py" \
    || _die "SC_SGLANG_SHADOW is not patched"
  SC_SGLANG_PREFIX="${SC_SGLANG_SHADOW}:"
  GRPO_ARGS+=(--enable-score-centering --sc-top-k "${SC_TOP_K:-32}")
fi
```

**零回归论证**：`ENABLE_SCORE_CENTERING` 未设 → `SC_SGLANG_PREFIX=""` → PYTHONPATH 字符串**与今天逐字节相同**，且不追加任何 CLI 参数。

**对 keepalive 零干扰**（已核实）：`keepalive_ctl.sh` 既不引用 sglang 也不设 PYTHONPATH（grep 为空），它只按 GPU 利用率跑占位负载。切换不触碰它，也不需要重启节点。

**对 Ray 集群零干扰**：切换只改 `ray job submit --runtime-env-json` 的一个 env var，**不重启 Ray head、不重启 raylet、不换镜像**。旧 job 与新 job 可在同一集群先后运行。

### 1.4 为什么不用 Docker 镜像 Tag 切换

你提到了"动态切换 Docker 镜像 Tag"。**不推荐**，三个理由：
1. 本集群的 Ray head 与 raylet 是**长驻**的（基线脚本 `CLEAN_START=0` / `USE_EXISTING_RAY_HEAD=1`，注释明确写"never tear it down"）。换镜像意味着重启全部 16 节点的 raylet，会触发 keepalive 保护窗口（启动期 ~0% util），风险远大于收益。
2. 重建镜像会重新执行 `Dockerfile:109` 的 patch 循环，而该循环包含**打不上的** `sglang-top_p.patch`（B-11）→ 构建即失败或产生 fuzz 混合体。
3. 影子方案的回退是"删掉一个 env var"，镜像方案的回退是"重启集群"。

---

## 二、B-10 解决方案：数据通道 + 切片对齐契约

### 2.1 `llm.py` 改造（请求 / 解析 / 零开销）

**新增构造参数**（`SlimeSGLangLLM.__init__`，与现有 `use_rollout_routing_replay` 同构）：
```python
sc_top_k: int | None = None,        # None => SC off
```
由 `slime_plugins/harbor/rollout.py:1503` 一带注入（`llm_kwargs.setdefault` 已是现成模式）：
```python
llm_kwargs.setdefault("sc_top_k", args.sc_top_k if args.enable_score_centering else None)
```

**请求端**（`llm.py:119-124` 附近）：
```python
payload = {"input_ids": prompt_token_ids,
           "sampling_params": sampling_params,
           "return_logprob": True}
if self._sc_top_k:                              # None / 0 => 不注入任何字段
    payload["top_logprobs_num"] = self._sc_top_k
    # return_text_in_logprobs 保持默认 False -> 走二进制分支。
    # 显式不传，避免与 sglang 默认值耦合。
if self._use_rollout_routing_replay:            # 已有，不动
    payload["return_routed_experts"] = True
```

**关闭 SC 时的零开销保证（逐项）**：
| 开销类型 | 保证 |
|---|---|
| 网络 | payload 字典**键集合与今天完全相同** |
| SGLang 计算 | `top_logprobs_num=0` → 不做 topk、不填 state |
| 响应体积 | 不产出 `output_topk_*` 三字段 |
| 内存 | `Sample.rollout_topk_*` 保持 `None` |
| Ray 序列化 | `None` 的开销可忽略（且 `to_dict` 是 `__dict__.copy()`，字段已在 dataclass 里） |

**解析端**（新函数，放在 `slime_plugins/harbor/sc_head_codec.py`，**独立模块便于单测**）：

```python
def decode_sampler_head(meta_info: dict, expected_rows: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode the binary sampler head into (ids[int32], logps[float32]).

    Two wire formats are accepted, in this order:

      1. `output_topk_{token_ids,log_probs,shape}` -- base64 little-endian
         int32 / float32, emitted by the patched sglang (design section 1.2).
      2. `output_top_logprobs` -- the unpatched JSON list of
         `[(logprob, token_id, text_or_None), ...]` rows. Slower and ~5-10x
         larger on the wire, but it lets stage 4b run on an unpatched engine.

    Raises on anything ambiguous rather than guessing: a wrong head is worse
    than no head, because it is not detectable downstream.
    """
```

**防下溢/防错的校验（在 decode 内，逐 chunk 以控内存）**：
```
shape == (expected_rows, k)                     # 行数必须等于本 turn 的 output token 数
ids   互异、>= 0、<= int32 max
logps 有限、<= 0、exp().sum(-1) <= 1 + 1e-4      # 是一个合法的子分布
dtype 严格 int32 / float32（不做隐式转换）
```

### 2.2 攻克核心风险：切片对齐契约

#### 2.2.1 先纠正一个事实（B-13）

我先前记录的"5 处切片点（`:426,429,439,444,832`）"**是错的**。实测：`atif_adapter.py` 有**三个彼此独立**的轨迹构建器，共 11 处并行数组变更点：

| 构建器 | 行 | 服务的 sample_mode | 变更点 | 是否需改 |
|---|---|---|---|---|
| `_trajectory_alignment_debug` | `:377` | **仅调试**（写 metadata，不产 `Sample`） | `:384,411,426,429,439,444` | ❌ 不需要 |
| **`_normalized_trajectory_samples_from_detail`** | `:731` | **`trajectory_retokenized`（生产基线）** + `trajectory_normalized` | `:779,823,832,866` | ✅ **必须** |
| `_stitch_turns_preserving_outputs` | `:1255` | `trajectory` / `last_turn` | `:1276,1328,1345,1394` | ⚠️ 可选 |

**我原先列的 5 处里有 4 处属于调试构建器** —— 改它们毫无意义。真正的生产路径只有 **4 处**。

#### 2.2.2 设计：`ParallelTracks` —— 让错位在类型层面不可表达

**不采用**"在 4 处机械复制切片逻辑"（你已明确否决，我同意：这正是静默错位的来源）。

**核心思想**：把 `response_ids / loss_mask / rollout_log_probs / head_ids / head_logps` 从**五个独立 list** 变成**一个对象的五条轨道**，并且**只暴露整体操作**。任何单独动一条轨道的写法都不存在。

**新文件**：`slime_plugins/harbor/parallel_tracks.py`

```python
@dataclasses.dataclass
class ParallelTracks:
    """Response-aligned tracks that can only be mutated together.

    Invariant, checked after every mutation:

        len(response_ids) == len(loss_mask) == len(rollout_log_probs)
                          == len(head_ids) == len(head_logps)          # when SC on

    The point is not convenience. `trajectory_retokenized` drops prompt
    suffixes, drops response prefixes and suffixes, splits chunks and
    retroactively re-masks tokens; each of those is a place where one track
    can move and another can't. Making the tracks inseparable means a
    misalignment cannot be *written*, rather than being written and then
    hopefully caught.
    """

    k: int | None = None                       # None => SC off, head tracks stay empty
    response_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    rollout_log_probs: list[float] = field(default_factory=list)
    _head_ids: list[np.ndarray] = field(default_factory=list)
    _head_logps: list[np.ndarray] = field(default_factory=list)

    # ---- the only four mutators -------------------------------------------

    def extend_action(self, *, token_ids, log_probs, head_ids=None, head_logps=None) -> None:
        """Append generated (trainable) tokens. loss_mask = 1."""

    def extend_context(self, *, token_ids) -> None:
        """Append replayed / environment tokens. loss_mask = 0, log_prob = 0.0,
        head = the dummy distribution (distinct ids, -log k each).

        The dummy head must be FINITE, not -inf: the reducer multiplies by
        loss_mask, and `0 * inf` is NaN. See score_centering.py's note.
        """

    def truncate(self, length: int) -> None:
        """Drop a suffix from EVERY track (retokenize drift, :429)."""

    def remask(self, start: int, stop: int) -> None:
        """Zero loss_mask and log_probs over a span, keeping heads in place
        (retroactive masking, :426). Heads stay because the rows still
        correspond to real tokens -- they are simply not trained on."""

    # ---- the only reader --------------------------------------------------

    def finish(self) -> TrackArrays:
        """Validate the invariant and return immutable arrays for the Sample."""
```

**关键设计决定与理由**：

| 决定 | 理由 |
|---|---|
| 只有 4 个 mutator，没有 `.append` / 无属性直写 | 5 条轨道不可能被单独推进 |
| `extend_context` **不接受** head 参数 | observation 的 head 由内部生成，调用方无从传错 |
| `truncate` 一次砍全部轨道 | `:429` 的 `[:matched_len]` 语义，不可能只砍一条 |
| `remask` 保留 head 行 | `:426` 的追溯 mask 只改 mask，不删 token；head 行与 token 仍一一对应 |
| chunk `split` = **新建一个 `ParallelTracks`** | `:823` 的"清空重建"语义；新对象天然五轨同步为空 |
| `finish()` 是唯一出口，内含断言 | 每个 chunk 落地前必过不变式 |
| `k is None` 时 head 轨道恒空 | SC 关闭时零内存、零分支成本 |

**生产构建器的改造形态**（`:731`，示意）：
```python
tracks = ParallelTracks(k=args.sc_top_k if args.enable_score_centering else None)
...
if transition["strategy"] == "split":
    chunks.append(tracks.finish())                       # 落地旧 chunk
    tracks = ParallelTracks(k=tracks.k)                  # 新 chunk，五轨同步归零
else:
    tracks.extend_context(token_ids=transition["context_ids"])      # 原 :832 三行 → 一行
...
tracks.extend_action(token_ids=turn.output_ids,                      # 原 :866 三行 → 一行
                     log_probs=_turn_log_probs(...),
                     head_ids=turn_head_ids, head_logps=turn_head_logps)
```

**净效果**：4 处变更点变成 4 次方法调用，**没有任何一处能单独操作某一条轨道**。

#### 2.2.3 三类丢弃操作的逐一映射

| 丢弃类型 | 当前代码 | `ParallelTracks` 对应 | head 行为 |
|---|---|---|---|
| `prompt_suffix_dropped` | 只动 `prompt_ids` | **不涉及**（prompt 不在 response 轨道里） | 无影响 |
| `response_prefix_dropped` | `:426` 追溯 mask | `remask(start, stop)` | 行保留、mask 归零 |
| `response_suffix_dropped` | `:429` `[:matched_len]` | `truncate(matched_len)` | 五轨同砍 |
| chunk `split` | `:823` 清空重建 | 新 `ParallelTracks` | 天然同步 |
| `context_tail` | `:439`（调试构建器） | `extend_context` | 哑分布 |

### 2.3 自动拦截：让 1 个 token 的偏移在 Step 0 崩溃

三道防线，**层层独立**（任一道单独都能抓到，不依赖其他两道）：

**防线 1 —— 写入期不变式**（`ParallelTracks.finish()`）
纯长度断言，`O(1)`。抓"轨道长度不等"。这一道让**大多数**错位在构造期就崩。

**防线 2 —— 跨通道对账**（已实现：`sc_terms.py::_check_head_alignment`）
`rollout_log_probs`（标量通道）与 head（矩阵通道）是**两条独立传输路径**。采样 token 若命中 head，两者给出的 logprob 必须一致（容差 1e-2）。

**这是唯一能抓到"长度对但内容错位"的手段** —— 防线 1 抓不到整体平移 1 行（长度不变）。已有测试：
- `test_misaligned_head_logprobs_are_detected`（值平移 → 抛错）
- `test_misaligned_head_ids_collapse_the_hit_rate`（ids 也平移 → `sc_head_hit_rate` 坍塌）

**Step 0 崩溃的保证**：`--sc-assert-alignment sample` 在**每个 micro-batch 的第一个样本**上运行 → 第一个 step 的第一个 micro-batch 就会检查。建议 Stage 4 全程用 `full`（每样本都查），稳定后再降到 `sample`。

**防线 3 —— 端到端计数守卫**（新增，Rollout 侧）
在 `ray/rollout.py::_convert_samples_to_train_data` 的 SC gate 内：
```python
for sample in samples:
    # Heads must exist for every sample and match response_length exactly.
    # Raising here (rollout side, before any Ray transport) localises the
    # failure to the adapter instead of surfacing it as a shape error deep
    # inside the loss three layers later.
    validate_sampler_head(sample, k=args.sc_top_k)
```
并上报 `rollout/sc_head_rows_total` 与 `rollout/sc_head_dummy_frac` 两个指标：若 dummy 占比与 `1 - mean(loss_mask)` 偏离超过 1%，说明 observation 与 head 的对应关系已经漂移。

### 2.4 涉及文件与函数签名清单

| 文件 | 改动 | 新增签名 |
|---|---|---|
| `slime/utils/types.py` | 改 | `Sample` +2 字段（`rollout_topk_token_ids`, `rollout_topk_log_probs`），默认 `None` |
| `slime_plugins/harbor/parallel_tracks.py` | **新建** | `ParallelTracks`（4 mutator + `finish`）、`TrackArrays` |
| `slime_plugins/harbor/sc_head_codec.py` | **新建** | `decode_sampler_head(meta_info, expected_rows, k)`、`dummy_head(rows, k)`、`validate_sampler_head(sample, k)` |
| `slime_plugins/harbor/llm.py` | 改 | `__init__(..., sc_top_k=None)`；请求条件注入；响应条件解析 |
| `slime_plugins/harbor/rollout.py` | 改 | `llm_kwargs.setdefault("sc_top_k", ...)` 一行 |
| `slime_plugins/harbor/atif_adapter.py` | 改 | `_normalized_trajectory_samples_from_detail` 改用 `ParallelTracks`（4 处 → 4 次调用） |
| `slime/ray/rollout.py` | 改 | `_convert_samples_to_train_data` SC gate + `_split_train_data_by_dp` key ×2 |
| `slime/backends/megatron_utils/actor.py` | 改 | `_get_rollout_data` head CP 切分（复用 `slice_log_prob_with_cp`） |
| `slime/backends/megatron_utils/model.py` | 改 | `forward_step` keys ×2 |
| `slime/backends/megatron_utils/data.py` | 改 | `log_rollout_data` skip 列表 ×2 |
| `docker/patch/latest/sglang-score-centering-topk.patch` | **新建** | 1 文件 / ~25 行 |
| `scripts/build_sglang_sc_shadow.sh` | **新建** | 影子构建 + 校验 + provenance |
| `examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh` | 改 | `SC_SGLANG_PREFIX` 解析 + PYTHONPATH 一处 |

### 2.5 单测覆盖方案（重点：极端 retokenized 轨迹）

**新文件 `tests/test_parallel_tracks.py`**（纯 CPU，无 Megatron）

| # | 测试 | 断言 |
|---|---|---|
| P-1 | 不变式：随机 200 次任意 mutator 序列 | 五轨长度恒等 |
| P-2 | `k=None` 时 head 轨道恒空 | 零内存，`finish()` 返回 `None` head |
| P-3 | `truncate` 砍全部轨道 | 逐轨长度与内容 |
| P-4 | `remask` 只改 mask/logprob，**保留 head 行** | head 行数不变、内容不变 |
| P-5 | `extend_context` 的哑分布**有限** | `isfinite` 全真；`exp().sum()==1`；ids 互异 |
| P-6 | **属性直写不可达** | `ParallelTracks` 无公开 `append`；`_head_ids` 私有 |
| P-7 | **黄金轨迹**：手工构造含 3 类丢弃 + 2 次 split 的 6-turn 轨迹 | 逐 token 比对 head 行与 `rollout_log_probs`，`allclose` |
| P-8 | 对 P-7 的每一个 mutator 调用点**注入 off-by-one** | 每一处都必须被防线 1 或 2 抓到（参数化，遍历所有注入点） |

> **P-8 是这套设计的核心验收**：它不是"测正确路径能跑通"，而是**穷举每一个可能错位的位置，证明每一处都有断言覆盖**。

**新文件 `tests/test_sc_head_codec.py`**

| # | 测试 | 断言 |
|---|---|---|
| C-1 | base64 round-trip（int32/float32 小端） | 逐字节 |
| C-2 | 行数不符 / k 不符 / dtype 不符 | 各自 `raises` |
| C-3 | 非法分布（logp>0、sum>1、含 inf、ids 重复、ids 负） | 各自 `raises` |
| C-4 | 未打 patch 的 JSON list 格式回退 | 与 base64 路径结果一致 |
| C-5 | `expected_rows=0`（空 turn） | 返回 `(0,k)` 空数组，不抛 |

**扩展 `tests/test_harbor_atif_adapter.py`**

| # | 测试 | 断言 |
|---|---|---|
| A-1 | `trajectory_retokenized` + SC 开启，极端轨迹 | 每个 `Sample` 的 head 行数 == `response_length` |
| A-2 | SC 关闭时 | `sample.rollout_topk_* is None`，且产出的 `Sample` 与今天**逐字段相同** |
| A-3 | `trajectory` / `last_turn` 模式 + SC 开启 | 明确抛错（不静默降级），错误信息指向未适配的构建器 |

---

## 三、落地节奏

### 3.1 我赞同你的判断，但要加一个前置和一个修正

**赞同**：先用档 0（合成 head，$q\equiv p$）在**未打 patch 的现有环境**里把 `atif_adapter.py` 的切分对齐、`Sample` 搬运、CP/TP 切分全部调通并跑通零回归，再接真实 SGLang。理由：档 0 下修正项**恒为 0**（已有 `test_sharp_distribution_drives_the_correction_toward_zero` 覆盖该代数性质），所以可以用"loss 与基线逐 bit 相同"作为**整条管道的正确性判据** —— 这是零风险且判据极强的组合。

**必须加的前置（B-15）**：在任何环境改动之前，**先备份 `/sgl-workspace/sglang` 的 1480 行未提交修改**。这是所有后续步骤的第 0 步。

**一个修正**：档 0 有一个它**测不到**的东西 —— $q$ 与 $p$ 的**真实差异**。档 0 下 `sc_head_hit_rate` 恒为 1、`sc_rho` 恒为 1、跨通道对账恒过。因此档 0 能验证"管道不错位"，但**不能**验证"解码出的 $q$ 是对的"。所以档 1（未打 patch 的 JSON list 格式，C-4 覆盖）不是可选项，而是档 0 与档 2 之间的**必经一步**。

### 3.2 五步路线

| 步 | 内容 | 环境 | 判据 | 风险 |
|---|---|---|---|---|
| **S0** | 备份 sglang 本地修改并校验可回放 | 现有 | 备份 patch 可 `--check` 通过 | 零（只读 + 写备份） |
| **S1** | `ParallelTracks` + `sc_head_codec` 纯函数 + 单测 P-1~P-8 / C-1~C-5 | 现有 | 全绿，特别是 **P-8 全注入点被抓** | 零（未挂载） |
| **S2** | `atif_adapter` 改用 `ParallelTracks`；`Sample` 字段；Ray/actor/model 数据通道 | 现有 | **A-2：SC 关闭时 `Sample` 逐字段与今天相同**；全仓库套件不退化 | 低（改生产构建器，但有 A-2 守门） |
| **S3** | **档 0 合成 head** 端到端 | 现有（不动 sglang） | `sc_correction` < 1e-5；loss 与基线相对误差 < 1e-6；实测主机 RAM | 零 |
| **S4a** | **档 1** 真实 $q$，`--sc-correction-scale 0.0`（影子） | 现有（请求 `top_logprobs_num`，读 JSON list） | 对账零触发；`sc_head_mass_q` > 0.99；`sc_rho ∈ [0.5,2]`；**梯度与 SC-off 逐 bit 相同** | 低（不改梯度） |
| **S4b** | 最小 patch + 影子树 + `--sc-correction-scale 1.0` | 影子 | canary：同 prompt 打/未打 patch 的 `output_token_logprobs` 逐 bit 一致；先 122B/84k（内存压力 1/3） | 中 |

**S4a 的价值**（这是我上面"一个修正"的落点）：它用**真实的 $q$** 验证解码与对账，但 `--sc-correction-scale 0.0` 保证梯度不变 —— 于是"解码正确性"与"梯度影响"被拆成两个独立可验证的步骤。而且它**不需要打 patch**，因为未打 patch 的 sglang 在 `top_logprobs_num > 0` 时**照样产出 `output_top_logprobs`**（原逻辑），只是格式是 JSON list。

---

## 四、需要你决策的 5 点

| # | 决策 | 选项 | 我的建议 |
|---|---|---|---|
| 1 | **B-15 备份** | 先备份 1480 行本地修改 / 直接开工 | **先备份**。这是不可逆风险，成本是一条命令 |
| 2 | **B-11 patch 路径** | P1 最小 patch（1 文件/25 行）/ P2 重建环境 / P3 永久档 0 | **P1**。P2 会丢 delta 权重同步；P3 让 SC 数学上退化为 no-op |
| 3 | **环境切换机制** | PYTHONPATH 影子（75 MB 副本）/ Docker Tag 切换 | **PYTHONPATH 影子**。集群长驻、`CLEAN_START=0`，换镜像要重启 16 节点 raylet |
| 4 | **非生产 sample_mode** | `trajectory`/`last_turn` 开 SC 时明确抛错 / 也改造 `_stitch_turns_preserving_outputs` | **先抛错**。基线是 `trajectory_retokenized`；多改一个构建器就多一处失步风险 |
| 5 | **档 1 是否必做** | S4a 必做 / 直接从档 0 跳到档 2 | **必做**。档 0 测不到"解码出的 $q$ 是否正确"（§3.1 的修正） |

---

**设计结束。未修改任何生产代码，未创建 commit，未触碰 `/sgl-workspace/sglang`（排查全程只用 `git apply --check` 等只读命令）。等待你的确认后进入实现。**

---

# 附录 A：融合 `SC_PR2405_PHASE1_CONFLICT_REPORT_zh.md` 的复核结论（2026-09-26）

按指令深度复核阶段一冲突报告并与源码逐条比对。本附录记录**哪些可直接采用、哪些必须弃用、以及它补上了本设计的哪个漏洞**。

## A.1 一句话结论

**报告的"隔离架构"骨架高度可用，且 Stage 1-3 已事实上采纳；其"数据流契约"可直接搬到 S2；但它的 C-1（SGLang 风险定性）已被推翻，且它对真正的阻断 B-11 / B-15 没有任何帮助 —— 因为它的设定是"SC 不真正启用"，从未尝试应用那个 patch。**

最重要的收获是它补上了本设计的一个漏洞：**`--allgather-cp` 的 head 布局（→ B-17）**。

## A.2 可直接采用的部分（已复核成立）

### (1) §4.3 的 8 项数据字典契约 —— **直接采用，构成 S2 的验收清单**

我逐行与源码复核，8 行全部成立。其中 6 行正是本设计 S2 的待做项，**报告为它们提供了已验证的写法**：

| 契约 | 源码复核 | 归属 |
|---|---|---|
| `Sample` +2 字段 `= None` | `types.py:130-135` 的 `__getattribute__`/`__setattr__` **只特判 `rollout_id`** → 新字段无副作用 | S2 |
| `ray/rollout.py` 写入整块 gate | — | S2 |
| `_split_train_data_by_dp` key + `if key not in data: continue` | 循环体确有该防御 | S2 |
| `actor.py::_get_rollout_data` + `if key not in rollout_data: continue` | — | S2 |
| `model.py::forward_step` keys → `get_next` 返回 `None` | `data.py:252-255` 确为 `.get(key, None)` | S2 |
| `data.py::log_rollout_data` skip 列表 | 遍历实际存在的 key | S2 |
| `sc_terms` 用 `batch.get` + 显式 `raise` | — | **Stage 3 已完成** |
| Ray `do_post` 条件传 kwarg | 成立，但**基线不经过** | 不适用（见 A.2(2)） |

**核心价值**：这条原则 —— *"SC 关闭时新 key 在 dict 里根本不存在，而不是存在但为 None"* —— 比 `dict.get(k, None)` 强，它让**老训练流的数据结构 byte-identical**。S2 的每一处都按此执行。

### (2) C-3 的结论：**`http_utils.py` 完全不需要改**（比报告更强的推论）

报告 C-3 指出 Harbor 的生成请求不走 `http_utils.post`。我已验证：
```
$ grep -n 'http_utils' slime_plugins/harbor/llm.py slime_plugins/harbor/rollout.py
(空)
$ grep -rn 'await post(' slime/
slime/rollout/sglang_rollout.py:71,201        ← 通用 rollout 路径，基线不用
```

**推论（报告未明说，但是本设计的直接收益）**：PR #2405 对 `post`/`_post`/`do_post` 的三处签名变更（+35 行）、以及报告 C-3 花了大段篇幅讨论的"跨版本 Ray actor `TypeError`"风险，**在本地方案中全部不存在**。

→ **§2.4 的文件清单删去 `slime/utils/http_utils.py`**。head 解析改在 `llm.py` 自己的 `httpx` 逻辑里（`sc_head_codec.decode_sampler_head`），不碰 slime 的 HTTP 层。

### (3) C-2 的 gate 写法 —— Stage 1 已采纳并加强

报告主张把无条件校验改为 gate 包裹。Stage 1 已实施，**并比报告更严格**：报告的 gate 条件是 `use_score_centering or pg_loss_type is not None`；我们没有 `pg_loss_type`（PPO 路线不需要），所以条件更简单，且**把校验点移到了 YAML 合并之后**（B-6，报告未涉及 —— 它没发现 YAML 合并在 `slime_validate_args` 内部）。

### (4) C-6 的判断 —— Stage 1-3 已采纳

报告主张 **不重构 `vanilla_tis_function` / `icepop_function`**，理由是本地 `train_loss.py:26` import 了 `vanilla_tis_function`，改它会触及基线 import 图。已复核该 import 确实存在。Stage 1-3 一行未动这两个函数。

### (5) C-8 的签名 bug —— **直接采用，是 B-17 修复的必要前提**

报告指出 PR 调用 `_allgather_cp_redistribute` 时漏传 `args=`。复核本地签名（`loss.py:148-155`）：
```python
def _allgather_cp_redistribute(res, *, logits_local_len, args: Namespace,
                               total_lengths, response_lengths, max_seq_lens=None)
```
`args` 是必需的关键字参数，且内部 `slice_log_prob_with_cp` 需要 `args.qkv_format`。**移植时必须补 `args=args, max_seq_lens=batch.get("max_seq_lens")`**。

## A.3 必须弃用的部分

### (1) ❌ C-1 的风险定性 —— 已推翻（同 B-9）

报告称该 patch 是"**唯一真正的破坏性变更**""**无条件的语义替换**"。证据显示不成立：被改代码整段位于 `if top_logprobs_num > 0:` 之内，且 **patch 未改动该守卫**（`grep 'if top_logprobs_num > 0' <patch>` 无匹配），而省略该参数会被 `io_struct.py` 归一化为 `0`。

**对本设计的影响**：§1.2 的"三重安全性论证"取代 C-1 的定性。真正的隔离来自**请求端不传参**，不是"不应用 patch"。

**保留 C-1 的一个产出**：它要求加 CPU 断言锁死契约。**仍然值得做**，但语义改为断言"外层守卫仍在 + `else` 分支保留原逻辑"，已写入 §1.2 的 S-3。

### (2) ❌ C-1 的方案"合入 patch 但不应用" —— 与当前目标不兼容

该方案在"SC 不真正启用"的设定下成立。我们要真正启用，必须应用 patch，于是暴露 **B-11（文件截断 + 11/18 文件打不上）** 与 **B-15（1480 行未提交修改）**。

**报告对这两项零帮助**，因为它从未 `git apply` 过，也未检查运行时工作区。→ §一 的 P1 最小 patch 方案不变。

### (3) ❌ C-5 的 `DiskTensorRef` —— 维持不移植

报告主张不移植 `tensor_store.py`，走纯内存 numpy。已采纳，并进一步用 `--sc-top-k 32`（默认）把 27B/256k 的主机内存从 27.4 GiB 降到 6.8 GiB。

### (4) ❌ 报告"5 处切片点"的数字 —— 已被 B-13 更正

报告与我先前的记录都写"5 处（`:426,429,439,444,832`）"。实测是 **3 个独立构建器 / 11 处**，且那 5 处里有 4 处属于 `_trajectory_alignment_debug`（**不产出 `Sample`**）。生产路径只有 `_normalized_trajectory_samples_from_detail` 的 **4 处**。

→ §2.2.1 的表为准。这说明该错误数字源自比本报告更早的排查，而非本报告独有。

### (5) ❌ 报告 §4.1 的参数命名（`--use-score-centering` / `--pg-loss-type`）

报告的参数集服务 **REINFORCE** 路线。本项目是 `PPO_SC.md` 的**保留 clip + 加性中心化**路线，故：
- 用 `--enable-score-centering`（已实施）；
- **不引入 `--pg-loss-type`** —— 它的存在意义是切换到 REINFORCE，与"保留 PPO clip"的前提冲突；
- **不引入独立的 SC epsilon** —— 中心化必须减去**优化器实际施加**的那个算子的期望，故复用 `--eps-clip`（B-1）。

## A.4 数学目标承载性：骨架可用，算子必须换

**问题**：报告的隔离设计能否在不破坏通信/数据流骨架的前提下承载修正后的 $\bar w = r\cdot\mathbf 1[\text{未截断}]$？

**答案：能，且已经在做。** 二者是**正交的两层**：

| 层 | 报告提供 | 本项目替换 |
|---|---|---|
| 数据流（Sample → Ray → actor → CP 切分 → batch） | ✅ 全部可用 | 不变 |
| 通信（请求 top-k / 解析 head） | 思路可用，`http_utils` 部分弃用 | `llm.py` + `sc_head_codec` |
| TP 词表 gather（`calculate_topk_log_probs`） | ✅ 可用（Stage 2 已移植） | 不变 |
| CP 布局（zigzag / allgather） | ✅ **补上了本设计的漏洞**（B-17） | 采纳 |
| **PG 算子与权重函数** | ❌ REINFORCE + `importance_weights` | **替换为 `ppo_effective_weight`** |
| **挂载点** | core `policy_loss_function` 单挂载 | **双挂载**（Harbor fork + core，共享 `pg_objectives`） |

**关键论证**：head 数据的形状契约（`[R_loc, k]` int32 + float32，与 `log_probs` 同行序）**与用什么权重函数无关**。`score_centering_correction` 消费这对张量，内部用哪个 $\bar w$ 是它自己的事。所以骨架可以原样用，算子层独立替换 —— 这正是 Stage 2/3 已经跑通 89 个测试的结构。

**一处报告未覆盖的断层（已由 Stage 3 补上）**：报告 C-8 承认"SC 移植进 core 后，你的两个 run 即使加了参数也不会生效"，但它的阶段三计划仍只改 core `loss.py`，仅在文字上建议"train_loss.py 本阶段不改"。**这是一个真实的调用盲区**：基线走 `--loss-type custom_loss`，只改 core 等于参数被接受但完全无效（静默失效）。

Stage 3 的解法（已实施）：`slime/utils/pg_objectives.py` 作单一真源 + 双挂载 + `assert_score_centering_was_applied` 运行时断言（开关开启但未被调用 → 崩溃）。**这是本项目相对该报告的实质增量。**

## A.5 §2.4 文件清单的修订

基于本附录，对 §2.4 做两处修订：

| 变更 | 文件 | 理由 |
|---|---|---|
| **删除** | ~~`slime/utils/http_utils.py`~~ | A.2(2)：Harbor 不走该路径，`post` 三处签名变更不需要 |
| **新增** | `slime/utils/score_centering_validate.py`（改） | B-17 修复第 1 级：`enable_score_centering and allgather_cp` → `raise` |
| **新增** | `slime/backends/megatron_utils/sc_terms.py`（改） | B-17 修复第 2 级：移植 `_slice_allgather_response_rows` + 修正签名的 redistribute 调用 |

## A.6 新增：§2.6 `--allgather-cp` 的 head 布局处理（B-17）

**问题**：`get_responses` 在 `allgather_cp=True` 时返回**连续**行区间，而 `actor.py` 用 `slice_log_prob_with_cp` 把 head 切成 **zigzag** 两段。行数可能相同而内容错位。基线不受影响（`allgather_cp` 默认 `False`，两个启动脚本及父链均无该参数，已 grep 确认）。

**两级修复**

**第 1 级（S1，零成本，立即拒绝）** —— `score_centering_validate.py`：
```python
if getattr(args, "allgather_cp", False):
    raise ValueError(
        "--enable-score-centering does not yet support --allgather-cp: the "
        "sampler heads are sliced into the zigzag CP layout by the actor, while "
        "get_responses returns contiguous rows under allgather-CP. Mixing them "
        "would silently misalign the correction. Remove --allgather-cp, or "
        "implement the redistribute path (PPO_SC_INTEGRATION_BLOCK.md B-17)."
    )
```
> **明确拒绝优于静默出错**：这与本项目一贯的处理一致（缺 head 时 `_require_heads` 抛错、非生产 sample_mode 抛错）。

**第 2 级（S2，解除限制）** —— `sc_terms.py`，移植 PR #2405 的两个部件：

```python
def _slice_allgather_response_rows(values, total_lengths, response_lengths, logits_local_len):
    """Slice full per-sample [R, ...] response tensors to this rank's rows under
    allgather-CP, mirroring the allgather branch of get_responses (loss.py:99).

    Under allgather-CP each rank owns a contiguous slice of the global sequence,
    so the head rows must be cut the same way -- not with the zigzag
    slice_log_prob_with_cp the actor applies for the default layout.
    """
```
以及在末尾（`allgather_cp` 为真时）把标量 correction 重分布回 zigzag：
```python
_allgather_cp_redistribute(
    res,
    logits_local_len=logits.size(1),
    args=args,                                  # ← PR 漏传；本地签名必需（报告 C-8）
    total_lengths=total_lengths,
    response_lengths=response_lengths,
    max_seq_lens=batch.get("max_seq_lens"),     # ← 同上
)
```
**为什么需要重分布**：`advantages` 是 zigzag 布局的，correction 必须与它同布局才能逐元素相乘。

**单测（扩展 `tests/test_ppo_sc_harbor_integration.py`）**

| # | 测试 | 断言 |
|---|---|---|
| G-1 | `allgather_cp=True` + SC（第 1 级修复后） | `raises ValueError`，信息含 `allgather-cp` |
| G-2 | 第 2 级实施后，CP=2 gloo，`allgather_cp` True vs False | 同一批样本的 correction（all-gather 后）`allclose` |
| G-3 | `allgather_cp=True`，head 故意平移 1 行 | 跨通道对账仍能抓到 |

## A.7 修订后的路线（替换 §3.2 的 S1/S2 描述）

| 步 | 内容（**粗体 = 本附录新增/修订**） |
|---|---|
| **S0** | 备份 sglang 1480 行未提交修改（B-15） |
| **S1** | `ParallelTracks` + `sc_head_codec` + 单测 P-1~P-8 / C-1~C-5；**加 B-17 第 1 级校验（allgather-cp 拒绝）+ G-1** |
| **S2** | `atif_adapter` 改用 `ParallelTracks`；**按 A.2(1) 的 8 项契约实施数据通道（6 项 TODO）**；**B-17 第 2 级（移植 `_slice_allgather_response_rows` + 修正签名的 redistribute）+ G-2/G-3**；**不动 `http_utils.py`** |
| **S3** | 档 0 合成 head 端到端 |
| **S4a** | 档 1 真实 $q$ + `--sc-correction-scale 0.0` |
| **S4b** | 最小 patch + 影子树 + `scale 1.0` |

## A.8 待你决策（在原 5 点之外新增 1 点）

| # | 决策 | 选项 | 建议 |
|---|---|---|---|
| 6 | **B-17 的处理时机** | S1 只加拒绝校验（S2 再解除）/ S1 就一次做完 redistribute | **先拒绝**。基线不用 `allgather-cp`，S2 有更高优先级的契约工作；redistribute 需要多进程 CP 测试（G-2），放 S2 更稳 |

