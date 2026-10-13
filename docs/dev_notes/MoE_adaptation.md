# MoE 适配差异清单：qwen3.5 专用分支 vs. slime-terminal-RL

> 本文对比两个 slime 分支中**与 MoE 模型训练相关**的适配差异：
>
> - **A 树 —— `slime-terminal-RL`**（本仓库，Harbor 终端 agent RL，主打 dense 模型，但也有一套 MoE 跑法）
>   源码根：`slime-terminal-RL/slime/`
> - **B 树 —— `slime_qwen3.5`**（专门为 Qwen3.5-35B-A3B 这类 MoE 模型做的分支）
>   源码根：`slime_qwen3.5/slime/slime/`（注意 B 树多套了一层目录）
>
> **一句话结论**：两树共享**全部 MoE 底座**（模型 shape、`qwen3_5` spec、`qwen3moe/glm4moe/qwen3_5` 权重桥、EP 参数、R3 routing-replay，逐字节相同，均来自上游 slime v0.3.0）。差异集中在 B 树**额外**为 MoE RL 加的一套**研究性/工程性工具链**：MoE 路由 staleness 分析、大 MoE 的异步 checkpoint 转换 + 保存锁、权重同步显存泄漏排查、staleness sweep 脚本。而 A 树反而带了若干 B 树没有的健壮性补丁（NCCL/Gloo 超时、MTP/VLM 桥补丁等），但那些**不是 MoE 适配**。
>
> ⚠️ **对"terminal-RL 只针对 dense"这一前提的更正**：terminal-RL **确实有** MoE 跑法（`run_full89_grpo_96gpu_35b_a3b_...sh` 训练 Qwen3.5-35B-A3B，`TP2×PP1×EP8`、DeepEP flex dispatcher、SGLang dp-attention+EP、可选 R3）。所以准确的说法是：**两树都能训 MoE，但只有 B 树带 MoE 专用的分析与大规模 checkpoint 工具链**。下文按重要性从高到低列出所有不一致内容。

---

## 差异总览（按重要性排序）

| # | 差异 | B 树（qwen3.5） | A 树（terminal-RL） | 对 MoE 训练的重要性 |
|---|---|---|---|---|
| 1 | **MoE 路由 staleness 分析子系统** | `slime/staleness/`（含 52KB 的 `moe_staleness_hook.py`） | **完全没有** | ★★★★★ 核心 MoE 适配 |
| 2 | **大 MoE 异步 checkpoint 转换 + 保存锁** | `ckpt_async_convert.py` + `save_lock.py` | **完全没有** | ★★★★☆ |
| 3 | **actor 权重同步显存泄漏排查/清理** | `actor.py` 加了 `_inspect_memory` + `_cached_tensors.clear()` | 无（`actor.py` 无此逻辑） | ★★★★☆ |
| 4 | **staleness / sweep 参数与运行脚本** | `--enable-staleness-analysis` 等 4 个参数 + 3 个 sweep 脚本 | 无 | ★★★☆☆ |
| 5 | **math 自定义奖励** | `custom_rewards/prime_math_reward.py` | 无 | ★☆☆☆☆ |
| 6 | **（反向）A 树独有的健壮性补丁** | 无 | NCCL/Gloo 超时、MTP/VLM 桥补丁、EV、critic ckpt 隔离 | 非 MoE 适配，但部分对 MoE 有益 |
| — | **MoE 底座（模型/spec/桥/EP/R3）** | 逐字节相同 | 逐字节相同 | 无差异 |

---

## 1. ★★★★★ MoE 路由 staleness 分析子系统（B 树独有）

**路径（仅 B 树）**：`slime_qwen3.5/slime/slime/staleness/`
- `__init__.py`、`tracker.py`（异步 JSONL 写入器）
- `moe_staleness_hook.py`（**52 KB，核心**）
- `policy_staleness_hook.py`
- `training_inference_discrepancy.py`

A 树**完全没有** `staleness/` 目录。这是 B 树存在的根本目的，也是**唯一真正 MoE-specific 的适配**。它通过 slime 的公开钩子（`--rollout-data-postprocess-path` / `--custom-megatron-before-train-step-hook-path` / `--custom-tis-function-path`）挂入，不改核心。三个模块：

### Module C —— MoE 路由 staleness（`moe_staleness_hook.py`，真正的 MoE 专用件）

对比**SGLang 在 rollout 时选的专家** vs **Megatron 在训练时选的专家**（逐 token、逐 layer）。三个指标：

1. **专家重叠 / Jaccard**（`compute_and_log_moe_overlap`）
   - 推理侧路由来自 SGLang：`rollout_data["rollout_routed_experts"]`，形状 `[total_len-1, num_layers, topk]`；由 `postprocess_save_routing` 在 `fill_routing_replay` 删除它之前先存到 CPU。
   - 训练侧路由：在 Megatron 的 `TopKRouter` 模块上挂**前向 hook**（`register_moe_forward_hooks`），从 `routing_map` 抽 top-k 专家索引。
   - 逐 layer 算 Jaccard `|inf∩train| / |inf∪train|`。对齐代码高度 MoE/并行感知：要还原 **sequence-parallel 切分**、`--use-dynamic-batch-size` 下 `get_seqlen_balanced_partitions` 的 microbatch 重排、每 sample 的 prompt/response 偏移，才能把两个路由张量逐 token 对齐（还要处理 SGLang 只发 `len(tokens)-1` 条、最后一个 response token 是假 pad 路由）。
2. **Router-Shift Ratio γ**（`compute_and_log_router_shift_ratio`，实现 RSPO 论文 arXiv:2510.23027）
   - 从 `module.gating(hidden_states)` 重算 router logits，存 step-0 作为"旧策略"参考，`γ = exp(-Δ)`，`Δ = mean_l (1/K) Σ_k |log r_new(e_k) - log r_old(e_k)|`；输出 `clip_frac_0.8 / clip_frac_0.5`。
3. **Routing Agreement Rate (RAR)**（`compute_and_log_routing_agreement`）
   - router **想选的** top-K vs **实际用的**路由（R3 下是被回放的推理路由）的重叠；R3 开启纠偏时 RAR<1，关闭时 =1。

**Token 采样**：`--staleness-moe-sample-fraction`（默认 0.1）。注意设计细节——Jaccard 路径存全部路由（对齐需要完整位置信息，采样推迟到比较时），而 logit/RSPO/RAR 路径用**按 layer 名确定性 seed**（`gen.manual_seed(hash(layer_name) % 100000)`，不按 step 变）采 10%，保证跨 step 比较的是同一批 token 位置。

**为什么对 MoE RL 至关重要**：256 专家 / top-8 的模型做 off-policy RL 时，router 本身是个移动靶。若训练时路由与 SGLang 生成轨迹时的路由发散，GRPO/GSPO 的重要性采样假设在**专家层级**（而不仅 token 层级）就破了。这些指标正是量化这种专家选择漂移的工具 —— 也是 R3（routing replay）与 TIS 想缓解的现象。

### Module A —— 策略 staleness（`policy_staleness_hook.py`）

跨 mini-step / 跨 `update_weights_interval` 窗口，跟踪 **router 权重梯度的余弦相似度**。monkey-patch `optimizer.step()` 与 `zero_grad_buffer()`，在 Megatron 分布式优化器（`--optimizer-cpu-offload --use-precision-aware-optimizer`）清梯度**之前**快照 router 参数（`ROUTER_PARAM_PATTERN = re.compile(r"(router|gate).*weight")`）的 `main_grad`。用大小为 `update_weights_interval` 的环形缓冲比较"当前 vs 若干步前"的梯度方向。也拥有 `before_train_step_hook` / `after_last_train_step_hook` 并分发到 Module C。

### Module B —— 训练/推理差异（`training_inference_discrepancy.py`）

自定义 TIS 函数（`--custom-tis-function-path`）。算 IS 修正因子 `w_IS = exp(train_logp - rollout_logp)`、极端 token 比例 `F(τ)`（τ∈{1.5,2,3,5,10}）、`KL(train‖rollout)`、abs-log-ratio 百分位。可选做 vanilla TIS reweight，但在 IcePop/KPop 模式下不 reweight（它们自带 0/1 mask）。非严格 MoE-specific，但属同一套 staleness 套件。

### 参数（B 树 `slime/utils/arguments.py` 新增 `add_staleness_analysis_arguments`）

```
--enable-staleness-analysis
--staleness-output-dir        (默认 "staleness_results")
--staleness-modules {policy,discrepancy,moe}+   (默认三个都开)
--staleness-moe-sample-fraction   (默认 0.1)
```

这是 B 树对 `arguments.py` **唯一的功能性新增**。它挂的三个钩子在 A 树里也都存在（上游就有），只是 A 树从不用来做这件事。

---

## 2. ★★★★☆ 大 MoE 的 checkpoint 工具链（B 树独有）

**路径（仅 B 树）**：`slime_qwen3.5/slime/slime/utils/ckpt_async_convert.py`、`.../slime/utils/save_lock.py`。A 树无。

### `ckpt_async_convert.py` —— 后台 torch_dist→HF 转换

每次 `save_model()` 后（`train.py` 与 `train_async.py` 都调用），`maybe_convert_after_save(...)` 启动一个 **纯 CPU**（`CUDA_VISIBLE_DEVICES=""`）后台子进程跑 `tools/convert_torch_dist_to_hf.py`，把刚存的 `iter_NNNNNNN/` 转成 `iter_NNNNNNN_hf/`。会回收上一次转换（成功后删源 torch_dist 目录）、幂等、不并发第二个转换；关停时 `wait_for_previous()` 阻塞以保证最终 ckpt 是 HF 形式。

**为什么 MoE 需要**：35B-A3B / 256 专家的 checkpoint 巨大、转 safetensors 很慢；内联转会卡训练循环，在 GPU 上转会与训练/rollout 争抢。eval（SGLang）又需要 HF 格式权重。于是把"可恢复的 torch_dist 保存"与"eval 就绪的 HF 导出"解耦，不烧 GPU、不阻塞主循环。

### `save_lock.py` —— 跨进程保存锁

`acquire_save_lock(...)` 基于 `filelock` 在共享 cephfs 上（`<ckpt_root>/.global_save.lock`）加锁，包住两个训练循环里整个 `save_model` 块。docstring 明确点出 MoE 动机：

> 每个 Megatron 训练 actor 在保存时会飙到 **~177 GB CPU 内存**（fp32 param-buffer 备份 + state_dict gather）。单训练节点约 2.2 TB RAM —— 两个同调度的 8-actor 作业就足以超阈值、被 Ray 杀 worker。

每 30s 轮询、记录当前持有者、有 2 小时安全阀旁路。锁**放在转换调用之外**，这样别的作业的保存能在本作业 HF 转换进行时继续。

**为什么 MoE 需要**：大 EP 模型的 fp32 参数+优化器备份极大；跑 staleness **sweep**（多作业共享一个 Ray 集群，正是 `run_staleness_sweep.sh` 的做法）时同时保存会把宿主 OOM。此锁把它们串行化。

### 接线差异（`train.py` / `train_async.py`）

两个入口都新增 `import os`、两个新导入、`_SLIME_DIR`、把保存块用 `with acquire_save_lock(args, rollout_id):` 包住、其后 `maybe_convert_after_save(...)`、结尾 `wait_for_previous(timeout=None)`。

---

## 3. ★★★★☆ actor 权重同步显存泄漏排查/清理（B 树独有）

**路径**：`slime_qwen3.5/slime/slime/backends/megatron_utils/actor.py`（与 A 树不同）。

B 树的 diff 新增 `_inspect_memory(tag, rank)`（psutil RSS + `torch.cuda.memory_reserved/allocated` + gc 对象数 + `megatron_to_hf._cached_tensors` 大小），并在 `train()` 各阶段与 `update_weights` 前后打 `[memleak-train]` / `[memleak]` 日志。关键：`update_weights` 之后做尽力清理：

```python
from slime.backends.megatron_utils.megatron_to_hf import _cached_tensors as _ct
if _ct: _ct.clear()
_gc.collect(); torch.cuda.empty_cache()
```

注释把它关联到观测到的多小时级 OOM：宿主 RSS / NCCL pinned buffer 跨 weight-sync 循环累积。`_cached_tensors` 是 DeepSeek 式 q_lora 配对缓存 —— 对跨 EP 的 MoE 权重 resharding，每循环的宿主缓冲大到让泄漏在长 35B-A3B run 上变致命（sweep 脚本注释佐证："host_used 在 4.5h 内涨到 2.15TB 撞 Ray 95% 阈值"）。

其余部分功能等价：routing-replay 路径（`fill_routing_replay`、`rollout_routed_experts`、逐 layer `RoutingReplay.record`）在**两树都存在且相同**。

---

## 4. ★★★☆☆ staleness / sweep 参数与运行脚本（B 树独有）

**B 树独有脚本**：`run-qwen3.5-35B-A3B_staleness_stepwise_async.sh`（真正训练入口，645 行）、`run-qwen3.5-35B-A3B_staleness_sweep.sh`、`run_staleness_sweep.sh`（并行错峰启动器）。A 树无任何 `*staleness*` 文件。共享脚本（`run-qwen3.5-35B-A3B-sft.sh`、`run-qwen3.5-27B.sh`、`scripts/models/*.sh`）逐字节相同。

这些脚本比 Harbor/terminal-RL 配置多设的东西：

**staleness 块**
```
--enable-staleness-analysis
--staleness-output-dir "${STALENESS_OUTPUT_DIR}"
--staleness-modules policy discrepancy moe
--custom-megatron-before-train-step-hook-path slime.staleness.policy_staleness_hook.before_train_step_hook
--get-mismatch-metrics
--custom-tis-function-path slime.staleness.training_inference_discrepancy.compute_staleness_discrepancy
--rollout-data-postprocess-path slime.staleness.moe_staleness_hook.postprocess_save_routing
--staleness-moe-sample-fraction 0.1
```
外加 Ray 运行时环境 `STALENESS_ANALYSIS_ENABLED=1`。

**EP / MoE 并行**（按节点数的 `case` 块驱动 `DEFAULT_EP_SIZE`）：`--tensor-model-parallel-size 4 --sequence-parallel --expert-model-parallel-size ${EP_SIZE} --expert-tensor-parallel-size 1`，EP_SIZE 随节点数 4→8→16 扩展。SGLang 侧 `--sglang-enable-return-routed-experts`（喂 Module C 与 R3 必需）、`--sglang-mem-fraction-static 0.80`、chunked-prefill 4096（注释记录了针对 248320 词表 + 35B logits all-gather 的反复 OOM 调参）。

**sweep 轴**（Harbor 配置没有）：`RL_METHOD` ∈ {GRPO, GSPO, K2PO, M2PO, OAPL}、`TIS` ∈ {False, True, Icepop, Kpop}、`ROLLOUT_ROUTING_REPLAY`（R3）、`NUM_MINI_STEPS` ∈ {1,2,4,8}（GBS = 2048/N）、`UPDATE_WEIGHTS_INTERVAL`（异步 staleness）—— 全部喂给上面的 staleness 指标。

> 对照：A 树的 MoE 跑法在 `examples/harbor_terminal_rl/run_full89_grpo_96gpu_35b_a3b_...sh` 里，是 `TP2×PP1×CP1×EP8`（256 专家 → 32 专家/GPU）、训练侧 `--moe-token-dispatcher-type flex --moe-enable-deepep`（DeepEP all-to-all）、rollout 侧 `--sglang-enable-dp-attention --sglang-ep-size 8 --sglang-enable-dp-lm-head --sglang-moe-dense-tp-size 1 --sglang-mamba-scheduler-strategy extra_buffer`（GatedDeltaNet 混合注意力）。它**能训 MoE，但不带 staleness/sweep 那套**。

---

## 5. ★☆☆☆☆ math 自定义奖励（B 树独有，次要）

**路径**：`slime_qwen3.5/slime/slime/custom_rewards/prime_math_reward.py`。A 树无 `custom_rewards/` 包。

一个兼容 `--custom-rm-path` 的异步 `compute_reward`，按文件路径加载 `utils/rewards.py`，用更强的 `prime_math` grader（多种答案抽取 + SymPy/`math_equal` 数值容差）替代默认只认 `\boxed{}` 的 deepscaler 抽取；并为 thinking 模型剥 `</think>` / `###Response` 前缀。**非 MoE-specific** —— 只是 staleness sweep 里用的更好的数学验证器（`CUSTOM_RM_PATH=...prime_math_reward.compute_reward`，实验后缀 `_primeRM`）。它与 MoE 沾边仅仅因为这些 RL 实验恰好在 MoE 模型上跑数学数据。

---

## 6. （反向）A 树独有、B 树缺失的补丁 —— 不是 MoE 适配，但需知道

以下核心文件两树不同，但方向是 **A 树多了代码、B 树是基线**，因此**不属于 B 树的 MoE 适配**；其中部分对大 MoE 反而有益（等于 B 树的短板）：

- **`backends/megatron_utils/loss.py`**：A 树多了 `compute_value_explained_variance`（PPO critic 诊断，MoE 无关）。
- **`megatron_to_hf/qwen3_5.py` + `model_provider.py`**：A 树多了 Qwen3-VL 桥的命名归一化（`module.module.language_model.*` → `module.module.*`、跳过 `vision_model.*`）与 `calculate_per_token_loss` 注入。
- **`update_weight_from_distributed.py`、`reloadable_process_group.py`、`distributed_utils.py`**：A 树给权重同步/组重建加了显式 NCCL/Gloo 超时（`SLIME_NCCL_GROUP_TIMEOUT_MINUTES` 默认 30 / `SLIME_GLOO_GROUP_TIMEOUT_MINUTES` 默认 60），避免继承 torch 的 10min / gloo 30min 默认超时把整个作业拖垮。**这对大 MoE 的跨节点权重广播其实很有帮助，而 B 树没有 —— 更像 B 树的缺口而非 A 树的适配。**
- **`utils/megatron_bridge_utils.py`**：A 树多了 Qwen3.5-MTP 禁用桥补丁 `_patch_qwen35_mtp_for_bridge`（除非 `SLIME_ENABLE_QWEN35_MTP`，否则把 `mtp_num_hidden_layers` 置 None）。
- **`ray/placement_group.py`**：A 树多了 critic checkpoint 目录隔离（`_isolate_critic_checkpoint_dirs`，PPO 用），MoE 无关。

---

## 附：确认相同（无差异）的 MoE 底座

以下 MoE 相关文件在两树**逐字节相同**，说明 MoE 的模型/权重/并行底座是共享的上游能力，不是任何一方的"专用适配"：

- 模型 shape：`scripts/models/qwen3.5-35B-A3B.sh`、`qwen3.5-27B.sh`（`--num-experts 256 --moe-router-topk 8 --moe-token-dispatcher-type alltoall --moe-shared-expert-intermediate-size 512 --moe-shared-expert-gate --moe-grouped-gemm --moe-router-dtype fp32 --moe-permute-fusion --moe-aux-loss-coeff 0`）
- 模型 spec：`slime_plugins/models/qwen3_5.py`（`get_qwen3_5_spec`）
- MoE 权重桥：`megatron_to_hf/qwen3moe.py`、`glm4moe.py`（`qwen3_5.py` 仅在 A 树多了 VLM 归一化，见 §6）
- EP 参数：`--expert-model-parallel-size` / `--expert-tensor-parallel-size` 均为 Megatron 透传，两树 `backends/megatron_utils/arguments.py` 相同
- R3 routing-replay：`slime/utils/routing_replay.py`、SGLang 侧 `--sglang-enable-return-routed-experts` → `return_routed_experts` → `sample.rollout_routed_experts`，两树相同

---

## 如果要把 B 树的 MoE 能力搬到 terminal-RL

按优先级（高→低）：

1. **移植 `slime/staleness/` 整目录** + `arguments.py` 的 `add_staleness_analysis_arguments`，并在 run 脚本里挂三个钩子。这是唯一真正 MoE-specific、且对 off-policy MoE RL 诊断价值最高的部分。
2. **移植 `ckpt_async_convert.py` + `save_lock.py`** 并在 `train.py`/`train_async.py` 接线 —— 若要在共享集群上跑大 MoE 或多作业，几乎是必需的。
3. **移植 `actor.py` 的显存泄漏排查/`_cached_tensors.clear()`** —— 长时 MoE run 的稳定性关键。
4. （可选）`prime_math_reward.py` —— 仅数学任务需要。

反向地，terminal-RL 的 NCCL/Gloo 超时、MTP/VLM 桥补丁在大 MoE 权重同步/转换上是有价值的，B 树若要长期跑也应回补。

---

## 7. 模拟运行：不移植任何 MoE 适配代码，直接用 slime-terminal-RL 训 MoE 会怎样

> 场景：拿本仓库现成的 `examples/harbor_terminal_rl/run_full89_grpo_96gpu_35b_a3b_...sh`（训 Qwen3.5-35B-A3B），**一行 MoE 适配代码都不补**，走一遍从启动到长跑的流程，逐阶段推演会遇到什么。
>
> 先给结论分层：
> - **✅ 能跑起来、能出模型**：MoE 底座（模型 shape / spec / 权重桥 / EP / DeepEP / R3）两树逐字节相同，训练在功能上**不会因为缺 MoE 适配而无法启动或立即报错**。
> - **⚠️ 长跑会死**：最可能在**数小时后因宿主内存（host RAM）被 Ray 判超阈值杀 worker** 而中断 —— 这是不补适配代码时最现实、最致命的问题。
> - **🔇 看不见问题**：即使侥幸跑完，也**完全没有 MoE 路由 staleness / 训推差异的可观测性**，off-policy 下专家路由漂移导致的训练退化会“悄悄发生、无法诊断”。
> - **🐢 运维退化**：checkpoint 导出阻塞、多作业共享集群时保存互相 OOM。

按“会不会挡住你 / 多快出现 / 后果多重”排序：

### 问题 1（致命，数小时后触发）：宿主内存单调上涨 → Ray 95% 阈值杀 actor

**根因不是 dense/MoE 的代码分支差异，而是缺了 §3 的 `actor.py` 显存治理 + §2 的 save_lock。** B 树的 `MEMORY_LEAK_TRAIN_STEP.md` 有实测：8 个 Megatron train actor 每个 RSS 从 ~145GB 涨到 ~184–185GB，**节奏约 +5GB/actor/cycle**，跑 5–15h 后节点撞到 `2151.99 / 2265.25 GB (0.95)`，Ray 抛 `OutOfMemoryError` 杀 worker，整个作业挂。文档明确归因为多因素叠加，其中 **MoE 相关项占大头**：

1. Megatron `--optimizer-cpu-offload + --use-precision-aware-optimizer` 的 CPU staging buffer 每步累积；
2. **MoE token_dispatcher 的 expert all-gather buffer**（256 experts × 40 layers，dynamic batch 下 seqlen 浮动，buffer 不复用）—— 日志里 `EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP` NCCL 超时就是这个热点；
3. PyTorch caching allocator 的 pinned host buffer 不归还。

terminal-RL 的 `actor.py` **没有** `_inspect_memory` 埋点，也**没有** `update_weights` 后的 `gc.collect() + torch.cuda.empty_cache()` 清理。实测表明这套清理能把 `update_weights` 那段的 `delta_rss` 压到 ≈0（有效但只治 update_weights 那部分）。**不补 → 你连“内存涨在哪个子阶段”都看不到，只能看着作业在半夜某个 cycle 猝死。**

> 注意一个**反直觉的点**：B 树 `actor.py` 里那句 `_cached_tensors.clear()` 对 Qwen3.5 **其实是空操作** —— 该缓存只在 `args.q_lora_rank is not None`（DeepSeek-MLA 路径）才填充，而 Qwen3.5-35B-A3B 用的是 gated attention + GQA，**根本不走这条路**（`scripts/models/qwen3.5-35B-A3B.sh` 无 `q-lora-rank`）。所以“移植 `actor.py`”真正有价值的是 **`gc.collect()+empty_cache()` 那半句 + `_inspect_memory` 埋点**，而不是 `_cached_tensors.clear()`。写文档/移植时别把重点放错。
>
> 也就是说：这个致命问题**并非“dense 适配天生训不了 MoE”**，而是 slime-terminal-RL 从没在“256 专家 + CPU-offload 优化器 + dynamic batch”这种 MoE 高内存组合上长跑过、因而缺了对应的内存治理与诊断。

**不补的现实后果**：短跑 smoke（几个 cycle）能过；**正式 2-epoch / 数十 rollout 的长跑大概率跑不完**。缓解手段（不改代码时的临时招）：调小 `MAX_TOKENS_PER_GPU`（如 16384→更低，实测能延长撞墙时间但治标不治本）、重启 Ray 时设 `RAY_memory_monitor_refresh_ms=0` / 提高 `RAY_memory_usage_threshold`（但这要重启 raylet daemon 才生效，且有被 Linux OOM killer 杀整节点的风险）。

### 问题 2（严重，静默）：MoE 路由 staleness 完全不可观测 → 训练退化无法诊断

不补 §1 的 `slime/staleness/`，你会**彻底失去以下信号**：

- **专家重叠 / Jaccard**：SGLang 在 rollout 时选的专家 vs Megatron 训练时选的专家，逐 token/逐 layer 的重合度；
- **Router-Shift Ratio γ / clip_frac**：router 分布相对旧策略的漂移；
- **Routing Agreement Rate**：R3 到底在多大程度上纠偏；
- **训推 IS 修正因子 / 极端 token 比例 / KL(train‖rollout)**（Module B）。

**为什么这对 MoE 尤其要命**：256 专家 / top-8 的 off-policy RL 里，router 本身是移动靶。若训练时路由与生成轨迹时的路由发散，GRPO/GSPO 的重要性采样假设在**专家层级**就破了 —— 表现为 reward 停滞甚至倒退、熵异常，但你**无法把它归因到“路由漂移”**，因为没有任何指标。terminal-RL 现有的 Harbor 指标（reward_mean、response_len、truncated_ratio…）全是**任务级/token 级**的，**没有一个是专家级的**。

**能否用 R3 兜底？** terminal-RL 的 MoE 脚本可以开 R3（`--use-rollout-routing-replay` + `--sglang-enable-return-routed-experts`，两树 `routing_replay.py` 相同），R3 会在训练时**回放**推理侧的专家选择来减小漂移。但 R3 是“缓解手段”，**不是“度量手段”**：开了 R3 你也不知道它纠偏了多少、还剩多少残差 —— 那正是 RAR / Jaccard 指标要回答的问题。**不补 staleness = 开着 R3 也是盲跑。**

### 问题 3（中等，运维退化）：checkpoint 导出阻塞 + 多作业保存互相 OOM

不补 §2 的 `ckpt_async_convert.py` + `save_lock.py`（且 terminal-RL 的 `train.py`/`train_async.py` **确实没有**这两处接线）：

- **HF 导出不再自动化/异步**：35B-A3B / 256 专家的 torch_dist checkpoint 巨大，转 HF safetensors 很慢。terminal-RL 的做法是训练存 torch_dist、**另跑离线脚本**（`convert_qwen35_27b_torch_dist.sh` 之类）转 HF。对纯训练不致命，但你失去了“存完自动后台转 HF 供 eval”的便利；若想在训练中途用 SGLang 对 HF 权重做 eval，得手工转、时序上容易踩空。
- **多作业共享集群会互相 OOM**：`save_lock.py` 的 docstring 点明——每个 train actor 保存时飙 ~177GB CPU 内存，两个同调度的 8-actor 作业就能超 2.2TB 节点内存被 Ray 杀。terminal-RL 无此锁，所以**如果在同一 Ray 集群上跑多个 MoE 作业（或 sweep）**，保存瞬间叠加就可能整节点 OOM。单作业跑通常不触发，但这恰好和**问题 1** 叠加：保存时的内存尖峰会提前把已经高位运行的宿主内存推过 95% 阈值。

### 问题 4（低，仅特定场景）：缺 prime_math 奖励 / staleness sweep 编排

- §5 的 `custom_rewards/prime_math_reward.py`：只有跑**数学任务**才需要（Harbor 是终端 agent 任务，用自己的 verifier reward，**不受影响**）。
- §4 的 sweep 脚本与 `--enable-staleness-analysis` 等参数：不做 RL_METHOD/TIS/R3/mini-steps 的系统性扫描就用不上。**对“只想训一个 MoE 模型”无影响**，只对“研究 MoE staleness”有影响。

### 一个容易被忽略的正向点：A 树的补丁反而对 MoE 有利

见 §6 —— terminal-RL **独有**而 qwen3.5 **缺失**的 NCCL/Gloo 显式超时（`SLIME_NCCL_GROUP_TIMEOUT_MINUTES` / `SLIME_GLOO_GROUP_TIMEOUT_MINUTES`）、MTP 桥禁用补丁，在大 MoE 的跨节点权重广播 / `EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP` all-gather 偶发 stall 时**能防止 10min/30min 默认超时把作业拖垮**。也就是说，在“权重同步健壮性”这一维度上，**terminal-RL 现状对 MoE 反而更稳**（这也是 B 树 `MEMORY_LEAK` 文档里出现 NCCL 超时告警的部分原因）。

### 模拟运行总结（时间线视角）

| 阶段 | 现象 | 是否被 MoE 适配缺失影响 |
|---|---|---|
| 启动 / 模型构建 | ✅ 正常（spec/桥/EP 相同） | 否 |
| 首个 rollout + train step | ✅ 正常出 loss/reward | 否 |
| Smoke（几个 cycle） | ✅ 大概率通过 | 否 |
| 长跑 5–15h | ⚠️ **宿主内存撞 Ray 95% 被杀，作业中断** | **是（问题 1）** |
| 训练全程 | 🔇 **无任何专家路由漂移可观测性，退化无法诊断** | **是（问题 2）** |
| 保存 checkpoint | 🐢 HF 导出需手工离线转；多作业同存可能 OOM | 是（问题 3） |
| 权重同步健壮性 | ✅ 反而比 qwen3.5 更稳（A 树独有超时补丁） | 否（正向） |

**一句话**：不补 MoE 适配，slime-terminal-RL **训得动 MoE、但训不长、也训得盲**。要真正把 Qwen3.5-35B-A3B 这类 MoE 跑到收敛并能诊断，按优先级至少补 **问题 1（actor 内存治理）→ 问题 2（staleness 可观测性）→ 问题 3（save_lock + 异步转换）**；`prime_math` 与 sweep 编排可按需。

---

*本文基于 A 树 `harbor/v030-port` 与 B 树 `async_qwen3.5_260624` 分支的逐文件 diff 整理。差异方向约定：源码 diff 中 `<` 为 terminal-RL、`>` 为 qwen3.5。第 7 节的内存数字与根因引自 B 树 `MEMORY_LEAK_TRAIN_STEP.md` 实测。*
