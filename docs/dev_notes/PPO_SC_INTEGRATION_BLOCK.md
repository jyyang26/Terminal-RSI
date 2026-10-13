# PPO + Score Centering 集成 —— 阻碍与异常记录

**文档用途**：按任务红线要求，记录 PPO_SC 集成各阶段遇到的技术阻碍、设计冲突、Bug 与环境异常。
**关联**：`PPO_SC.md`（理论）、`PPO_SC_INTEGRATION_PLAN.md`（实施计划）
**最后更新**：2026-09-26（复核 PR#2405 阶段一报告；推翻其 C-1；新增 B-17）

---

## 目录

| # | 标题 | 阶段 | 严重度 | 状态 |
|---|---|---|---|---|
| B-1 | `PPO_SC.md` 的 $\bar w = \text{clip}(r)$ 与代码实际行为不符 | Stage 2 | **设计冲突（高）** | 已修正并锁死 |
| B-2 | 基线 PPO 分母是 $\pi_{old}$ 而非采样器 $q$ | Stage 2 | **设计冲突（高）** | 已参数化，待实测决策 |
| B-3 | `.float()` 静默降精度（自有代码缺陷） | Stage 2 | Bug（中） | 已修复 |
| B-4 | Head 内重复 token id 导致对账误报 | Stage 3 | Bug（中） | 已修复 |
| B-5 | 指标归约语义：per-sample 标量被求和 | Stage 3 | Bug（中） | 已修复 |
| B-6 | YAML 合并晚于校验，SC 可绕过校验 | Stage 3 | Bug（中） | 已修复 |
| B-7 | 测试套件 `sys.modules` 桩污染 | Stage 2/3 | 环境异常（既有） | 已缓解 |
| B-8 | `gradcheck` 无法验证 stop-gradient 边界 | Stage 2 | 方法学限制 | 已规避并记录 |
| B-9 | SGLang patch 的"语义静默变空"风险 | Stage 4 | ~~阻断（高）~~ → **Low** | **已推翻**：改动在 `if top_logprobs_num > 0` 守卫内 |
| B-10 | Harbor rollout 不请求也不透传 top-k head | Stage 4 | ~~阻断（高）~~ → **设计完备** | 数据契约经复核可用（第三批） |
| **B-11** | **`sglang-top_p.patch` 文件已截断，且 11/18 文件打不上** | **Stage 4（待）** | **阻断（高）** | **未解决，需决策** |
| B-12 | SGLang 以 in-process `multiprocessing` 启动，切换只能 job 级 | Stage 4 | 设计约束（中） | 已纳入设计 |
| B-13 | 破坏性切片点是 3 构建器 / 11 处，非先前记录的 5 处 | Stage 4 | 记录更正（中） | 已更正 |
| B-14 | `Sample.rollout_topk_*` 字段尚不存在 | Stage 4 | 低 | 已确认可自动序列化 |
| **B-15** | **sglang checkout 有 1480 行未提交修改（含 delta 权重同步）** | **Stage 4（待）** | **阻断（高）** | **未解决，需先备份** |
| B-16 | `PYTHONPATH` 整包影子切换可行（实测），禁止部分影子 | Stage 4 | 已验证 | 已纳入设计 |
| **B-17** | **`sc_terms.py` 未处理 `--allgather-cp`（我自己的 Stage 3 缺陷）** | Stage 3 | Bug（中） | 已定位，修复方案已定 |

> B-1 ~ B-8 已解决或规避。**B-9 经实测被推翻**（Low），并同时**推翻了阶段一报告的 C-1 结论**（第三批）。
> **B-10 已降级**为"设计完备、待实施"——阶段一报告 §4.3 的 8 项数据契约经逐行复核成立，可直接采用。
> **当前唯一的未解决阻断是 B-11（patch 文件损坏）与 B-15（环境 1480 行未提交修改）**，二者均需你决策。
> B-17 是复核中发现的我方缺陷（不影响基线）。解决方案见 `PPO_SC_B9_B10_RESOLUTION_DESIGN.md`。

---

## B-1 `PPO_SC.md` 的有效权重公式与代码实际行为不符

**阶段 / 位置**：Stage 2 / `slime/utils/ppo_utils.py:125` `compute_policy_loss`

**现象**
`PPO_SC.md` §1 声明 $\bar w(y_t)=\text{clip}(r_t,1-\epsilon,1+\epsilon)$。对本仓库实际使用的 `max(-rA, -\text{clip}(r)A)` 算子做 autograd 反解，结果不符（$\epsilon=0.2$）：

| $r$ | 0.5 | 0.79 | 0.81 | 1.0 | 1.19 | 1.21 | 3.0 |
|---|---|---|---|---|---|---|---|
| autograd，$A>0$ | 0.5 | 0.79 | 0.81 | 1.0 | 1.19 | **0** | **0** |
| autograd，$A<0$ | **0** | **0** | 0.81 | 1.0 | 1.19 | 1.21 | **3.0** |
| 文档 $\text{clip}(r)$ | 0.8 | 0.8 | 0.81 | 1.0 | 1.19 | 1.2 | 1.2 |

**根本原因**
min-clip 算子在被截断侧进入**平坦区**，该侧梯度恒为 0（而非"梯度乘以 clip 后的比率"）；未截断侧传递的是**原始 $r$**，不是 clip 后的值。且平坦区落在哪一侧由 $\text{sign}(A)$ 决定。正确闭式：

$$\bar w(v)=r_v\cdot\mathbf 1\big[\neg((A>0\wedge r_v>1+\epsilon_h)\vee(A<0\wedge r_v<1-\epsilon_\ell))\big]$$

$\bar w_{\text{tail}}$ 同理：文档给的 $\text{clip}(1/\rho)$ 在被截断侧应为 **0**（例：$\rho=2.5,A<0$ 时真值 0，文档给 0.8）。

**风险评估**
若按文档实现：减去的是**优化器从未产生过的**那个算子的期望，$\mathbb E_q[g_{\text{PPO+SC}}]=0$ 的零漂移恒等式**不成立**。$r=0.5,A>0$ 处会用 0.8 而真值 0.5，漂移项系统性偏大 —— 且不会报错，只会让训练偏向一个错误目标。

**修复**
`slime/utils/score_centering.py::ppo_effective_weight` 实现上述闭式。单测 `test_ppo_sc_math.py::test_t5_effective_weight_matches_autograd` 与 autograd 逐点 `allclose`；`test_t5_effective_weight_is_not_clip_of_ratio` **显式断言它 ≠ `clip(r)`**，防止后来者"按文档改回去"。

---

## B-2 基线 PPO 的分母是 $\pi_{\theta_{old}}$，不是采样器 $q$

**阶段 / 位置**：Stage 2 / `examples/harbor_terminal_rl/run_tmax15k_..._strongest.sh:510-511`

**现象**
`PPO_SC.md` §1 假设 "$\pi_{\theta_{\text{old}}}$ 对应采样推理引擎分布 $q$"。实测基线：`KEEP_OLD_ACTOR=1`、`USE_ROLLOUT_LOGPROBS=0`，因此
$$r_t=\frac{p_\theta(y_t)}{\pi_{\theta_{old}}(y_t)},\qquad \pi_{\theta_{old}}=\text{old-actor 权重快照的重算前向}\ \ne\ q_{\text{SGLang}}$$

**根本原因**
链路里存在**两个独立失配**：$p_\theta$ vs $\pi_{old}$（策略滞后，PPO clip 管这个）与 $\pi_{old}$ vs $q$（TIM：SGLang 量化/数值/陈旧，SC 管这个）。中心化的期望必须在 $q$ 下取，而权重 $\bar w$ 由 $p/\pi_{old}$ 决定 —— 两者不是同一个分布。严格实现需要 $\pi_{old}(v)$ 在**整个头部集合**上的值，而 old-actor 前向目前只产出采样 token 的 logprob。

**风险评估**
选 `sampler`（令 $\pi_{old}:=q$）时，消除的是"以 $q$ 为分母的 PPO"的漂移；与真实 PPO 的残差 $\propto|\log\pi_{old}-\log q|$。该偏差**已有现成指标可量化**：`train_rollout_logprob_abs_diff`（`train_loss.py:207`）正是 $|\log\pi_{old}-\log q|$ 的掩码均值。

**处理**
参数化为 `--sc-weight-denominator {sampler, old_actor}`，默认 `sampler`（低成本、无额外前向）。`old_actor` 分支已在 `sc_terms.py` 中实现读取 `batch["sc_old_head_log_probs"]`，但**该字段的生产者尚未接入**（需在 `actor.py:490` 的 old-actor 前向里额外 gather 一次 head）。

**建议**：Stage 4.4 先读 `train_rollout_logprob_abs_diff`。< 0.05 → 用 `sampler`；> 0.2 → 上 `old_actor` 并强制 `--sc-top-k 32`（$k=128$ 时该分支额外占用约 6.9 GiB GPU，不可接受）。

---

## B-3 `.float()` 静默降精度（自有代码缺陷，被单测抓出）

**阶段 / 位置**：Stage 2 / `slime/utils/score_centering.py`

**现象**
初版用 `train_head_logp.float()` 做 FP32 提升。生产路径输入本就是 fp32，无影响；但 float64 参考测试中它**静默截断**，导致 `test_t4_zero_anchor_holds_with_a_mismatched_sampler_while_unclipped`、`test_t8_gradient_equals_the_detached_coefficients` 等 3 个测试给出假阳性（`tail_eps=1e-15` 在 fp32 下 `1-1e-15 == 1.0`，clamp 失效）。

**根本原因**
`.float()` 是**无条件转 fp32**，对 fp64 是降精度。而尾质量 `1 - sum(p)` 是一次**抵消运算**，在它之前丢弃尾数位正是最糟的位置。

**修复**
`_at_least_fp32()`：只把 fp16/bf16 提升到 fp32，fp32/fp64 原样通过。

---

## B-4 Head 内重复 token id 导致对账误报

**阶段 / 位置**：Stage 3 / `slime/backends/megatron_utils/sc_terms.py::_check_head_alignment`

**现象**
初版用 `(sampler_head * in_head).sum(-1)` 取出"采样 token 在 head 中的 logprob"。当同一 id 在 head 里出现两次时，masked sum 会把**两个 logprob 相加**，产生虚假偏差并触发 misalignment 异常。

**根本原因**
masked-sum 隐含"每行至多一个匹配列"的假设。head 理应 id 互异，但这是上游契约而非本地保证；一个良性的上游小瑕疵会被放大成硬失败。

**修复**
改为 `in_head.float().argmax(-1)` 取**第一个**匹配列后 `gather`。

---

## B-5 指标归约语义：per-sample 标量被求和

**阶段 / 位置**：Stage 3 / `sc_terms.py`

**现象**
`sc_head_hit_rate` 报出 **2.0** —— 对一个"率"不可能。

**根本原因**
`get_sum_of_sample_mean`（`cp_utils.py:81`）返回的是**per-sample 均值之和**，下游 `train_one_step` 再除以 `step_global_batch_size`。因此一个"每样本一个标量"的指标会报成 `c × num_samples`。原实现用 `.reshape(1).expand(...)` 把标量广播成 per-token，语义上仍是 per-sample 标量。

**修复**
改为发射**逐 token 指示量**（`hit.float()`），与既有 `pg_clipfrac` 完全同构。另外，因为对账只在部分样本上运行，其张量不能与 per-token 张量直接 `torch.cat`（reducer 会按 `response_lengths` 错切），故先池化为标量再广播到总行数。

---

## B-6 YAML 合并晚于校验，SC 可绕过校验

**阶段 / 位置**：Stage 3 / `slime/utils/arguments.py`

**现象**
`--custom-config-path` 的 YAML 合并位于 `slime_validate_args` **内部**（`:2004` 附近），而我最初把 SC 校验放在该函数**入口**（`:1754`）。于是一个只在 YAML 里写 `enable_score_centering: true` 的启动会**完全跳过所有校验**。

**风险评估**
任务明确要求支持 YAML 传参。若不修，YAML 路径可以带着 `advantage_estimator: gspo`、`use_tis: true`、`rollout_top_p: 0.95` 等**数学上非法**的组合进入训练，且不报错。

**修复**
把校验移到 YAML 合并**之后**。新增测试 `test_yaml_can_enable_score_centering` 用 `inspect.getsource` 断言 `validate_score_centering_args` 在源码中出现在 `custom_config_path` **之后**，把这个顺序固化为回归测试。

---

## B-7 测试套件 `sys.modules` 桩污染（既有问题）

**阶段 / 位置**：Stage 2/3 / `tests/`

**现象（两类）**
1. 多个 Harbor 测试在 import 时注入桩 `torch`（如 `test_harbor_atif_adapter.py:7` 的 `if "torch" not in sys.modules: sys.modules["torch"] = torch_stub`）。一旦注入，后续任何 `import torch.nn` 或 `import sglang_router`（`arguments.py` 模块级就有）都会失败。
2. `test_explained_variance.py:41` 用桩**替换 `megatron.core.mpu`** 且不恢复，此后 `mpu.initialize_model_parallel` / `get_context_parallel_group` 在整个 session 内消失。

**根本原因**
`if "torch" not in sys.modules` 这类 guard 是**顺序依赖**的：谁先 import 谁赢。`sys.modules` 替换没有 teardown。

**证明其为既有问题（非本次引入）**
把本次新增的 5 个文件全部移出、`arguments.py` 还原为 pristine 后跑完整套件，**同样的 4 个 `test_qwen3_linear_attention_cu_seqlens` 失败照旧出现**（235 passed / 4 failed / 19 errors）。

**处理**
- 新增 `tests/conftest.py`，在 collection 前 import 真 `torch`，让那些 guard 变成 no-op。**顺带修好了上述 4 个既有失败**（全套件 235 → 299 passed）。真正无 torch 的环境里该 import 软失败，桩逻辑照旧。
- `test_ppo_sc_args.py` 通过**子进程**解析参数，彻底免疫。
- `test_ppo_sc_harbor_integration.py` 检测到 mpu 已被替换时 `pytest.skip` 并给出单文件运行提示 —— 不去触碰其他测试的全局状态。

**遗留风险**：mpu 被桩替换时，21 个集成测试在全套件运行中会 **skip**（单文件运行全部执行）。**CI 需要把该文件单独成 job**，否则这些断言在合流时不会真正生效。这是一个需要你确认的 CI 配置项。

---

## B-8 `gradcheck` 无法验证 stop-gradient 边界

**阶段 / 位置**：Stage 2 / `tests/test_ppo_sc_math.py`

**现象**
对 `score_centering_correction` 直接 `gradcheck` 报 Jacobian mismatch。

**根本原因**
`coef` 是 stop-gradient 且 `ppo_effective_weight` 含硬阈值 → `coef` 对 $\log p$ **分段常数**。有限差分探针会跨越阈值看到 `coef` 跳变，而解析梯度（**正确地**）视其为常数。这不是实现缺陷，是 gradcheck 方法学本身无法覆盖 stop-gradient 不连续点。

**处理**
两个版本：(a) 宽 trust region（$\epsilon=10^6$，不可能截断）；(b) 生产 $\epsilon$ 但构造严格内点（$q$ 与 $p$ 仅差每行常数，ratio 恒为 1）。另有 `test_t8_gradient_equals_the_detached_coefficients` 用**独立重算的 coef** 与解析梯度 `allclose` 对账，这比 gradcheck 更强。

---

## B-9 🔴 SGLang 未应用 `sglang-top_p.patch`，SC 无法产出数据

**阶段 / 位置**：Stage 4（未开始）/ 运行时 SGLang + `docker/patch/latest/sglang-top_p.patch`

**现象（实测）**
```
运行时 sglang 0.5.12.post1 @ /sgl-workspace/sglang/python/sglang
  output_topk_token_ids            : False   ← top_p.patch 未应用
  top_p_token_ids                  : False   ← 同上
  routed_experts                   : True    ← sglang.patch 已应用（R3 依赖，正常）
```

**根本原因**
SC 需要 `meta_info["output_topk_{token_ids,log_probs,shape}"]`，这三个字段只来自 `sglang-top_p.patch` 的新 hunk。该 patch 存在于 `docker/patch/latest/` 且被 `docker/Dockerfile:109` 的 patch 循环引用，但本集群的 sglang 不是从那一层装的。

**对现有训练逻辑的风险评估**
- **当前为零风险**，而且是**双重安全**：(a) patch 未应用；(b) Harbor 的 `llm.py:119-124` 请求体不带 `top_logprobs_num`，`:150` 只读 `output_token_logprobs`，从不读 `output_top_logprobs`。
- **但重建镜像有真实风险**：该 patch 把 `output_top_logprobs` 的填充移入 `if return_text_in_logprobs:` 分支 —— **不带该标志的请求，该字段会静默变空**。这是一个无条件的语义替换，不受任何 slime 侧 `args` 控制。

**可选修复方案**

| 档 | 做法 | 增量开销 | 风险 | 用途 |
|---|---|---|---|---|
| **档 0（推荐先做）** | 不改 SGLang。trainer 用自己 logits 的 top-$k$ 当合成 head（$q:=p$） | 0 网络 | **零** | 端到端验证整条管道（搬运/对齐/CP/TP/内存/autograd），且因 $q=p$ 时修正项**恒为 0**，loss 与基线逐 bit 相同。已有单测 `test_sharp_distribution_drives_the_correction_toward_zero` 覆盖该性质 |
| 档 1 | 请求 `top_logprobs_num=k`，读**未打 patch 的** `output_top_logprobs`（detokenized JSON list） | JSON 体积 ×5~10 | 中 | 小批量真实 head 验证 |
| 档 2 | 重建镜像应用 patch，读 base64 二进制 | base64 ×4/3 | **高** | 正式训练 |

**档 2 的硬前置条件**：
1. `grep -rn 'output_top_logprobs' slime/ slime_plugins/ harbor/` 必须为空；
2. 新增契约测试锁死该字段语义；
3. 单节点 canary：同一 prompt 在打/未打 patch 下 `output_token_logprobs` 逐 bit 一致（patch 只应**加**字段，不应改采样 logprob）。

**状态：未解决。需要你决策走哪一档。**

---

## B-10 🔴 Harbor rollout 不请求也不透传 top-k head

**阶段 / 位置**：Stage 4（未开始）/ `slime_plugins/harbor/llm.py`、`slime_plugins/harbor/atif_adapter.py`

**现象**
即使 SGLang 侧产出了 head，数据也到不了 trainer：
- `llm.py:127-128` 自建 `httpx.AsyncClient().post()`，**不走** `slime/utils/http_utils.post`，请求体不含 `top_logprobs_num`；
- `llm.py:150` 只解析 `output_token_logprobs`；
- `atif_adapter.py` 装配 `Sample` 时没有 head 字段，`Sample` 本身也还没有 `rollout_topk_*` 两个字段（Stage 3 未引入，属计划的 Stage 3 数据通道部分）。

**根本原因**
本次 Stage 3 的范围是**损失侧双挂载**，数据通道（`Sample` 字段、`ray/rollout.py` 打包、`actor.py` CP 切分、`llm.py` 请求/解析、`atif_adapter.py` 搬运）属于计划书的 Stage 3 后半段，尚未实施。当前 `get_score_centering_terms` 在缺少 head 时会**显式抛错**（`test_missing_heads_raise_a_clear_error` 已覆盖），不会静默降级。

**最高风险点（需特别注意）**
`harbor_sample_mode: trajectory_retokenized` 对 token 做三类破坏性操作：`prompt_suffix_dropped_tokens`、`response_prefix_dropped_tokens`、`response_suffix_dropped_tokens`，外加 chunk `split` 与 retokenize 追溯 mask。共有 **5 处** `rollout_log_probs` 写入点（`atif_adapter.py:426/429/439/444(865)/832(1328)`）必须配对搬运 head。

**漏任一处 = head 与 token 静默错位**：shape 仍匹配、$\rho$ 仍合理、loss 仍下降，只是在优化错误目标。

**已有的缓解**：`sc_terms.py::_check_head_alignment` 的**双通道对账** —— 采样 token 若命中 head，则 head 中对应 logprob 必须与独立传输的 `rollout_log_probs` 一致（容差 1e-2）。这是唯一能真正抓到错位的手段，已有两个测试覆盖（值错位 → 抛错；ids 也错位 → `sc_head_hit_rate` 坍塌）。

**状态：未解决。** 这是一个独立的、工作量不小的数据通道适配项（计划书估计 `atif_adapter.py` +90 行、`llm.py` +35 行）。

---

## 附：本次 Stage 3 的零回归证明

| 检查项 | 结果 |
|---|---|
| `compute_policy_loss` 字节码 MD5 | `e608eac4...` **与 Stage 0 完全一致**（PPO clip 算子一行未动） |
| `slime/utils/ppo_utils.py` | +140 / **−0** |
| `slime/utils/arguments.py` | +92 / **−0** |
| `slime/backends/megatron_utils/loss.py` | +12 / **−0** |
| `slime/backends/megatron_utils/actor.py` | +42 / **−0** |
| `slime_plugins/harbor/train_loss.py` | +22 / **−0** |
| 基线 args namespace | 恰好 6 个新 key，0 删除，**0 值变化** |
| 开关关闭时 loss / 全部梯度 | 与"无 SC 属性的 args"**逐 bit 相同**（Harbor fork 与 core 双路径各验一次） |
| 开关关闭且 batch 带 head | 仍与基线**逐 bit 相同**（不会静默启用） |
| 新增测试 | 87 passed |
| 全仓库 unit 套件 | 4 failed（**既有**，已用移除本次改动的方式证明）/ 299 passed |

---

**若后续阶段再出现阻碍，追加到本文档并暂停等待确认。**


---

# 第二批阻碍（B-9 / B-10 深入排查，2026-09-26）

以下为按指令对 B-9、B-10 做深入排查后的**证据、修正与新发现**。B-9 的严重度依据实测证据**下调**；同时新增 4 项阻碍 B-11 ~ B-14，其中 **B-11 是新的高危阻断**。

---

## B-9 修正：严重度 High → **Low**（patch 语义风险不成立）

**原判断**：patch 把 `output_top_logprobs` 移入 `if return_text_in_logprobs:` 分支 → 不带该参数时字段静默为空 → 属无条件语义替换。

**实测证据（推翻原判断）**

被修改的整个代码块位于一个**外层守卫**内。运行时 `tokenizer_manager.py:1939-1959`：

```python
# 2. Handle top logprobs
if top_logprobs_num > 0:                       # ← 外层守卫（patch 未改动它）
    ...
    meta_info["input_top_logprobs"]  = state.input_top_logprobs
    meta_info["output_top_logprobs"] = state.output_top_logprobs
```

patch 的 diff 上下文（`docker/patch/latest/sglang-top_p.patch:900-950`）显示它只重写了该守卫**内部**的赋值，守卫本身原样保留。

而 `io_struct.py` 的归一化逻辑：
```
top_logprobs_num: Optional[Union[List[int], int]] = None
if self.top_logprobs_num is None:
    self.top_logprobs_num = 0
```

**结论（逐条）**
1. 不传 `top_logprobs_num` → 归一化为 `0` → `if top_logprobs_num > 0` 为假 → **被修改的代码一行都不执行**，`output_top_logprobs` 本来也不会被写入（原版同样不写）。
2. 因此**不存在"静默变空"**：原版在 `top_logprobs_num == 0` 时也不产出该字段。
3. Harbor 的 `llm.py:119-124` 不传该参数，且 `:150` 只读 `output_token_logprobs` —— **双重无关**。

**修正后的风险评估**：**Low**。语义变更仅对"显式请求 top-logprobs **且** `return_text_in_logprobs=False`"的请求生效，而这正是 SC 自己要发的请求。基线永不进入。

**`llm.py` 请求端需要传的确切字段**（回答问题 1）

| 字段 | SC 开启 | SC 关闭 | 说明 |
|---|---|---|---|
| `return_logprob` | `True` | `True`（**已有**，`llm.py:122`） | 不变 |
| `top_logprobs_num` | `k` | **不传** | 唯一新增；不传 → 归一化 0 → 新分支不可达 |
| `return_text_in_logprobs` | **不传**（默认 `False`） | 不传 | 必须保持 `False` 才走二进制分支；默认即是 |

**"仅在开启 SC 时才生效"的安全传参方式**：即 §B-10 的 `if self._sc_top_k:` 条件注入。因为新分支的**唯一入口**是 `top_logprobs_num > 0`，而该值只由我们的请求体决定，所以**"不传参数"就是完备的隔离**，不需要任何运行时切换。

---

## B-11 🔴 **新增高危**：`sglang-top_p.patch` 文件本身已损坏，且与当前 checkout 严重偏离

**阶段 / 位置**：Stage 4 前置排查 / `docker/patch/latest/sglang-top_p.patch`

### 现象一：patch 文件被截断（硬错误）

```
$ cd /sgl-workspace/sglang && git apply --check docker/patch/latest/sglang-top_p.patch
error: corrupt patch at line 1176
```

文件共 1175 行。末尾 hunk 头 `@@ -463,6 +463,7 @@`（第 1169 行）声明需要 7 行 body（6 context + 1 added），但实际只有 **6 行**，且**文件末尾缺失换行符**（`od -c` 显示最后一字节为空格而非 `\n`）。

补一个换行后 corrupt 错误消失，说明**截断是纯粹的文件完整性问题**，不是格式问题。

### 现象二：修复换行后，18 个文件里 **11 个** 打不上

```
ok    logits_processor.py / detokenizer_manager.py / multi_tokenizer_mixin.py
ok    scheduler_components/{batch_result_processor,logprob_result_processor,output_streamer}.py
FAIL  disaggregation/decode.py          FAIL  disaggregation/utils.py
FAIL  layers/sampler.py                 FAIL  layers/utils/logprob.py
FAIL  managers/io_struct.py             FAIL  managers/schedule_batch.py
FAIL  managers/tokenizer_manager.py  ← SC 关键文件
FAIL  managers/utils.py                 FAIL  sampling/sampling_batch_info.py
FAIL  speculative/{eagle_worker_v2,multi_layer_eagle_worker_v2,ngram_worker}.py
```

**更正一处我先前的口误**：我曾说 "`tokenizer_manager.py` 在隔离下可干净应用" —— 那是 `--include` 过滤写法导致的空输出误读。正确结论是**它也失败**：

```
$ git apply --check --include='python/sglang/srt/managers/tokenizer_manager.py' ...
error: patch failed: python/sglang/srt/managers/tokenizer_manager.py:36
```

用 `patch(1)` 带 fuzz 试探，该文件 **3 个 hunk 中 1 个失败**（Hunk #1 at 670），另 2 个以 fuzz 2~3 勉强命中。

### 根本原因

1. **截断**：patch 文件在生成/拷贝过程中丢失尾部（很可能是重定向被中断，或跨文件系统复制截断）。它是 untracked 文件（`git status` 显示 `??`），没有 git 历史可回溯。
2. **偏离**：`/sgl-workspace/sglang` 是**可编辑安装的 git checkout**（`pip show sglang` → `Editable project location: /sgl-workspace/sglang/python`），当前 HEAD 为 `5a15cde [Cherry-pick to release/v0.5.12] fix deepseek v4 CP error (#26154)`，且**工作区已有大量本地修改**（`git status --short` 显示 `M` 遍布 `configs/`、`disaggregation/`、`layers/` 等）。该 patch 是针对**另一个** sglang 基线生成的，与这份已被改动过的 checkout 不同源。
3. `git apply --3way` 也不可用：`error: repository lacks the necessary blob to perform 3-way merge` —— patch 记录的 blob 不在本地对象库里，无法三方合并。

### 对现有训练逻辑的风险评估

- **对基线：零风险**。patch 未应用、且（据 B-9）即使应用也不影响不传 `top_logprobs_num` 的请求。
- **对 SC 落地：这是硬阻断**。`档 2`（真实 SGLang head）在**当前证据下不可执行** —— 没有一个能干净应用的 patch。
- **强制盲目应用的风险极高**：11 个失败文件里含 `sampler.py`、`schedule_batch.py`、`sampling_batch_info.py`、3 个 speculative worker —— 这些是**采样与调度热路径**。用 fuzz 强推会产生语义不明的混合体，可能破坏已验证的 R3（`routed_experts`）功能，而 R3 是 122B 基线脚本 `USE_ROUTING_REPLAY=1` 的硬依赖。

### 候选解决路径（需你决策）

| 路径 | 做法 | 优点 | 风险/代价 |
|---|---|---|---|
| **P1（推荐）** | **只提取 SC 所需的最小改动，手工重写为针对当前 checkout 的新 patch**。SC 只需要 `tokenizer_manager.py` 的一个 hunk（`meta_info["output_topk_*"]` 三个字段）+ `numpy` import。**不需要** top-p replay、speculative、disaggregation 的任何改动 | 面积从 18 文件 / 1175 行缩到 **1 文件 / ~25 行**；可 review、可测试、不触碰采样热路径 | 需要人工编写并在单节点 canary 验证 |
| P2 | 找到 patch 的原始完整版本与其对应的 sglang commit，在干净 checkout 上重建环境 | 与上游一致 | 需要定位来源；且本 checkout 已有本地修改，重建会丢失它们（含 R3？需先确认） |
| P3 | 放弃真实 head，**永久使用档 0 合成 head** | 零环境改动 | $q \equiv p$ → 修正项恒为 0 → **SC 在数学上退化为 no-op，无算法价值**。只能用于管道验证，不能用于出效果 |

**P1 的具体范围**（已核对 B-9 的证据）：
```
python/sglang/srt/managers/tokenizer_manager.py
  + import numpy as np
  在 `if top_logprobs_num > 0:` 块内（运行时 :1940-1959），
  把 `meta_info["output_top_logprobs"] = ...` 改为：
      if not return_text_in_logprobs and (finish_reason is not None or state.obj.stream):
          meta_info["output_topk_token_ids"] = b64(int32 ids)
          meta_info["output_topk_log_probs"] = b64(float32 logps)
          meta_info["output_topk_shape"]     = [n, k]
      else:
          <原逻辑不变>
```
**关键**：`else` 分支保留原逻辑 → 即使有第三方代码请求 top-logprobs + `return_text_in_logprobs=True`，行为完全不变。这比上游 PR 的写法更保守。

**状态：未解决，等待你在 P1 / P2 / P3 之间决策。**

---

## B-12 🟡 SGLang 以 **in-process `multiprocessing.Process`** 启动，"动态切换运行时"的可选方案受限

**阶段 / 位置**：Stage 4 / `slime/backends/sglang_utils/sglang_engine.py:50-64,195`

**现象**
```python
def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    from sglang.srt.entrypoints.http_server import launch_server
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
```
SGLang **不是**以独立进程/容器启动的，而是在 Ray actor 进程内 `import sglang` 后 fork。

**根本原因与后果**
`multiprocessing.Process` 继承父进程**已经 import 的模块对象**。因此：
- ❌ **不能**通过"仅对 rollout actor 改 `PYTHONPATH`"在同一 Ray job 内切换 sglang 版本 —— 父 actor 已经 import 了一份。
- ❌ **不能**在一个 job 内同时存在旧/新 sglang。
- ✅ **可以**在**整个 Ray job 粒度**上切换：`ray job submit --runtime-env-json` 的 `env_vars.PYTHONPATH` 对该 job 的所有 worker 生效（`run_qwen35_9b_terminal_bench2_64h20.sh:777` 已在用这个机制注入 PYTHONPATH）。

**风险评估**
"按 `--enable-score-centering` 动态切换运行时"这个要求**只能在 job 提交时刻实现**，不能在 Python 内部实现。好消息是这与需求一致：一次训练 run 的 SC 开关是固定的。

**对 keepalive 的影响：零**。`keepalive_ctl.sh` 不引用 sglang 也不设 PYTHONPATH（已 grep 确认为空），它只按 GPU 利用率跑占位负载，与 sglang 版本完全解耦。切换方案不触碰它。

**设计影响**：见下方 §B-9/B-10 设计方案的"影子 checkout + PYTHONPATH 前置"路线。

---

## B-13 🟡 `atif_adapter.py` 的破坏性切片点是 **3 个独立构建器 / 11 处**，不是我先前记录的 5 处

**阶段 / 位置**：Stage 4 / `slime_plugins/harbor/atif_adapter.py`

**现象（实测 grep）**
`rollout_log_probs` 共出现 30 次，其中**并行数组的变更点**分布在三个**彼此独立**的轨迹构建器里：

| 构建器 | 行 | 服务的 sample_mode | 变更点 |
|---|---|---|---|
| `_trajectory_alignment_debug` | `:377` | 仅调试（`:500` 被调用，写入 metadata） | `:384,411,426,429,439,444` |
| **`_normalized_trajectory_samples_from_detail`** | `:731` | **`trajectory_normalized` + `trajectory_retokenized`（生产基线）** | `:779,823,832,866` |
| `_stitch_turns_preserving_outputs` | `:1255` | `trajectory` / `last_turn`（经 `:481,:682` 调用） | `:1276,1328,1345,1394` |

**我先前记录的"5 处（:426,429,439,444,832）"是错的** —— 那 5 处**混合了调试构建器和生产构建器**，其中 4 处（`:426,429,439,444`）属于 `_trajectory_alignment_debug`，**根本不产出 `Sample`**。

**根本原因**
三个构建器是历史演进产物，各自维护自己的 `response_ids / loss_mask / rollout_log_probs` 三元并行数组，没有共享抽象。

**风险评估**
- **好消息**：真正必须改的**生产路径只有 `_normalized_trajectory_samples_from_detail` 的 4 处**，比原估计少。
- **坏消息**：若只改生产构建器，则 `trajectory` / `last_turn` 模式在开启 SC 时会**缺少 head → 显式抛错**（`sc_terms.py::_require_heads` 已覆盖），不会静默错位。这是可接受的降级（明确失败）。
- **真正的风险**：三处独立实现意味着未来任何一处被改动都可能与 head 失步。这正是需要"切片对齐契约"的原因。

**候选解决路径**：见下方设计方案 §2.2 的 `ParallelTrackBuilder`。

---

## B-14 🟡 `Sample.rollout_topk_*` 字段尚不存在；`to_dict/from_dict` 已可自动承载

**阶段 / 位置**：Stage 4 / `slime/utils/types.py`

**现象**
`Sample` 目前没有 `rollout_topk_token_ids` / `rollout_topk_log_probs` 两个字段（Stage 3 只做了损失侧，未动数据契约）。

**好消息（已核实）**
`Sample.to_dict()`（`types.py:153`）是 `self.__dict__.copy()`，`from_dict()`（`:161`）按 `__dataclass_fields__` 过滤 —— **新增 dataclass 字段会自动随 Ray 序列化，无需改动这两个方法**。

**风险评估**：低。但需注意 `types.py:130-135` 有自定义 `__getattribute__` / `__setattr__`（deprecation warning 用），新增字段前需确认无副作用（Stage 4 第一步验证）。

---

## B-15 🔴 **新增高危**：`/sgl-workspace/sglang` 有 1480 行**未提交**的本地修改，且其中含生产关键功能

**阶段 / 位置**：Stage 4 前置排查 / `/sgl-workspace/sglang`（运行时 sglang checkout）

**现象**
```
$ cd /sgl-workspace/sglang && git diff --shortstat HEAD
 41 files changed, 1480 insertions(+), 168 deletions(-)
$ git stash list
(空)
```
这些修改**没有 commit、没有 stash、没有备份**。抽样查看 `tokenizer_manager.py` 的本地 diff，内容是 slime 自己的功能补丁，例如：
```python
-            if result.success:
+            if result.success and obj.load_format != "delta":
```
即 **delta 权重同步**（`--update-weight-mode delta`）所依赖的 sglang 侧改动，以及 `send_to_scheduler.send_pyobj` 的 await 移除等。

**根本原因**
这份 checkout 是通过 `docker/Dockerfile:109` 的 patch 循环（`sglang.patch`、`sglang-pull_weights.patch`、`sglang-release_hicache.patch`、`sglang-rl-cuda-graph-recapture.patch`）在镜像构建时打上的，`patch(1)` 不产生 git commit，所以全部停留在工作区。

**风险评估（高）**
1. **任何"重建干净 checkout"的方案（B-11 的 P2）会丢失这 1480 行**，其中包含 delta 权重同步 —— 而 `--update-weight-mode` 默认 `full`，但若有 run 在用 `delta`，重建会直接破坏它。
2. **`git apply` / `git checkout` / `git stash` 都有误伤风险**。我在排查中只用了 `--check`（只读），**没有**执行任何写操作。
3. R3（`routed_experts`）本身在 committed HEAD 里（`git show HEAD:...| grep -c routed_experts` → 6），不依赖本地修改，这一点是好消息。

**候选解决路径**
- **必须先做**：在任何环境改动前，`git -C /sgl-workspace/sglang diff HEAD > <备份路径>/sglang_local_mods_$(date +%F).patch` 并校验可回放。**这是 B-11 任何路径的前置条件。**
- 之后 B-11 的 P1（最小 patch）才安全：它只追加一个文件的一个 hunk，不触碰其余 40 个文件。
- 强烈建议：把这 1480 行**提交到一个本地分支**，让环境状态可版本化。这超出本次范围，需你决策。

**状态：未解决，需你决策是否先做备份/提交。**

---

## B-16 🟡 SGLang 的 `PYTHONPATH` 影子切换**可行**（实测），但只能整包、不能部分

**阶段 / 位置**：Stage 4 / 环境切换机制可行性验证

**现象（实测，支持 B-9 的动态切换要求）**

sglang 是 **editable install + MetaPathFinder**：
```
$ pip show sglang | grep Editable
Editable project location: /sgl-workspace/sglang/python
$ ls /usr/local/lib/python3.12/dist-packages/ | grep sglang
__editable___sglang_0_5_12_post1_finder.py
__editable__.sglang-0.5.12.post1.pth
```

**关键实测结论**：
```
# 整包影子 -> 生效
$ PYTHONPATH=/tmp/sgl_shadow python -c "import sglang; print(sglang.__shadow_marker__)"
SHADOW  /tmp/sgl_shadow/sglang/__init__.py        ← 成功遮蔽 editable finder

# 部分影子（只放一个空 __init__.py）-> 子模块回落到真实路径
$ PYTHONPATH=/tmp/sgl_part python -c "import sglang.srt.managers.tokenizer_manager as t; print(t.__file__)"
/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py   ← 混合来源！
```

**根本原因**
stdlib 的 `PathFinder`（走 `sys.path`，含 `PYTHONPATH`）在 `sys.meta_path` 中排在 `_EditableFinder` **之前**，所以 PYTHONPATH 可以遮蔽。但一旦顶层 `sglang` 包被影子接管，其 `__path__` 就只指向影子目录，**子模块必须在影子里存在**，否则 `ImportError`；而若影子里放了顶层 `__init__.py` 却缺子模块，实测会出现**部分回落**（上例），产生新旧混合的 import 图 —— 这是最危险的状态。

**风险评估**
- ✅ "按 `--enable-score-centering` 在 **job 提交时刻**切换 sglang" 是**可行的**，代价是影子目录需要是**完整副本**（75 MB，`du -sh` 实测）。
- ❌ **严禁部分影子**（只覆盖 `tokenizer_manager.py`），会得到混合来源的 import 图，且不报错。
- 结合 B-12（in-process `multiprocessing.Process`）：切换粒度只能是**整个 Ray job**，不能是"仅 rollout actor"。

**结论**：`PYTHONPATH` 整包影子 + job 级切换是 B-9 的可行落地机制。设计见下文 §一。

**状态：已验证可行，纳入设计方案。**

---

# 第三批：复核 `SC_PR2405_PHASE1_CONFLICT_REPORT_zh.md` 后的状态同步（2026-09-26）

按指令深度复核该报告并逐条与源码比对。结论：**报告的"隔离架构"部分高度可用且已被 Stage 1-3 验证采纳；但其 C-1 结论已被推翻。** 另发现一处我自己在 Stage 3 留下的真实缺陷（B-17）。

## 状态变更汇总

| 阻碍 | 变更 | 依据 |
|---|---|---|
| **B-9** | 维持 **Low**，并**同时推翻报告 C-1** | 二者是同一误判；证据见下 |
| **B-10** | **降级**：从"阻断"→"设计已定、路径已验证" | 报告 §4.3 的 8 项数据契约经复核可直接采用；6 项正是待做的 S2 |
| **B-11** | **维持阻断**，报告对此无帮助 | 报告 C-1 的方案是"合入 patch 文件但不应用"，它没有发现文件已损坏 |
| **B-15** | **维持阻断**，报告对此无帮助 | 报告成文时未检查 `/sgl-workspace/sglang` 的工作区状态 |
| **B-13** | 报告同样记录了"5 处"这一错误数字 | 说明该错误源自更早的排查，已在 B-13 更正为 3 构建器/11 处 |
| **B-17** | **新增（中危，我自己的缺陷）** | `sc_terms.py` 未处理 `--allgather-cp`，见下 |

---

## B-9 / 报告 C-1：同一误判，正式推翻

报告 C-1 断言该 patch 是 **"唯一真正的破坏性变更"**、**"无条件的语义替换"**，并据此要求"合入 patch 文件但明确不重建镜像"。

**复核证据（逐条）**

1. 被修改的整段代码位于外层守卫内，且 **patch 没有改动该守卫**：
   ```
   运行时 tokenizer_manager.py:1939-1940
       # 2. Handle top logprobs
       if top_logprobs_num > 0:        ← 守卫
   $ grep -n 'if top_logprobs_num > 0' docker/patch/latest/sglang-top_p.patch
   (无匹配 → 该行既不是 '-' 也不是 '+'，即未被触碰)
   ```
2. `io_struct.py`：`if self.top_logprobs_num is None: self.top_logprobs_num = 0`

→ **不传 `top_logprobs_num` 时，被改代码一行都不执行**；且原版在 `top_logprobs_num == 0` 时**同样不产出** `output_top_logprobs`。所谓"静默变空"不存在。

**因此报告 C-1 的定性（"唯一真正的破坏性变更"）不成立**，其"不重建镜像"的建议虽然结论上无害，但理由是错的 —— 真正的隔离来自请求端不传参，而非"不应用 patch"。

**保留报告 C-1 的一个有价值产出**：它要求"加一个 CPU 断言测试锁死 `output_top_logprobs` 契约"。这一条**仍然值得做**，但语义要改成：断言**外层守卫仍然存在**、且 `else` 分支保留原逻辑。已写入设计文档 §1.2 的 S-3 测试。

---

## B-10 降级：报告 §4.3 的数据契约可直接采用

报告 §4.3 给出一张 8 行的"数据字典契约安全清单"。**我逐行与源码复核，8 行全部成立**，且其中 6 行正是我尚未实施的 S2 工作项：

| 契约行 | 复核结果 | Stage 3 状态 |
|---|---|---|
| `Sample` +2 字段（`= None`） | ✅ dataclass 保证属性存在；且 `types.py:130-135` 的 `__getattribute__`/`__setattr__` **只特判 `rollout_id`**，新字段无副作用（已实读确认） | **TODO（S2）** |
| `ray/rollout.py` 写入整块 gate | ✅ | **TODO（S2）** |
| `_split_train_data_by_dp` + `if key not in data: continue` | ✅ 循环体确有该防御 | **TODO（S2）** |
| `actor.py::_get_rollout_data` + `if key not in rollout_data: continue` | ✅ | **TODO（S2）** |
| `model.py::forward_step` keys → `DataIterator.get_next` 返回 `None` | ✅ `data.py:252-255` 确为 `.get(key, None)` | **TODO（S2）** |
| `data.py::log_rollout_data` skip 列表 | ✅ 遍历实际存在的 key | **TODO（S2）** |
| `sc_terms` 用 `batch.get` + 显式 `raise` | ✅ | **已完成（Stage 3）** |
| Ray `do_post` 条件传 kwarg | ✅ 但**基线不经过此路径**（见下） | 不适用 |

**关键复核结论（比报告更强的一条）**：报告 C-3 指出"Harbor 的生成请求根本不走 `http_utils.post`"。我已验证：
```
$ grep -n 'http_utils' slime_plugins/harbor/llm.py slime_plugins/harbor/rollout.py
(空)
```
`http_utils.post` 的使用者只有 `slime/rollout/sglang_rollout.py:71,201` —— 那是**通用 rollout 路径，基线不用**。

→ **推论：B-10 的 `llm.py` 改造完全不需要动 `http_utils.py`**，因此 PR #2405 对 `post`/`_post`/`do_post` 的三处签名变更（+35 行）以及相应的跨版本 Ray actor 风险（报告 C-3 的核心担忧）**在本地方案中全部不存在**。这把 B-10 的改动面进一步缩小。

**B-10 状态**：阻断 → **设计完备、待实施**。数据通道有了经复核的契约，切片对齐有了 `ParallelTracks` 设计，解码有了 `sc_head_codec`。

---

## B-11 / B-15：报告无法解决，维持阻断

报告 C-1 的方案是"**把 patch 文件更新合入，但不重建镜像 / 不 re-apply**"。这个方案**回避**了 patch 是否可用的问题 —— 因为在报告的设定下（SC 不真正启用），patch 永远不需要被应用。

而我们现在要真正启用 SC，就必须应用它，于是才暴露出：
- **B-11**：文件已截断（`corrupt patch at line 1176`），修复换行后 18 文件中 11 个打不上，含 `tokenizer_manager.py`。
- **B-15**：`/sgl-workspace/sglang` 有 1480 行未提交修改，重建即丢失。

**报告对这两项没有任何帮助**，因为它成文时既未尝试应用 patch，也未检查运行时工作区状态。**维持阻断，等待 P1/P2/P3 与备份决策。**

---

## B-17 🟡 **新增（中危，我自己在 Stage 3 留下的缺陷）**：`sc_terms.py` 未处理 `--allgather-cp`

**阶段 / 位置**：Stage 3 / `slime/backends/megatron_utils/sc_terms.py`

**现象**
```
$ grep -n 'allgather_cp' slime/backends/megatron_utils/sc_terms.py
(空)
```
而 `get_responses`（`loss.py:99`）有一条 `elif args.allgather_cp:` 分支，其行布局与 zigzag **完全不同**：allgather-CP 下每个 rank 持有全局序列的一段**连续**区间，而 zigzag 下持有两段不相邻的 chunk。

**根本原因**
`sc_terms.py` 从 `batch["rollout_topk_*"]` 取 head 时，假设它已按 **zigzag** 切分（因为 `actor.py` 用 `slice_log_prob_with_cp`），而 `get_responses` 在 `allgather_cp=True` 时返回的是 **contiguous** 行。两者行数可能恰好相同而内容错位。

**风险评估**
- **对基线：零影响**。基线 `allgather_cp` 未传（默认 `False`），已 grep 确认两个启动脚本及其父链均无 `ALLGATHER_CP` / `--allgather-cp`。
- **对未来：中危**。若有人在开启 SC 的同时加 `--allgather-cp`，会得到**静默错位的 correction**（行数可能对得上，`_check_head_alignment` 的跨通道对账**能**抓到，但那是最后一道防线而非第一道）。

**PR #2405 恰好解决了这个问题**（这是报告最有价值的遗漏项）：PR 的 `get_score_centering_terms` 包含
- `_slice_allgather_response_rows(values, total_lengths, response_lengths, logits_local_len)` —— 把完整 `[R,...]` 响应张量切到本 rank 的连续行，镜像 `get_responses` 的 allgather 分支；
- 末尾调用 `_allgather_cp_redistribute(...)` 把标量 correction 从 allgather 布局重分布回 zigzag，再与 zigzag 优势相乘。

报告 C-8 还记录了该调用的**签名 bug**（PR 漏传 `args=`，本地 `loss.py:148-155` 要求它并需要 `max_seq_lens=`）—— 已复核签名确实如此。

**修复方案（纳入设计文档 §2.6）**
两级：
1. **立即（零成本）**：在 `score_centering_validate.py` 增加一条校验 —— `enable_score_centering and allgather_cp` → `raise`，明确拒绝而非静默出错。
2. **S2 阶段**：移植 PR 的 `_slice_allgather_response_rows` + 修正签名的 `_allgather_cp_redistribute` 调用，解除该限制。

**状态：已定位，修复方案已定，待 S1/S2 实施。**

