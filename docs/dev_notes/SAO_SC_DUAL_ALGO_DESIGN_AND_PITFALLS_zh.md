# SAO × Score Centering 双算法实施方案与避坑报告

**日期**：2026-09-25
**仓库**：`/path/to/workspace/slime-terminal-RL/`（slime v0.3.0 worktree，分支 `harbor/v030-port`，HEAD `2cf4f30`）
**状态**：**纯设计文档。未修改任何 `.py` / `.sh`，未创建 commit，未 apply 任何 patch。**

**输入资产（均已实读）**
| 资产 | 路径 | 用途 |
|---|---|---|
| 冲突报告 | `SC_PR2405_PHASE1_CONFLICT_REPORT_zh.md` | 工程现状基线 |
| SC 论文 | `paper_list/PO_methods/SCORE CENTERING arXiv-2609.20807v1/sections/{04_method,06_implementation_details}.tex` | 式 (5)(9)(12)(13)(14) |
| SAO 论文 | `paper_list/PO_methods/Single-Rollout Asynchronous OptimizationarXiv-2607.07508v1/{2.preliminary,4.method,5.experiment,7.appendix}.tex` | DIS / TTUR / Skip-Obs GAE |
| SC 参考实现 | THUDM/slime PR #2405（diff 已取，28 文件 +1966/−40） | 工程对照 |
| 既有理论稿 | `paper_list/PO_methods/ASC_PPO_derivation.md`（58 KB，1400+ 行） | 双算法统一推导、事件掩码定义、Critic-first 1:1 |
| 既有对照表 | `pipeline/ppo_config_vs_sao_paper.csv`（40 行） | 本地配置 vs SAO 论文逐项差异 |

---

## 0. 执行摘要：三句话

1. **SC 与 SAO 在数学上正交、在工程上共享一条数据通道**。SC 需要 sampler 的 top-$k$ 分布（新数据）；SAO 的 DIS 需要 sampler 的 sampled-token logprob（**本地已有**，`sample.rollout_log_probs` 无条件填充）。因此 **SAO 的落地成本远低于 SC**，且 SAO 不依赖任何 SGLang patch。

2. **本地已存在两个关键插件钩子**，使得"零回归"目标可以在**几乎不动 slime core** 的前提下达成：`--custom-advantage-function-path`（已有 `slime_plugins/harbor/length_adaptive_gae.py` 作为范例，Skip-Obs GAE 的天然落点）与 `--custom-loss-function-path`（基线已在用，DIS/SC 的天然落点）。**这比 PR #2405 改 core `loss.py` 的路线更安全，且是唯一能让新算法在基线路径上真正生效的路线。**

3. **最大的单点地雷不是算法，是 `trajectory_retokenized` 的 token 重分词与丢弃**。它已经把 observation 的 `rollout_log_probs` 填成了 `0.0`（`atif_adapter.py:832`）——这个值会让 DIS 的 ratio 变成 $e^{\log \pi_\theta - 0} = \pi_\theta$，量级 $\sim 10^{-5}$，**被双侧截断全部 mask 掉**。这恰好与 SAO 要求的"observation 不参与梯度"一致，但它是**巧合而非设计**，必须用显式断言锁死（§4.2）。

---

# 第 1 章 数学公式与代码映射对照

## 1.1 符号总表

| 符号 | 含义 | 本地 Tensor / 变量 | Shape | 阶段 |
|---|---|---|---|---|
| $y_t$ | 第 $t$ 个采样 token | `batch["unconcat_tokens"][i]` | `[R_i]` int64 | Loss |
| $p_v=\pi_\theta(v\mid s_t)$ | trainer 分布 | `logits`（vocab-parallel 分片） | `[1,T,V_p]` fp32 | Loss |
| $\log p_{y_t}$ | trainer 对采样 token 的 logprob | `log_probs`（`get_log_probs_and_entropy`） | `list[[R_i]]` fp32 | Loss |
| $q_v=\mu(v\mid s_t)$ | sampler 分布 | **新** `batch["rollout_topk_log_probs"]` | `list[[R_i,k]]` fp32 | Rollout→Loss |
| $H$ | top-$k$ 头部 token 集合 | **新** `batch["rollout_topk_token_ids"]` | `list[[R_i,k]]` int32 | Rollout→Loss |
| $\log q_{y_t}$ | sampler 对采样 token 的 logprob | `batch["rollout_log_probs"]` | `list[[R_i]]` fp32 | **已有** |
| $\log \pi_{\theta_{old}}$ | 旧策略 logprob | `batch["log_probs"]` | `list[[R_i]]` fp32 | 已有 |
| $\hat A_t$ | 优势 | `batch["advantages"]` | `list[[R_i]]` fp32 | GAE |
| $V_\phi$ | critic 值 | `rollout_data["values"]` | `list[[R_i]]` fp32 | critic fwd |
| $M_{\text{act}}(t)$ | 动作掩码 | `batch["loss_masks"]`（**复用**） | `list[[R_i]]` int | 已有 |
| $r_t$ | DIS ratio | **新** 中间量 `dis_ratio` | `[ΣR_i]` fp32 | Loss |
| $\rho$ | 尾质量比 | **新** 中间量 | `[ΣR_i]` fp32 | Loss |
| $\alpha=\rho f(1/\rho)$ | 尾部权重系数 | **新** 中间量 | `[ΣR_i]` fp32 | Loss |

> **`R_i` 的真实含义**：在 CP>1 且非 `allgather_cp`（基线配置）下，`R_i` **不是**样本的完整 response 长度，而是本 CP rank 持有的 **zigzag 两段拼接后**的局部行数。所有新张量必须走同一条 `slice_log_prob_with_cp` 切分路径，否则与 `log_probs` 错位。这是 §4.2 的核心。

## 1.2 基线 PPO（当前在跑的、必须逐行保持的）

$$
L_{\text{PPO}} = -\mathbb{E}_t\Big[\min\big(r_t^{old}\hat A_t,\ \text{clip}(r_t^{old},1-\epsilon,1+\epsilon_h)\hat A_t\big)\Big],
\qquad r_t^{old}=\frac{\pi_\theta(y_t|s_t)}{\pi_{\theta_{old}}(y_t|s_t)}
$$

**代码映射**（`slime_plugins/harbor/train_loss.py`，非 core）：
```
ppo_kl  = old_log_probs - log_probs           # :98   [ΣR_i]
pg_loss = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high)  # :100
        → ratio = (-ppo_kl).exp()             # ppo_utils.py:132
        → max(-ratio*A, -ratio.clamp(1-ε,1+ε_h)*A)
```
优势来自 `compute_advantages_and_returns` 的 `advantage_estimator=="ppo"` 分支（`loss.py:738-750`）→ `get_advantages_and_returns_batch`，$\gamma=1.0$、$\lambda=1.0$（`arguments.py:955-956` 默认，基线未覆盖）。

**关键事实**：基线 `USE_ROLLOUT_LOGPROBS=0` → `old_log_probs = batch["log_probs"]`（训练侧重算的 $\pi_{\theta_{old}}$），**不是** rollout logprob。这正是 SAO 要替换掉的那一项。

## 1.3 Score Centering

**漂移分解（论文式 1）**：
$$
\underbrace{\mathbb{E}_q[R\,s_{y_t}]}_{\text{PG update}} = \underbrace{\mathbb{E}_q[R]\,\bar s}_{\text{drift}} + \underbrace{\text{Cov}_q(R,s_{y_t})}_{\text{signal}},\qquad \bar s=\sum_v q_v \nabla_\theta\log p_v
$$

**中心化（式 2）**：$\tilde s_{y_t}=s_{y_t}-\bar s$，使 $\mathbb{E}_q[\tilde s_{y_t}]=0$，drift 精确消去。

**尾部模型（式 9）**：
$$
\hat q_v=\begin{cases} q_v & v\in H\\ \rho p_v & v\in T\end{cases},\qquad \rho=\frac{1-\sum_{v\in H}q_v}{1-\sum_{v\in H}p_v}
$$

**头部恒等式（式 12）** —— 利用 $\mathbb{E}_p[s]=0$，尾部求和消掉：
$$
\mathbb{E}_{\hat q}[s_{y_t}]=\sum_{v\in H}(q_v-\rho p_v)s_v
$$

**复合 IS 后（式 14）**：$\alpha=\rho\,f(1/\rho)$，
$$
\mathbb{E}_{\hat q}[\hat w\, s]=\sum_{v\in H}(q_v w_v-\alpha p_v)s_v
$$

**可微 scalar loss（式 13 top-$k$ 版）**：
$$
\boxed{L_{\text{SC}}=-R\Big(\text{sg}[w_{y_t}]\log p_{y_t}-\sum_{v\in H}\text{sg}[q_v w_v-\alpha p_v]\log p_v\Big)}
$$

**代码映射**（PR #2405 `slime/utils/score_centering.py:score_centering_correction`）：
```python
p, q        = train_head_logp.exp(), sampler_head_logp.exp()   # [ΣR_i, k]
p_mass,q_mass = p.sum(-1), q.sum(-1)                           # [ΣR_i]
rho         = (1-q_mass).clamp_min(eps) / (1-p_mass).clamp_min(eps)
alpha       = rho * importance_weights(rho.reciprocal(), mode, low, high)   # f(1/ρ)
weights     = importance_weights((train_head_logp-sampler_head_logp).exp(), ...)  # w_v
residual    = q*weights - alpha.unsqueeze(-1)*p                # sg[·]，全在 no_grad 内
return (residual * train_head_logp).sum(-1)                    # ← 唯一有梯度的乘法
```
$\log p_v$（头部）由 `calculate_topk_log_probs(logits, head_ids, tp_group, chunk_size, temperature)` 提供 —— 一个 vocab-parallel、chunked、自定义 backward 的 autograd Function，返回 `[ΣR_i, k]` 的**全词表归一化** logprob。

**挂载点**：$\log p_{y_t}$ 前的系数是 $\text{sg}[w_{y_t}]$。当 SC 与 IS 复合时，PR 的做法是把 correction 加在**已加权的** pg_loss 之后：
```
pg_loss = pg_loss + advantages.detach() * sc_terms["sc_correction"]   # loss.py:1195
```
注意 PR 在 `pg_loss_type == "reinforce"` 下 `pg_loss = -A.detach()*log_probs`，所以 $-A\log p_{y_t} + A\cdot(\text{correction})$ 正是式 13 展开。**这也意味着 SC 只在 REINFORCE 目标下数学自洽**：与 PPO 的 clip surrogate 复合时，$\log p_{y_t}$ 前不再是单一 $\text{sg}[w]$ 系数，式 13 的推导不成立。这是一条硬约束（PR 也是这么 gate 的）。

## 1.4 SAO

### (a) DIS —— Direct Double-Sided Importance Sampling

$$
r_t(\theta)=\exp\big(\log\pi_\theta(a_t|s_t)-\log\pi_{\text{rollout}}(a_t|s_t)\big)
$$
$$
f(x;\epsilon_\ell,\epsilon_h)=\begin{cases}x & 1-\epsilon_\ell<x<1+\epsilon_h\\ 0 & \text{otherwise}\end{cases}
$$
$$
\boxed{L_{\text{SAO}}=-\hat{\mathbb{E}}_t\big[\text{sg}[f(r_t)]\,\hat A_t\,\log\pi_\theta(a_t|s_t)\big]}
$$

**三点与基线的本质差异**：
1. **分母换人**：$\pi_{\text{rollout}}$ 取代 $\pi_{\theta_{old}}$ → 不需要 `keep_old_actor` 的 old-actor 前向，省一次全量 forward。
2. **双侧 mask 而非 clip**：越界 token 梯度**归零**，不是 clamp 到边界后仍贡献梯度。
3. **目标是 REINFORCE-with-weight**：$f(r_t)$ 是 detached 系数乘在 $\log\pi_\theta$ 上，不是 $r_t\hat A$ 的 min-clip。

> **这与 `icepop_function` 高度相似但不相同**。本地 `loss.py:889-912` 的 `icepop_function` 做 `pg_loss * ice_weight`，其中 `pg_loss` 已经是 PPO 的 clipped surrogate，且 `ice_ratio` 用的是 `train_log_probs_for_tis`（= `batch["log_probs"]`，即 $\pi_{\theta_{old}}$）与 `rollout_log_probs` 之比 —— 即 $\pi_{\theta_{old}}/\pi_{\text{rollout}}$。**SAO 的 ratio 是 $\pi_\theta/\pi_{\text{rollout}}$（当前策略，非旧策略）**，且乘的是裸 $\log\pi_\theta$ 而非 clipped surrogate。差异实质，不能复用。

**论文超参**：$\epsilon_\ell=0.3,\epsilon_h=5.0$（math/TIR）；$\epsilon_\ell=0.8,\epsilon_h=3.0$（coding agent）。**注意这比基线的 `eps_clip=0.2` 宽松 1.5–25 倍**（`ppo_config_vs_sao_paper.csv` 标为 `SEVERE`）。

### (b) Skip-Observation Token-level GAE

轨迹 $T=[a_0,o_0,a_1,o_1,\dots]$。令 $a_{i,N}$ 为动作 $i$ 末 token，$a_{i+1,0}$ 为下一动作首 token：
$$
\delta_{a_{i,N}}=r_t+\gamma V(a_{i+1,0})-V(a_{i,N}),\qquad \hat A(a_{i,N})=\delta+\gamma\lambda \hat A(a_{i+1,0})
$$

即：**GAE 递推只在 $M_{\text{act}}=1$ 的位置之间进行**，observation 位置既不产生 $\delta$、也不消耗 $\gamma\lambda$ 幂次。

对照基线（`ppo_utils.py:344-350`）：
```python
for t in reversed(range(response_len)):
    nextvalues = full_values[t+1] if t < response_len-1 else 0.0
    delta      = full_rewards[t] + gamma*nextvalues - full_values[t]
    lastgaelam = delta + gamma*lambd*lastgaelam      # ← 无条件对每个 t 递推
```
**基线在整个 response span（含 observation）上递推**。`ppo_config_vs_sao_paper.csv` 已把这一条标为 `KEY DIVERGENCE: NOT IMPLEMENTED`。

> **$\gamma=\lambda=1.0$ 下的重要免责**：基线 $\gamma=1,\lambda=1$，此时 $\gamma\lambda=1$，Skip-Obs 的"不消耗幂次"收益为零；唯一残留的差异是 $\delta$ 在 observation 位置的 $V(o)-V(o)$ 项是否引入噪声。数学上 $\gamma=\lambda=1$ 时 GAE 退化为 $\hat A_t=\sum_{l\ge t}r_l - V_t$（Monte-Carlo minus baseline），observation 位置的 $V$ 在望远镜求和中**精确抵消**。→ **Skip-Obs GAE 只有在 $\gamma<1$ 或 $\lambda<1$ 时才产生实质差异**。而 `length_adaptive_gae` 恰好把 $\lambda_{\text{policy}}=1-1/(\alpha L)$ 拉离 1（L=5000 时 0.99987）——所以两者**必须联合考虑**。这是本报告独立推出的结论，SAO 论文未讨论，必须写进验收单（§5 Stage 2）。

### (c) TTUR —— Faster Value Update ($K>1$)

每个 actor step 对应 $K$ 次 critic step（论文 $K=2$）。本地 `train_async.py:74-80` 是 1:1：
```python
value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)   # 1 次
if actor_trains_this_step:
    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
```

### (d) Frozen-Attention Critic

冻结 critic 的 attention 模块，只训 MoE projections。本地**已有机制**：`args.freeze_params_name_list`（`model_provider.py:283-285`）+ `--megatron-config-path` 的 per-role override。**无需新代码，只需配置。**

---

# 第 2 章 统一超参数与接口调度设计

## 2.1 设计原则

采用**正交特征开关**，而非单一 `--algorithm-type` 枚举。理由：

1. 论文本身是正交的 —— SC 论文 §is_composition 明确"SC 与 IS 可复合"；SAO 的 DIS/TTUR/Skip-Obs 三个组件各自独立消融。
2. 枚举会制造 $2^n$ 组合的命名爆炸（`ppo`/`sc`/`sao`/`sao+sc`/`sao_dis_only`/...）。
3. 与 PR #2405 的 `--use-score-centering` 保持一致，未来同步零重命名。
4. 每个开关独立默认 `False`/`None`，零回归的证明可以逐开关归纳，而非枚举穷举。

**但**：新增一个**只读的、不改变行为的**便利参数 `--algorithm-preset`，仅在启动时打印一条"本次运行等价于 preset X"的诊断日志，用于人工核对。它**不设置任何其他参数**（避免隐式行为替换，违反你的红线）。

## 2.2 新增参数总表

添加位置：`slime/utils/arguments.py` → `add_algo_arguments(parser)`（约 `:780`–`:1030` 区间，紧邻 `--advantage-estimator` / `--use-tis`）。

### A 组：Score Centering（3 个，与 PR #2405 完全一致）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--use-score-centering` | `store_true` | `False` | SC 总闸 |
| `--score-centering-top-k` | `int` | `128` | $k$；SC 关闭时不读取 |
| `--pg-loss-type` | `str`, choices `{ppo,reinforce,cispo}` | `None` | `None`→由 `get_pg_loss_type` 推导；基线 `advantage_estimator=ppo` → `"ppo"` |

### B 组：SAO DIS（4 个）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--use-sao-dis` | `store_true` | `False` | DIS 总闸 |
| `--sao-dis-eps-low` | `float` | `0.3` | $\epsilon_\ell$；SAO 论文 math 配置 |
| `--sao-dis-eps-high` | `float` | `5.0` | $\epsilon_h$ |
| `--sao-dis-ratio-source` | `str`, choices `{current,old}` | `current` | `current`=$\pi_\theta/\pi_{\text{rollout}}$（SAO 原义）；`old`=$\pi_{\theta_{old}}/\pi_{\text{rollout}}$（等价 icepop，供消融） |

> 默认值取论文值而非"最保守值"，因为 **DIS 关闭时它们完全不被读取**，给一个论文一致的默认值比给 `0.2` 更不容易误用。`--use-sao-dis` 才是零回归的唯一守门人。

### C 组：SAO Skip-Observation GAE（3 个）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--use-skip-obs-gae` | `store_true` | `False` | Skip-Obs 总闸 |
| `--skip-obs-action-mask-source` | `str`, choices `{loss_mask,explicit}` | `loss_mask` | 动作掩码来源。`loss_mask` 复用 `batch["loss_masks"]`；`explicit` 读新字段 `sample.action_mask`（阶段二才实现） |
| `--skip-obs-reward-placement` | `str`, choices `{last_action_token,last_token}` | `last_action_token` | 终局 reward 落点。基线落在 `k[-1]`（response 最后一个 token，可能是 observation！见 §4.5） |

### D 组：SAO TTUR + Critic（2 个）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--critic-updates-per-actor-update` | `int` | `1` | $K$。`1` = 现状 |
| `--critic-ttur-data-reuse` | `str`, choices `{same_batch,resplit}` | `same_batch` | $K>1$ 时后续 critic step 的数据来源 |

> **Frozen-Attention 不新增参数** —— 复用已有的 `--freeze-params-name-list` + `--megatron-config-path` 的 critic role override。这是本方案刻意的"零新增面"决策。

### E 组：诊断（1 个，不改变行为）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--algorithm-preset` | `str` | `None` | 仅打印诊断。**不设置任何其他参数。** 不传 = 不打印 |

**合计新增 13 个参数，其中 4 个是总闸（默认全 False），1 个纯诊断。**

## 2.3 组合校验矩阵

所有校验集中在 `slime_validate_args` 内一个**被 gate 包裹**的函数（§3.1），SC/SAO 全关时该函数第一行 return。

| 组合 | 合法 | 理由 / 处理 |
|---|---|---|
| 全关（基线） | ✅ | 零回归路径 |
| SC only | ✅ | 需 `pg_loss_type=reinforce`（PR 强制） |
| DIS only | ✅ | 与 `advantage_estimator=ppo` 兼容 |
| Skip-Obs only | ✅ | 与 `advantage_estimator=ppo` 兼容 |
| TTUR only | ✅ | 与任何 estimator 兼容 |
| **SC + DIS** | ⚠️ **条件合法** | 见下方专论 |
| SC + Skip-Obs | ✅ | 完全正交（一个在 Loss，一个在 GAE） |
| DIS + Skip-Obs + TTUR | ✅ | 这就是完整 SAO |
| SC + 完整 SAO | ⚠️ | = ASC-PPO（`ASC_PPO_derivation.md` 的主张）；条件同 SC+DIS |
| SC + `advantage_estimator ∈ {gspo,cispo}` | ❌ | PR 硬 raise；SC 是 REINFORCE 目标 |
| SC + `--use-opd/--use-opsm/--use-unbiased-kl` | ❌ | PR 硬 raise |
| DIS + `--use-tis` | ❌ | 两者都是 token-wise IS 重加权，复合会重复加权 |
| Skip-Obs + `--custom-advantage-function-path` 指向非 SAO 实现 | ❌ | 单一优势来源；见 §3.4 的融合方案 |
| TTUR($K>1$) + `advantage_estimator != ppo` | ❌ | 无 critic 时 $K$ 无意义 |

### 专论：SC + DIS 能否复合？

**数学上可以，但必须用同一个 $f$。** SC 论文式 14 要求 $\alpha=\rho f(1/\rho)$ 且 $w_v=f(p_v/q_v)$ —— **$f$ 必须是那个能作用在"每个候选 token 和建模尾部"上的函数**。SAO 的 $f$ 是
$$
f_{\text{SAO}}(x)=x\cdot\mathbf 1[1-\epsilon_\ell<x<1+\epsilon_h]
$$
这正是 SC 论文的 `mis_weight`（MIS）形式，只是边界从 $(0.5,5.0)$ 换成 $(1-\epsilon_\ell,1+\epsilon_h)$。

**结论：SC + DIS 是合法的，等价于 "SC composed with MIS"，边界取 DIS 的边界。**

**但 PR #2405 的 `get_score_centering_is_config` 读的是 `args.tis_clip_low/args.tis_clip`，且要求 `args.use_tis=True`**。而我们的 DIS 用的是 `--sao-dis-eps-low/high` 且 `--use-sao-dis`。→ **必须扩展 `get_score_centering_is_config` 使其能从 DIS 取边界**，映射为：
```
mode="mis", low = 1 - sao_dis_eps_low, high = 1 + sao_dis_eps_high
```
注意 SC 的 `importance_weights(..., "mis", low, high)` 的 low/high 是**ratio 的绝对边界**，而 DIS 的 $\epsilon$ 是**偏离 1 的量**。这个 $\pm 1$ 的转换是一个极易写错的地方，必须有单测（§5 Stage 2）。

另外：$\epsilon_\ell=0.3$ → `low=0.7`；$\epsilon_h=5.0$ → `high=6.0`。则 $1/\rho$ 若落在 $[0.7,6.0]$ 外，$\alpha=\rho\cdot 0=0$，尾部 correction 消失。$\rho$ 典型值 $\approx 1$（头部覆盖 99.9%），$1/\rho\approx 1$ 在界内 → 正常。但 SC 论文 §topk_ablation 指出量化场景下最差 batch 头部覆盖仅 95.8%，此时 $\rho$ 可能剧烈偏离 → 需监控 `sc_tail_ratio`。

## 2.4 启动脚本配置示例

### (a) 基线 PPO（当前在跑的）—— **零改动**

```bash
./run_27b_dense_128gpu_ppo_256k.sh
```
不传任何新参数。解析后 `use_score_centering=False`、`use_sao_dis=False`、`use_skip_obs_gae=False`、`critic_updates_per_actor_update=1`、`pg_loss_type=None`。**执行路径与今天逐行相同**（§3.6 有形式化证明）。

### (b) 纯 SAO（DIS + Skip-Obs + TTUR + Frozen-Attention Critic）

```bash
# 注意：本地 launcher 对 --loss-type custom_loss 是硬编码的，所以 SAO 必须落在
# Harbor fork 的 train_loss.py 里（见 §3.5 / §4.1），而不是 core policy_loss_function。
EXTRA_MISC_ARGS="--moe-token-dispatcher-type alltoall --use-rollout-routing-replay \
  --use-sao-dis \
  --sao-dis-eps-low 0.8 \
  --sao-dis-eps-high 3.0 \
  --sao-dis-ratio-source current \
  --use-skip-obs-gae \
  --skip-obs-action-mask-source loss_mask \
  --skip-obs-reward-placement last_action_token \
  --critic-updates-per-actor-update 2 \
  --custom-advantage-function-path slime_plugins.harbor.sao_gae.skip_obs_length_adaptive_gae \
  --algorithm-preset sao" \
USE_ROLLOUT_LOGPROBS=0 \
GAE_ALPHA=1.5 \
MEGATRON_CONFIG_PATH=examples/harbor_terminal_rl/sao_critic_frozen_attn.yaml \
./run_27b_dense_128gpu_ppo_256k.sh
```
配套 `sao_critic_frozen_attn.yaml`（per-role Megatron override）：
```yaml
critic:
  freeze_params_name_list:
    - "self_attention"
    - "linear_attention"     # 27B 是 GDN hybrid，两类都要冻
  lr: 5.0e-6                 # SAO 论文值
```
> 注意 `--sao-dis-eps-low 0.8 / --sao-dis-eps-high 3.0` 用的是论文的 **coding agent** 配置（Terminal RL 与 SWE-Bench 同属 agentic coding），而非 math 的 `0.3/5.0`。

### (c) 纯 SC

```bash
EXTRA_MISC_ARGS="--moe-token-dispatcher-type alltoall \
  --use-score-centering \
  --score-centering-top-k 128 \
  --pg-loss-type reinforce \
  --advantage-estimator grpo \
  --disable-grpo-std-normalization \
  --calculate-per-token-loss \
  --algorithm-preset score_centering" \
ADVANTAGE_ESTIMATOR=grpo \
USE_CUSTOM_NO_ENTROPY_LOSS=0 \
ROLLOUT_TEMPERATURE=1.0 \
ROLLOUT_TOP_P=1.0 \
ROLLOUT_TOP_K=-1 \
ENTROPY_COEF=0 \
KL_COEF=0 \
./run_27b_dense_128gpu_ppo_256k.sh
```
> `USE_CUSTOM_NO_ENTROPY_LOSS=0` 让它回落到 core `policy_loss_function`（SC 在那里实现）。**但这同时丢掉了 Harbor fork 的 entropy 惰性计算**，在 256k/27B 上可能 OOM。→ 这就是为什么 §3.5 的正确答案是"在 Harbor fork 里也实现 SC"，而非依赖 core。
> **另外**：SC 在本集群目前**跑不起来**（SGLang 未打 patch + Harbor `llm.py` 不透传 head），见冲突报告 §5 与本报告 §4.4。此配置是接口预留。

### (d) ASC-PPO（SC + 完整 SAO，`ASC_PPO_derivation.md` 的目标形态）

```bash
EXTRA_MISC_ARGS="--moe-token-dispatcher-type alltoall \
  --use-score-centering --score-centering-top-k 128 --pg-loss-type reinforce \
  --use-sao-dis --sao-dis-eps-low 0.8 --sao-dis-eps-high 3.0 \
  --use-skip-obs-gae --critic-updates-per-actor-update 1 \
  --advantage-estimator ppo \
  --custom-advantage-function-path slime_plugins.harbor.sao_gae.skip_obs_length_adaptive_gae \
  --algorithm-preset asc_ppo" \
...
```
> `ASC_PPO_derivation.md` §6.3 明确主张 **Critic-first 1:1**（$K=1$，先 critic 一步再 actor 一步，用冻结快照的优势），与 SAO 论文的 $K=2$ 不同。本方案通过 `--critic-updates-per-actor-update` 同时支持两者（`1`=ASC-PPO，`2`=SAO）。注意本地 `train_async.py:76-78` **已经是** critic-first 且用冻结 values（`external_data=value_refs`），所以 ASC-PPO 的 1:1 语义**本地已满足**，不需要新代码。

---

# 第 3 章 实施路径与系统改造模块清单

## 3.1 Config & Arguments

**文件**：`slime/utils/arguments.py`

| 改动 | 位置 | 内容 |
|---|---|---|
| 新增 13 参数 | `add_algo_arguments` ≈`:1030` | §2.2 全表 |
| 新增 gate 函数 | `slime_validate_args` ≈`:1685` 入口 | 见下 |

```python
# arguments.py, slime_validate_args 内最前
def _any_new_algo_enabled(args) -> bool:
    return (
        getattr(args, "use_score_centering", False)
        or getattr(args, "use_sao_dis", False)
        or getattr(args, "use_skip_obs_gae", False)
        or getattr(args, "critic_updates_per_actor_update", 1) != 1
        or getattr(args, "pg_loss_type", None) is not None
    )

def slime_validate_args(args):
    if _any_new_algo_enabled(args):
        from slime.utils.dual_algo_validate import validate_dual_algo_args   # 新文件
        validate_dual_algo_args(args)          # §2.3 全矩阵在这里
    args.eval_datasets = _resolve_eval_datasets(args)
    ...  # 以下原样不动
```

**与 PR #2405 的关键差异**：PR 是**无条件**调用 `get_pg_loss_type(args)` 和 `validate_score_centering_args(args)`。我们用 `_any_new_algo_enabled` gate 包裹 —— 基线连 `import` 都不发生。这是冲突报告 C-2 的结论，此处沿用并扩展到 SAO。

**`use_rollout_logprobs` 那条 assert（`arguments.py:1796-1797`）**：
```python
if args.use_rollout_logprobs:
    assert not args.use_tis, "..."
```
PR 改成 `and get_pg_loss_type(args) != "reinforce"`。我们改成：
```python
if args.use_rollout_logprobs and _pg_loss_type_or_ppo(args) != "reinforce":
```
其中 `_pg_loss_type_or_ppo` 在 gate 为 False 时**直接 `return "ppo"`，不 import 任何新模块**。基线 `use_rollout_logprobs=False` → 整条短路，改前改后均不执行。

## 3.2 Rollout Engine & SGLang 通信

### SC 侧（需要新数据）

| 层 | 文件 | 改动 |
|---|---|---|
| SGLang | `docker/patch/latest/sglang-top_p.patch` | 合入 PR 的 +54 行（`output_topk_*` 二进制字段）。**不 re-apply 到当前运行时**（§4.4） |
| slime 通用 rollout | `slime/rollout/sglang_rollout.py` | `payload.update(score_centering_request(args, sampling_params))`，gate 在 `score_centering_request` 内首行 |
| HTTP | `slime/utils/http_utils.py` | `post/_post/do_post` 新增 `score_centering_top_k=None`。**Ray `.remote()` 侧用 `**extra` 条件传参**（冲突报告 C-3.1） |
| **Harbor rollout** | `slime_plugins/harbor/llm.py` | **新增**：SC 开启时 payload 加 `return_logprob=True, top_logprobs_num=k`；response 解析 `meta_info["output_topk_*"]` |
| Harbor 适配 | `slime_plugins/harbor/atif_adapter.py` | **新增**：把 head 沿 turn 切分/重分词逻辑同步搬运（§4.2，最高风险项） |

### SAO 侧（数据已有！）

| 层 | 改动 |
|---|---|
| SGLang | **无** |
| slime rollout | **无** |
| Harbor `llm.py` | **无** —— `llm.py:122` 已 `return_logprob: True`，`:150-152` 已解析 `output_token_logprobs` |
| `atif_adapter.py` | **无** —— `sample.rollout_log_probs` 已无条件填充（`:650`, `:891`, `:1344`） |
| `ray/rollout.py` | **无** —— `:1009-1010` 已无条件透传 `rollout_log_probs` |
| `actor.py` | **无** —— `:292-296` 已 CP-slice `rollout_log_probs` |

> **这是本报告最重要的工程结论**：**SAO 的 DIS 在数据通道上零改动**。$\pi_{\text{rollout}}$ 已经端到端存在于基线里（正因为 `GET_MISMATCH_METRICS` 的诊断需求，Harbor 无条件填充它）。DIS 只需在 loss 里读 `batch["rollout_log_probs"]`。

## 3.3 Data Pipeline & ATIF Adapter

### 3.3.1 `trajectory_retokenized` 的真实结构

`atif_adapter.py:778-895`，每个 chunk 的 response span 按 turn 交替拼接：

```
chunk.response_ids = [ out_0 | ctx_1 | out_1 | ctx_2 | out_2 | ... ]
chunk.loss_mask    = [ 1×n_0 | 0×m_1 | 1×n_1 | 0×m_2 | 1×n_2 | ... ]
chunk.rollout_log_probs
                   = [ lp_0  | 0.0×m_1| lp_1 | 0.0×m_2| lp_2  | ... ]
                                 ↑ atif_adapter.py:832
```

**`loss_mask` 就是 $M_{\text{act}}$**（`:865` `loss_mask.extend([1]*len(turn.output_ids))` vs `:831` `loss_mask.extend([0]*len(context_ids))`）。

→ **Skip-Obs GAE 的动作掩码无需新字段**，`--skip-obs-action-mask-source loss_mask` 是正确默认。

**但有一个 caveat**：`loss_mask=0` 还覆盖了另外两类位置：
- `:425-428` retokenize drift 修正时被追溯 mask 掉的 token（`loss_mask[start:matched_len] = [0]*masked`）
- `:438` `context_tail`

这些**不是 observation**，但在 Skip-Obs 语义下会被同样跳过。影响：GAE 递推会跳过几个本属动作的 token。**方向是保守的**（少给这几个 token 信用），但必须量化：在 Stage 2 加一个统计 `skip_obs_masked_action_token_ratio`，若 > 1% 则需要 `--skip-obs-action-mask-source explicit`（阶段二实现独立 `action_mask` 字段）。

### 3.3.2 SC 的 head 搬运（最高风险）

`trajectory_retokenized` 对 token 做了三类破坏性操作：
| 操作 | 代码 | 对 head 的影响 |
|---|---|---|
| `prompt_suffix_dropped_tokens` | `:856` | prompt 侧丢弃，**不影响 response head** |
| `response_prefix_dropped_tokens` | `:858` | **response 头部丢弃 → head 必须同步丢弃前 N 行** |
| `response_suffix_dropped_tokens` | `:860` | **response 尾部丢弃 → head 必须同步丢弃后 N 行** |
| `split`（新 chunk） | `:815-827` | response_ids 清空重建 → head 必须同步切分 |
| retokenize 追溯 mask | `:425-428` | token 保留但 mask 归零 → head 行保留，但需填 dummy |

**必须的实现约束**：head 的搬运**必须与 `rollout_log_probs` 的搬运在同一个循环、同一个 `extend/slice` 调用点**，而不是事后按长度补齐。PR #2405 在 `types.py::append_response_tokens` 里做这件事（本地无此方法），我们必须在 `atif_adapter.py` 的每一处 `rollout_log_probs.extend(...)` 旁边加一处 `topk_ids.extend(...)`。

共有 **5 处** `rollout_log_probs` 的写入点需要配对（`:426`, `:429`, `:439`, `:444/:865`, `:832/:1328`）。**漏掉任一处 = head 与 token 静默错位 = SC correction 完全错误且不报错。** 这是 §4.2 的核心地雷。

## 3.4 Advantage & GAE 模块

**新文件**：`slime_plugins/harbor/sao_gae.py`
**挂载**：`--custom-advantage-function-path slime_plugins.harbor.sao_gae.skip_obs_length_adaptive_gae`
**Core 改动：零。** `compute_advantages_and_returns` 的 `custom_advantage_function_path` 分支（`loss.py:727-730`）已存在。

### 算法（向量化 + 分段）

```
输入（per sample i，均为本 CP rank 的局部视图）:
  values[i]       : [R_i]  fp32   ← critic forward，CP-sliced
  token_rewards[i]: [R_i]  fp32   ← KL penalty + terminal reward
  loss_masks[i]   : [R_i]  int    ← M_act
  total_lengths[i], response_lengths[i]

步骤:
 1. CP all-gather 到完整 response 视图（复用 all_gather_with_cp，
    与 length_adaptive_gae 完全相同的路径）
        full_values, full_rewards, full_mask : [R_i_full]
 2. 提取动作位置索引:  idx = nonzero(full_mask).squeeze(-1)    [N_act]
 3. 压缩到动作子序列:
        v_act = full_values[idx]                               [N_act]
        r_act = segment_sum(full_rewards, idx)                 [N_act]
           ← 区间奖励聚合：把 (idx[j], idx[j+1]) 之间所有 observation
             位置的 reward 加到 r_act[j]。基线下 observation 位置的
             reward 是 -kl_coef*kl = 0（kl_coef=0），故聚合是 no-op，
             但必须实现，否则 kl_coef≠0 时丢奖励。
 4. 在压缩序列上跑标准 GAE（复用 vanilla_gae / chunked_gae）:
        A_act, Ret_act = gae(r_act, v_act, gamma, lambda)       [N_act]
 5. 散射回完整长度:
        full_A = zeros(R_i_full); full_A[idx] = A_act
        full_Ret= zeros(R_i_full); full_Ret[idx] = Ret_act
    ← observation 位置 advantage=0。因为 loss_mask=0，它们在
      sum_of_sample_mean 里本就被剔除，填 0 与填任何值等价，
      但填 0 让 log_rollout_data 的 advantage 统计不被污染。
 6. CP re-slice 回本 rank:  slice_log_prob_with_cp(...)
```

**向量化要点**：步骤 3 的 `segment_sum` 用 `torch.zeros(N_act).index_add_(0, bucket_id, full_rewards)` 实现，其中 `bucket_id = (cumsum(full_mask) - 1).clamp_min(0)`。$O(R)$，无 Python 循环。

**$\lambda$ 的选择**：复用 `length_adaptive_gae` 的双 $\lambda$，但 **$L$ 必须换成 $N_{\text{act}}$ 而非 $R_i$**：
$$
\lambda_{\text{policy}}=1-\frac{1}{\alpha\cdot N_{\text{act}}}
$$
理由：SAO 论文 §appendix 在 step-level 消融里明确"scale $\lambda$ based on the total number of steps rather than the raw token length"。Skip-Obs 把时钟从 token 换成了**动作 token**，所以 $L$ 也应换。这是一个独立设计决策，需要 A/B（Stage 4）。

**CP 一致性**：`length_adaptive_gae` 的 docstring `:29-33` 已警告 `dist.nn.all_reduce` 是无条件的，循环长度必须每 rank 相同。我们的实现循环长度也是 `len(response_lengths)`，且 `all_gather_with_cp` 在步骤 1 无条件调用 → **CP 不会 desync**。但 `N_act` 在不同 sample 间不同（这是 per-sample 的，不是 per-rank 的），不影响集合通信计数。

## 3.5 Trainer & Loss 模块

### 3.5.1 Core vs Harbor fork 的解耦策略 —— 本方案的核心决策

冲突报告 §C-8 已确认：基线走 `train_loss.py` fork，core `policy_loss_function` 是死代码。PR #2405 只改 core → **在基线上完全不生效**。

**三个候选方案**：

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| ① 只改 core | 照搬 PR | 与上游同步 | **新算法在基线上无效**，等于没做 |
| ② core + fork 双份实现 | 两处各写一遍 | 都生效 | **逻辑重复，必然漂移**；`train_loss.py` 已是 fork，再 fork 一层 |
| ③ **提取共享纯函数，两处调用** | 新建 `slime/utils/pg_objectives.py`，core 与 fork 都调它 | 单一真源；fork 只多 3 行 | 需要设计一个干净的函数边界 |

**推荐 ③**。具体：

**新文件 `slime/utils/pg_objectives.py`**（纯函数，无 Megatron 依赖，可 CPU 单测）：
```python
def apply_sao_dis(args, *, pg_loss_or_logp, log_probs, rollout_log_probs, old_log_probs):
    """返回 (weighted, metrics)。DIS 关闭时原样返回 pg_loss_or_logp 与空 dict。"""
    if not getattr(args, "use_sao_dis", False):
        return pg_loss_or_logp, {}
    ref = log_probs if args.sao_dis_ratio_source == "current" else old_log_probs
    ratio = (ref - rollout_log_probs).exp()
    low, high = 1.0 - args.sao_dis_eps_low, 1.0 + args.sao_dis_eps_high
    w = torch.where((ratio > low) & (ratio < high), ratio, torch.zeros_like(ratio)).detach()
    return pg_loss_or_logp * w, {
        "sao_dis_ratio": ratio.detach(),
        "sao_dis_maskfrac": (w == 0).float(),
        "sao_dis_weight": w,
    }

def apply_score_centering(args, batch, logits, *, advantages, pg_loss):
    """SC 关闭时返回 (pg_loss, {})。"""
    if not getattr(args, "use_score_centering", False):
        return pg_loss, {}
    from slime.utils.score_centering import score_centering_correction
    from slime.backends.megatron_utils.sc_terms import get_score_centering_terms
    terms = get_score_centering_terms(args, batch, logits)
    return pg_loss + advantages.detach() * terms["sc_correction"], terms
```

**`train_loss.py` 的改动只有 3 处、共约 12 行**（全部 gate 后为 no-op）：
```python
# 1) 目标分派（:100 原地）
if _pg_loss_type_or_ppo(args) == "reinforce":
    pg_loss = -advantages.detach() * log_probs
    pg_clipfrac = torch.zeros_like(log_probs)
else:
    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

# 2) DIS（:104 之后，与 use_tis 分支平行且互斥）
pg_loss, dis_metrics = apply_sao_dis(args, pg_loss_or_logp=pg_loss,
    log_probs=log_probs, rollout_log_probs=torch.cat(batch["rollout_log_probs"]),
    old_log_probs=old_log_probs)

# 3) SC（:152 之前）
pg_loss, sc_terms = apply_score_centering(args, batch, logits,
    advantages=advantages, pg_loss=pg_loss)

# 4) 指标上报（:226 之后）
for k, v in {**dis_metrics, **sc_terms}.items():
    reported_loss[k] = sum_of_sample_mean(v.detach())
```

**Core `policy_loss_function` 同样调这两个函数**，保证行为一致。

> **DIS 的挂载位置有一个数学细节**：SAO 的目标是 $f(r_t)\hat A_t\log\pi_\theta$，而 `pg_loss` 在 `pg_loss_type=="ppo"` 下是 $\max(-r^{old}A, \dots)$。把 $f(r_t)$ 乘上去得到 $f(r_t)\cdot\text{PPOsurrogate}$ —— **这不是 SAO，是 "PPO with DIS gating"**（论文表格里的 `GRPO (w/ DIS)` 变体，是论文自己的一个 baseline）。**真 SAO 需要 `--pg-loss-type reinforce`**。因此：
> - `--use-sao-dis` + `--pg-loss-type reinforce` = 论文的 SAO ✅
> - `--use-sao-dis` 单独（`pg_loss_type` 推导为 `ppo`）= 论文的 `PPO w/ DIS` 消融变体 ✅ 也合法
> 校验逻辑需在 `--algorithm-preset sao` 时**警告**（不是报错）用户是否忘了 `--pg-loss-type reinforce`。

### 3.5.2 SC 的 TP/CP 适配

`get_score_centering_terms`（PR `loss.py:+70`）在本地需要 **3 处修正**（冲突报告 C-8 已指出其一）：

| 问题 | PR 代码 | 本地修正 |
|---|---|---|
| 缺 `args` | `_allgather_cp_redistribute(res, logits_local_len=..., total_lengths=..., response_lengths=...)` | 本地签名（`loss.py:147-155`）要求 `args=args`，且需 `max_seq_lens=` |
| 缺 vocab-parallel helper | 调用 `_maybe_all_reduce` / `_get_vocab_parallel_rank_size` | 本地 grep=0，**需一并新建**（可仿 `_VocabParallelEntropy` 的 `dist.all_reduce(..., group=process_group)` 写法） |
| `DiskTensorRef` 分支 | `local_heads()` 里判 `isinstance(value, DiskTensorRef)` | 本地无该类 → 删分支，走纯内存（冲突报告 C-5 方案 A） |

**放置建议**：新建 `slime/backends/megatron_utils/sc_terms.py` 而非塞进 1357 行的 `loss.py`。理由：`loss.py` 是基线热路径文件，新增 70 行会让 AST diff 审计变噪；独立文件让"core `loss.py` 只多 4 行 import + 2 行调用"这一事实可验证。

### 3.5.3 TTUR（$K>1$）在 Ray/Megatron 下的可行性

**结论：可行，但有一个必须回答的语义问题。**

`train_async.py:76` 目前：
```python
value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
```
`async_train`（`actor_group.py:120-136`）→ 每个 critic worker 的 `train(rollout_id, ref, external_data)` → `train_critic`（`actor.py:426-451`）→ `forward_only(get_values)` + `compute_advantages_and_returns` + `train(...)` 一次完整 epoch。

`train()`（`model.py:729`）内部 `for step_id in range(num_steps_per_rollout)` 已经是多步（每步一个 global batch）。

**实现 $K$ 的两条路**：

| 路径 | 做法 | 评价 |
|---|---|---|
| **A（推荐）** `train_async.py` 外层循环 | `for _ in range(K): value_refs = critic_model.async_train(...)` | 改动 2 行；每次重跑 `forward_only(get_values)`（一次全量 critic forward，非 trivial 开销）+ 重算 advantages |
| B `train_critic` 内层循环 | 在 `actor.py:434` 的 `train(...)` 外包 `for _ in range(K)` | 省掉重复 forward；但 **values/advantages 不更新**，第 2..K 次是在过时目标上拟合 |

**语义抉择**：SAO 论文说 "$K$ updates to the value network per policy update"，未明确 target 是否重算。
- 路径 A = target 每次重算 → **真 TTUR**，critic 追上当前自己的 bootstrap。
- 路径 B = target 冻结 → 等价于对同一批数据多跑 epoch，是**数据重用**而非 TTUR。

**`ASC_PPO_derivation.md` §6.3 明确反对路径 B 的变体**（"所有 microbatch 只累积梯度，不能中间 `step()`"，且要求目标冻结 + 恰好一步）。

→ **方案**：`--critic-updates-per-actor-update K` 走路径 A；`--critic-ttur-data-reuse {same_batch,resplit}` 控制第 2..K 次是否重新 DP 切分。**默认 `K=1` 时两条路径都不触发，`train_async.py` 的字节码不变**（用 `if K == 1: <原语句> else: for ...`，或更干净地 `for _ in range(K)` —— 后者在 $K=1$ 时语义完全等价但字节码变了，AST 审计会报差异；建议**显式 if/else 保留原语句**，以便 §5 Stage 1 的字节码对比通过）。

**成本估算（27B/256k）**：一次 critic `forward_only(get_values)` ≈ 一次全量 forward。基线一个 step 的 critic 侧已有 1×forward + 1×fwd/bwd。$K=2$ 路径 A → 2×forward + 2×fwd/bwd，critic 侧耗时约 2×。critic 与 actor 共享 placement group（`placement_group.py`: `result["critic"] = result["actor"]`），且 PPO 强制 `offload_train=True` → **每次 `async_train` 都有一次 offload/reload 的 166 process-group 重建**（参见 122B 脚本 `:334` 的注释）。$K=2$ 意味着**每 step 多一次完整的 NCCL 组 teardown/rebuild**。这是 TTUR 在本集群的真实主要成本，不是算力。→ 必须在 Stage 3 实测 `train_wait` 计时。

## 3.6 零回归形式化证明（全开关默认）

基线解析后：`use_score_centering=False`、`use_sao_dis=False`、`use_skip_obs_gae=False`、`critic_updates_per_actor_update=1`、`pg_loss_type=None`、`algorithm_preset=None`。

```
arguments.py:
  _any_new_algo_enabled(args) = False or False or False or (1 != 1)=False or (None is not None)=False
                              = False
  → 不 import dual_algo_validate / score_centering / pg_objectives    ✅
  _pg_loss_type_or_ppo(args) → gate False → return "ppo"（不 import）  ✅
  use_rollout_logprobs=False → assert 整条短路（改前改后同）            ✅

train_async.py:
  K=1 → 走显式 if 分支的原语句，字节码与合并前相同                      ✅

compute_advantages_and_returns (loss.py:727):
  custom_advantage_function_path is None（基线未传）
  → 走 advantage_estimator=="ppo" 分支，get_advantages_and_returns_batch  ✅
  （--use-skip-obs-gae 默认 False，且它只通过 custom path 生效）

train_loss.py::policy_loss_no_entropy_when_disabled:
  _pg_loss_type_or_ppo → "ppo" → else 分支 → compute_policy_loss(...)    ✅ 与原行同
  apply_sao_dis: use_sao_dis=False → 首行 return (pg_loss, {})            ✅ 恒等
  apply_score_centering: use_score_centering=False → return (pg_loss, {}) ✅ 恒等
  for k,v in {**{}, **{}}.items(): → 空循环                               ✅

core policy_loss_function: 同上（且基线不执行）                           ✅

ray/rollout.py / actor.py / model.py / http_utils.py:
  与冲突报告 §4.2 相同：SC gate 为 False → 新 key 根本不产生，
  train_data 字典 byte-identical；.remote() 不含新 kwarg               ✅

SGLang: patch 文件更新但不 re-apply → 运行时行为不变                      ✅
```

**结论：两个基线脚本在不传任何新参数时，参数字典（除新增 13 个默认值 key）、数据流、通信结构、损失计算与合并前 100% 一致。**

---

# 第 4 章 潜在陷阱、技术阻断与排雷方案

## 4.1 🔴 Harbor 独立 Fork 阻断

**现状**（冲突报告 §1.1 / C-8）：`strongest.sh:505` `USE_CUSTOM_NO_ENTROPY_LOSS=1` → `64h20.sh:586` 无条件 `--loss-type custom_loss --custom-loss-function-path slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled`。Core `policy_loss_function` 是死代码。

**风险**：若按 PR #2405 只改 core，`--use-score-centering` / `--use-sao-dis` **会被静默接受但完全无效** —— 参数被 argparse 吃掉、validate 通过、日志正常、训练跑完，指标里没有 `sc_correction` / `sao_dis_maskfrac`。**这是最恶劣的失效模式：静默无效。**

**排雷方案（三层）**：

1. **架构层**：§3.5.1 方案 ③ —— 共享纯函数 `slime/utils/pg_objectives.py`，core 与 fork 都调。
2. **断言层**：在 `pg_objectives.apply_sao_dis` / `apply_score_centering` 内部，**首次调用时打一条 INFO 日志并设置一个模块级 flag**；同时在 `train_actor` 结束后检查：若 `args.use_sao_dis` 为 True 但 flag 未设置 → **`raise RuntimeError("--use-sao-dis 已开启但 DIS 从未被调用：检查 --loss-type / --custom-loss-function-path 是否绕过了 DIS 挂载点")`**。这把"静默无效"变成"启动即崩"。
3. **CI 层**：新增 `tests/test_dual_algo_reaches_harbor_fork.py` —— 用 mock args + synthetic batch 直接调 `policy_loss_no_entropy_when_disabled`，断言开关打开时 `reported_loss` 含 `sao_dis_maskfrac` / `sc_correction`。这条测试锁死"fork 必须支持新算法"这一契约。

## 4.2 🔴🔴 重分词与 Token 对齐地雷（本报告最高风险项）

### 现象 1：DIS ratio 在 observation 上的巧合

`atif_adapter.py:832` 对 observation 填 `rollout_log_probs = 0.0`。则
$$
r_t = \exp(\log\pi_\theta(y_t) - 0) = \pi_\theta(y_t) \in (0,1)
$$
对 27B 的 248320 词表，典型 $\pi_\theta \sim 10^{-2}\ldots10^{-6}$。DIS 下界 $1-\epsilon_\ell = 0.2$（coding 配置 $\epsilon_\ell=0.8$）→ **几乎所有 observation token 的 $r_t$ 落在下界外 → 权重归零 → 无梯度**。

这**恰好**是 SAO 想要的（observation 不参与梯度），但：
- 它是巧合，不是设计。若某个 observation token 的 $\pi_\theta > 0.2$（高度可预测的模板 token，如换行、`$` 提示符 —— 在 terminal agent 里**非常常见**），它会**通过**双侧门控并贡献梯度。
- 它还会污染 `sao_dis_maskfrac` 指标（分母含 observation），让监控失真。

**排雷**：DIS 的 ratio 计算**必须先与 `loss_mask` 相乘**，且 metrics 用 `loss_mask` 归约：
```python
w = torch.where((ratio > low) & (ratio < high), ratio, 0.0).detach()
# loss_mask 已在 sum_of_sample_mean 的分母里，但 maskfrac 指标需显式：
# 只统计 loss_mask==1 位置的 mask 率
```
并加**启动期断言**：`assert (rollout_log_probs[loss_mask == 0] == 0.0).all()`，锁死"observation 的 rollout logprob 必为 0.0"这一隐式契约 —— 若将来有人改了 `atif_adapter.py`（比如填真实 logprob），这条断言立刻报出来，而不是让 DIS 静默地把 observation 拉进梯度。

### 现象 2：SC head 的 5 处搬运点

§3.3.2 已列出 5 处 `rollout_log_probs` 写入点。**任一处漏配 head → head 与 token 错位。**

错位的后果**不会报错**：`score_centering_correction` 只要 shape 对得上（`[R,k]` vs `[R]`）就能算，算出来的 $q_v$ 对应的是**另一个 token 的 sampler 分布**。此时 $\rho$ 仍在合理范围、correction 仍是有限值、loss 仍下降 —— 只是在优化一个错误的目标。

**排雷（四层防御）**：

1. **Token-ID 交叉验证（最强）**：head 的第一列**不必**是采样 token（top-1 未必被采样），但采样 token **必须在 head 中或在尾部**。加断言：
   ```python
   # 对 loss_mask==1 的位置，检查 sampler 对采样 token 的 logprob 与 head 的一致性
   in_head = (topk_ids == sampled_ids.unsqueeze(-1))          # [R, k] bool
   hit = in_head.any(-1)                                       # [R]
   # 命中的位置，head 里对应的 logprob 必须 ≈ rollout_log_probs
   gathered = (topk_logps * in_head).sum(-1)[hit]
   assert torch.allclose(gathered, rollout_log_probs[hit], atol=1e-3)
   ```
   **这条断言是唯一能真正抓到错位的手段** —— 它把两条独立的数据通道（`rollout_log_probs` 与 head）交叉对账。命中率在 $k=128$ 下应 > 95%（SC 论文说头部覆盖 99.9% 质量）。建议在 Stage 2/3 全开，Stage 4 降采样到每 rollout 抽 1% 样本（避免 256k×718 样本的断言开销）。

2. **Shape 断言**：每处搬运后 `assert len(topk_ids) == len(rollout_log_probs) == len(loss_mask)`。

3. **分布合法性断言**（PR 已有 `_validate_head_arrays`）：ids 去重非负、logps 有限且 $\le 0$ 且 $\sum e^{\text{logp}} \le 1.0001$。

4. **命中率监控**：把 `sc_head_hit_rate` 与 `sc_head_mass`（$\sum_{v\in H} q_v$）作为 wandb 指标。若 `sc_head_mass` 突然掉到 0.9 以下 → 立刻怀疑错位或量化异常。

## 4.3 🔴 内存与通信吞吐地雷

### 实测算量

| 项 | 计算 | 27B/256k | 122B/84k |
|---|---|---|---|
| 单 token head 字节 | $k\times(4_{\text{int32}}+4_{\text{fp32}})$ | 1024 B | 1024 B |
| 单样本（满 context） | $262144\times1024$ | **0.25 GiB** | 0.082 GiB |
| 一个 rollout batch | 718 样本 × 平均 40k resp tok | **~27 GiB** | 512 × 20k → ~9.8 GiB |

**27 GiB 是主机 RAM，且要过 Ray object store**，路径：
```
SGLang worker → HTTP JSON(base64) → Harbor llm.py → atif_adapter
  → Sample.rollout_topk_* (numpy)
  → RolloutManager._convert_samples_to_train_data
  → _split_train_data_by_dp → ray.put(Box) × dp_size(2)
  → actor._get_rollout_data → ray.get → CP-slice → GPU
```

**四个瓶颈**：
1. **HTTP/base64**：`output_topk_*` 是 base64 编码的二进制（PR 的 patch），膨胀 4/3。一个 40k-token turn 的 head = 40 MB → base64 54 MB 的单次 HTTP response。Harbor `llm.py:127` 用 `httpx.AsyncClient(timeout=600)` 且 **`HARBOR_NUM_CONCURRENT_TRIALS` 在 27B 配置里是几百** → 并发 54 MB response 会打爆 rollout worker 的内存。
2. **Ray object store**：默认 object store 是共享内存（`/dev/shm`）的一部分，27 GiB × dp_size 的副本可能触发 spill to disk，反而更慢。
3. **GPU 侧**：`actor._get_rollout_data` 把 head `.to(device="cpu")`（PR 代码是 cpu），在 loss 里才 `.to(logits.device)`。单 microbatch 的 head 是 `[R_mb, 128]` → 8192 tok/GPU × 128 × 8 B = 8 MB，**GPU 侧无压力**。
4. **`calculate_topk_log_probs` 的 FP32 workspace**：`O(chunk_size × V_p)`。27B: `V_p = 248320/4 = 62080`，`chunk_size = log_probs_chunk_size = 1024` → $1024\times62080\times4$ B = **254 MB** per chunk，×2（forward work + backward work）。可接受，但要注意它是**在已有 18.5 GB vocab-CE 峰值之上**叠加的（27B 脚本 `:214-234` 的实测表）。

**纯内存方案的边界与应对**：

| 措施 | 效果 | 代价 |
|---|---|---|
| `k=32` 而非 128 | 内存 ÷4 → 6.8 GiB | SC 论文 §topk_ablation 实测 $k=32$ 与全词表 SC 持平 → **推荐作为 256k 的默认** |
| fp16 存 logps | 再 ÷1.33 → 5.1 GiB | logp 范围 $[-20,0]$，fp16 精度 $\sim10^{-3}$，对 $q_v=e^{\text{logp}}$ 的相对误差 $\sim10^{-3}$ → 可接受但需单测 |
| int32→int16 存 ids | 不可行 | 词表 248320 > 65535 |
| 只对 `loss_mask==1` 的位置存 head | 按 `ppo_config_vs_sao_paper.csv` 的 agentic 轨迹，observation 占比可能 > 50% → **内存减半以上** | 需要 ragged 存储（类似 `rollout_top_p_token_offsets` 的 offsets 方案），实现复杂度上升 |
| disk spill（PR 的 `DiskTensorRef`） | 内存 → 0 | 需移植整个 `tensor_store.py`，冲突报告 C-5 方案 B |

**建议**：SC 在 256k 上的默认配置为 **`--score-centering-top-k 32` + 只存 action token 的 ragged head**。前者零实现成本，后者放到 Stage 4（若 Stage 3 实测内存确实撑不住）。**并在 `validate_dual_algo_args` 里加一条硬警告**：
```
if use_score_centering and seq_length > 131072 and score_centering_top_k > 32:
    logger.warning("SC at >128k context with k>32: estimated host RAM %.1f GiB/rollout. "
                   "Consider --score-centering-top-k 32.", est)
```
（警告而非报错 —— 是用户的决定。）

> **SAO 侧完全无此问题**：DIS 只用 `rollout_log_probs`（`[R]` fp32，4 B/token），基线**已经在传**。增量为零。

## 4.4 🟡 SGLang Patch 与生产环境不一致

**已实测确认的事实**：
```
运行时 sglang 0.5.12.post1 @ /sgl-workspace/sglang/python/sglang
  output_topk_token_ids  : False   ← sglang-top_p.patch 未应用
  top_p_token_ids        : False   ← 同上
  routed_experts         : True    ← sglang.patch 已应用（R3 依赖）
```

**两个独立结论**：

**(a) 对零回归是好消息。** PR 对 `sglang-top_p.patch` 的改动无条件改写了 `output_top_logprobs` 的返回语义（`if not return_text_in_logprobs:` 分支不再填该字段）。因为 patch 未应用，此语义变更**当前不可能生效**。且 Harbor `llm.py:119-124` 既不请求 `top_logprobs_num` 也不读 `output_top_logprobs`（只读 `output_token_logprobs`，`:150`）—— **双重安全**。

**(b) 对 SC 落地是硬阻断。** SC 需要 `meta_info["output_topk_{token_ids,log_probs,shape}"]`，只能来自这个 patch。

**部署策略（分三档，按风险递增）**：

| 档 | 做法 | 风险 | 适用 |
|---|---|---|---|
| **档 0（推荐先做）** | 不重建镜像。SC 走"**合成 head**"影子模式：由 trainer 自己的 logits 取 top-$k$ 当作 sampler head（即假设 $q=p$）。此时 $\rho=1$、`residual = p - p = 0`、correction 恒为 0 | 零 | **Stage 3 影子验证**：验证整条数据通道（搬运/对齐/CP/内存）正确，且 SC correction 恒 0 → loss 与基线逐 bit 相同。这是**唯一能在不动 SGLang 的前提下端到端验证 SC 管道的办法**，价值极高 |
| 档 1 | 只请求 `top_logprobs_num=k`，用**未打 patch 的** `output_top_logprobs`（detokenized 的 `[(logp, tok_id, text)]` 列表）解析 | 中：JSON 体积比 base64 大 5-10×，且 `return_text_in_logprobs` 默认行为需确认 | Stage 3 真实 head 的小批量验证 |
| 档 2 | 重建镜像应用 `sglang-top_p.patch` | **高**：语义变更 + 需全集群滚动替换 + 与 R3 的 `sglang.patch` 叠加冲突需验证 | Stage 4 正式训练 |

**档 2 的硬前置条件**（写进合并说明）：
1. `grep -rn 'output_top_logprobs' slime/ slime_plugins/ harbor/` 必须为空（确认无代码依赖该字段）；
2. 新增 `tests/test_sglang_top_logprobs_contract.py` 锁死契约；
3. 先在 1 个节点 canary，比对同一 prompt 在打/未打 patch 下 `output_token_logprobs` 逐 bit 一致（patch 只应加字段，不应改 sampled logprob）。

## 4.5 🔴 Multi-Turn Observation 梯度污染

分四个独立的污染面，**每个都要单独堵**：

### (1) Loss 面 —— 已安全
`loss_mask=0` 的位置在 `get_sum_of_sample_mean`（`cp_utils.py:53`）的分子与分母里都被剔除。基线已正确。✅

### (2) Advantage 面 —— 基线有污染，Skip-Obs 修复
基线 GAE 在 observation 位置产生 $\delta_{o}=r_o+\gamma V(o_{+1})-V(o)$。因为 $r_o=0$（`kl_coef=0`），$\delta_o=\gamma V(o_{+1})-V(o)$ —— **critic 在 observation 上的预测误差直接进入 `lastgaelam` 并向前传播到所有更早的动作 token**。

**$\gamma=\lambda=1$ 下望远镜抵消的严格边界**：
$$
\hat A_t\big|_{\gamma=\lambda=1}=\sum_{l\ge t} r_l - V_t
$$
所有中间 $V$ 精确抵消 → **observation 的 $V$ 不污染**。✅

**但**：`length_adaptive_gae` 把 $\lambda_{\text{policy}} = 1-1/(\alpha L)$。L=5000 → $\lambda=0.99987$。残留污染项量级 $\sim (1-\lambda)\cdot\text{Var}[V(o)] \approx 1.3\times10^{-4}\cdot\text{Var}[V]$ —— 小但非零，且**随 observation 数量线性累积**（terminal agent 一条轨迹可有 60 turn）。

→ **结论**：Skip-Obs GAE 的收益在 $\lambda\to1$ 时趋于零，在 `length_adaptive_gae` 下是**小而真实**的。这解释了为什么 `ppo_config_vs_sao_paper.csv` 把它列为 `KEY DIVERGENCE` 但基线仍能稳定训练。**必须在 Stage 2 用 synthetic tensor 量化这个差值**，而不是凭直觉决定要不要上。

### (3) Critic 目标面 —— 基线有污染，需决策
`value_loss_function` 用 `batch["returns"]` 回归。基线的 `returns` 在 observation 位置也有值，但 `loss_mask=0` → **不参与 critic loss**。✅ 已安全。

**但 Skip-Obs 下 `returns` 在 observation 位置被我填 0**（§3.4 步骤 5）。因为 `loss_mask=0`，不影响 critic loss。✅ 一致。

**隐患**：`compute_value_explained_variance`（`loss.py:574`）会用到 `returns` —— 需确认它也按 `loss_mask` 归约。这是 Stage 2 的一条检查项。

### (4) THD 打包 / CP 切分面 —— 需断言
基线 `qkv_format=thd`，所有样本的 token 拼成一条 `[1, T, V]`。CP=4 zigzag 把它切成 8 个 chunk，每 rank 持 2 个。

**风险**：Skip-Obs 的动作掩码 `full_mask` 必须与 `full_values` / `full_rewards` 在**同一个 all-gather 后的完整 response 视图**里对齐。`length_adaptive_gae` 走的是 `get_advantages_and_returns`（per-sample，内部自己 all-gather + re-slice）—— 我们沿用同一路径，但 **`loss_masks` 在 `rollout_data` 里是已经 CP-sliced 的局部视图**（`actor.py:236-238` 直接 `torch.tensor(t)`，t 来自 `rollout_data["loss_masks"]`，而 `process_rollout_data` 不做 CP slice）。

**这里必须查清**：`rollout_data["loss_masks"]` 是**完整 response 长度**还是**CP 局部长度**？

从 `loss.py:811-824`（`normalize_advantages` 分支）可见：
```python
full_mask = loss_masks[i]
local_mask_parts.append(full_mask[res_s0:res_e0])   # ← 对 loss_masks[i] 做 CP 切分
```
**→ `rollout_data["loss_masks"][i]` 是完整 response 长度**（`[R_i_full]`），核心代码在需要局部视图时**自己切**。而 `values`/`advantages` 是 CP 局部的。

**这是一个必须精确处理的不对称**：
- `values[i]`：CP 局部 `[R_i_local]`
- `loss_masks[i]`：**完整** `[R_i_full]`

`get_advantages_and_returns` 内部对 `values` 做 `all_gather_with_cp` 得到 full，此时才能与 `loss_masks[i]` 对齐。→ **Skip-Obs 实现必须在 all-gather 之后才用 `loss_masks`，且不得对 `loss_masks` 再做 all-gather**（会得到 4× 长度）。

**断言**：`assert full_values.shape == loss_masks[i].shape`，放在 all-gather 之后、压缩之前。这一条断言能抓住所有 CP 对齐错误。

---

# 第 5 章 分阶段落地路线图与验收单

## Stage 1 — 接口定义与零回归锁定（不含任何算法逻辑）

**产出**：13 个参数定义 + `_any_new_algo_enabled` gate + `validate_dual_algo_args` 骨架（只做参数校验，不实现算法）+ `_pg_loss_type_or_ppo`。

**自动化验收**：
| 手段 | 具体命令 / 断言 |
|---|---|
| **Dry-run 参数快照** | 改动前后各跑 `DRY_RUN=1 ./run_27b_dense_128gpu_ppo_256k.sh` 与 122B 脚本，dump 完整 resolved TRAIN_CMD → `diff` 必须为空 |
| **`vars(args)` 快照** | 用 `--dump-args-json /tmp/args_{before,after}.json`（或在 `parse_args` 后 monkeypatch 落盘）→ `diff` 只允许新增 13 个 key 且值等于设计默认值 |
| **AST 对比** | `python -m ast` dump `train_loss.py` / `loss.py` / `train_async.py` 改动前后，断言"新增节点全部位于 `if <gate>` 的 body 内"。脚本：`tests/tools/assert_gated_ast.py`（新建） |
| **字节码对比** | `dis.dis()` 基线路径的 3 个热函数（`policy_loss_no_entropy_when_disabled`、`compute_advantages_and_returns`、`train_async.train`）。允许差异：新增的 gate 判断指令；**不允许**：原有指令序列的顺序或操作数变化 |
| 单测 | `pytest tests/test_dual_algo_args.py` —— 参数默认值、§2.3 全矩阵的合法/非法组合各一例 |

**Stage 1 通过标准**：上述 5 项全绿。**此时可以安全地把代码合进主干并继续跑基线训练**（因为没有任何算法逻辑）。

## Stage 2 — 离线单测与数学验算（纯 CPU，无 GPU，无集群）

**产出**：`slime/utils/pg_objectives.py`、`slime_plugins/harbor/sao_gae.py`、`slime/backends/megatron_utils/sc_terms.py`、SC 的 `score_centering.py` 移植（去 `DiskTensorRef`）+ 缺失的 vocab-parallel helper。

**自动化验收**：
| # | 测试 | 断言 |
|---|---|---|
| 1 | `test_sao_dis_math.py` | 构造已知 ratio 的 synthetic tensor，逐元素比对 $f(x;\epsilon_\ell,\epsilon_h)$ 的三段（界内=x、下界外=0、上界外=0），含边界点 $x=1-\epsilon_\ell$ / $1+\epsilon_h$ 的开闭性 |
| 2 | `test_sao_dis_identity.py` | `use_sao_dis=False` 时 `apply_sao_dis` 返回的张量与输入 **`is` 同一对象**（不只是数值相等）→ 证明零开销 |
| 3 | `test_skip_obs_gae_gamma1.py` | **$\gamma=\lambda=1$ 时 Skip-Obs GAE 与基线 GAE 在 action 位置必须逐元素相等**（望远镜抵消，§4.5(2) 的数学结论）。这是一条**极强的正确性锚点** |
| 4 | `test_skip_obs_gae_lambda_lt1.py` | $\lambda=0.99$ 时，手工展开 3-turn 玩具轨迹（action 长度 2/3/2，observation 长度 4/4），逐项验证递推跳过 observation |
| 5 | `test_skip_obs_reward_aggregation.py` | `kl_coef != 0` 时，observation 位置的 reward 必须被聚合到**前一个** action token（不能丢，不能重复计入） |
| 6 | `test_sc_correction_zero_when_q_eq_p.py` | $q=p$ 时 `score_centering_correction` 返回 **0**（SC 论文："In the absence of TIM, score centering is a no-op"）。这是档 0 影子模式的理论基础 |
| 7 | `test_sc_topk_identity.py` | 小词表（$V=8$，$k=4$）下，比对式 12 的头部形式 $\sum_H(q_v-\rho p_v)s_v$ 与直接全词表 $\sum_V \hat q_v s_v$ 的数值一致性（应到 fp64 精度） |
| 8 | `test_sc_is_composition.py` | 式 14 的 $\alpha=\rho f(1/\rho)$：(a) $f=1$ → $\alpha=\rho$；(b) $f(r)=r$ → $\alpha=1$ 且 correction **恒为 0**（论文的 sanity check）；(c) DIS 的 $\epsilon\to$ ratio 边界转换 $low=1-\epsilon_\ell$ 的 $\pm1$ 正确性（§2.3 专论标出的易错点） |
| 9 | `test_topk_logprobs_autograd.py` | `calculate_topk_log_probs` 的 `gradcheck`（fp64、小 shape）+ chunk_size ∈ {−1, 1, 3, 大于 N} 的结果一致性 |
| 10 | `test_dual_algo_reaches_harbor_fork.py` | §4.1 层 3 —— 直接调 `policy_loss_no_entropy_when_disabled`，断言开关打开时 metrics 含新 key |
| 11 | `test_head_token_crossvalidation.py` | §4.2 防御 1 的交叉对账逻辑本身的单测（构造故意错位一行的 head，断言被抓到） |
| 12 | 全量回归 | `python -m pytest tests -m unit` 全绿；重跑 Stage 1 的 5 项快照 |

**Stage 2 通过标准**：12 项全绿。**测试 3 和 6 是两个不可妥协的数学锚点** —— 前者证明 Skip-Obs 在基线超参下是恒等变换，后者证明 SC 在无 TIM 时是 no-op。任一失败说明实现有根本错误。

## Stage 3 — 影子旁路验证（真实集群，不改变梯度）

**核心思想**：让新算法**计算但不生效**，比对它与基线的数值关系。

| 子阶段 | 配置 | 验证目标 | 通过标准 |
|---|---|---|---|
| **3a** SMOKE 零回归 | `SMOKE=1`，不传新参数 | 端到端确认合并后基线不变 | 与合并前 SMOKE run 的 `train/pg_loss`、`train/ppo_kl`、`train/advantages` **逐 step 逐 bit 相同**（需固定 seed 且关闭 dynamic batch 的非确定性；若无法逐 bit，则要求相对误差 < 1e-6） |
| **3b** Skip-Obs 影子 | `--use-skip-obs-gae` + $\gamma=\lambda=1$（基线超参，不挂 `length_adaptive_gae`） | 验证 §4.5(2) 的望远镜抵消在**真实 CP=4 / THD / 256k** 下成立 | `advantages` 在 `loss_mask==1` 位置与基线**数值相同**（这是测试 3 的集群版）。同时记录 `skip_obs_masked_action_token_ratio`（§3.3.1），要求 < 1% |
| **3c** DIS 影子 | `--use-sao-dis --sao-dis-eps-low 1e9 --sao-dis-eps-high 1e9`（门限放到无穷宽 → $f(x)=x$） | 验证 DIS 的数据通道与 ratio 计算 | 此时 `pg_loss *= ratio`，**不等于基线** → 改为验证 `sao_dis_maskfrac == 0`、`sao_dis_ratio` 的分布（中位数应 ≈ 1，观察长尾），并确认 `assert (rollout_log_probs[loss_mask==0]==0).all()` 不触发（§4.2 现象 1） |
| **3d** SC 档 0 影子 | `--use-score-centering` + **合成 head**（trainer 自己的 top-$k$ 当 sampler head，§4.4 档 0） | 验证 SC 的**全部管道**：搬运对齐、TP vocab-parallel、CP zigzag、内存、autograd | `sc_correction` 的绝对值 < 1e-5（理论应为 0，测试 6 的集群版）；`loss` 与基线相对误差 < 1e-6；实测 `sc_head_hit_rate`（应 100%，因 $q=p$）；**实测主机 RAM 峰值**（对照 §4.3 的 27 GiB 估算） |
| **3e** TTUR 计时 | `--critic-updates-per-actor-update 2`，1 个 rollout | 量化 §3.5.3 的 offload/NCCL 重建成本 | 记录 `train_wait` / `data_preprocess` / critic `train` 三段计时，给出 $K=2$ 的 wall-clock 倍率。若 > 2.5× 则 TTUR 在本集群不经济，需回报 |
| **3f** SC 档 1 真实 head | 小批量（`SMOKE=1`）+ 请求真实 `top_logprobs_num` | 验证 §4.2 防御 1 的交叉对账在真实数据上通过 | `sc_head_hit_rate` > 95%；`sc_head_mass` > 0.99；交叉对账断言零触发 |

**Stage 3 通过标准**：3a 是**门禁**（不过则回滚）。3b/3d 的数值恒等是正确性证明。3c/3e/3f 产出的是决策数据（要不要上、成本多少）。

## Stage 4 — 真实训练切换

**建议顺序（按风险递增，每步一个独立 run）**：

| 步 | 配置 | 理由 | 观测重点 |
|---|---|---|---|
| **4.1** | Skip-Obs + `length_adaptive_gae`（$\lambda<1$） | 风险最低：无新数据、无 SGLang 依赖、Stage 3b 已证明 $\lambda=1$ 下恒等 | `advantages` 分布 vs 基线；`explained_variance`（注意 `length_adaptive_gae` docstring `:36-40` 已警告该指标含义改变） |
| **4.2** | + DIS（`--pg-loss-type reinforce`，$\epsilon_\ell=0.8,\epsilon_h=3.0$） | 目标从 clipped surrogate 换成 REINFORCE-with-weight，是**实质算法变更** | `sao_dis_maskfrac`（论文 Fig 5c 说应维持"dynamic clip ratio"而非近零）；`entropy_loss`（REINFORCE 无 clip 保护，熵崩塌风险更高）；`train_rollout_logprob_abs_diff` |
| **4.3** | + TTUR $K=2$ | 依赖 4.2 稳定 + Stage 3e 的成本可接受 | critic `explained_variance` 应改善；wall-clock/step |
| **4.4** | + Frozen-Attention critic | 纯配置（`--freeze-params-name-list`），零代码 | critic `grad_norm`（论文的动机就是 attention 层 grad norm 过大） |
| **4.5** | SC（需先完成 §4.4 档 2 的镜像重建） | 最高风险：新 SGLang patch + 新数据通道 + 27 GiB 内存 | `sc_correction` 量级、`sc_head_mass`、主机 RAM；建议先在 122B/84k 上试（内存压力 1/3） |
| **4.6** | ASC-PPO（SC + 完整 SAO） | 全部前置通过后 | 全指标 |

**每步的共同验收**：与基线 run 同 seed、同数据池、同 step 数的 A/B；`train/reward`、`eval/*`、`entropy_loss`、`grad_norm` 四条曲线；任一 step 出现 NaN / grad_norm 突增 10× → 立即停并回退。

---

# 第 6 章 需要你决策的 6 个点

| # | 决策 | 选项 | 我的建议 |
|---|---|---|---|
| 1 | **Core/fork 解耦策略** | ①只改 core ②双份实现 ③共享纯函数 | **③**。①会让新算法在基线上静默无效（§4.1），②必然漂移 |
| 2 | **SC 内存策略** | 纯内存 $k=128$（27 GiB）/ 纯内存 $k=32$（6.8 GiB）/ 移植 disk spill | **纯内存 $k=32$**。SC 论文实测 $k=32$ 与全词表持平，零实现成本换 4× 内存 |
| 3 | **Skip-Obs 的 $\lambda$ 基数** | token 长度 $R$ / 动作 token 数 $N_{\text{act}}$ | **$N_{\text{act}}$**（SAO appendix 的立场），但列为 Stage 4.1 的 A/B 项 |
| 4 | **TTUR 实现路径** | A（外层循环，target 重算）/ B（内层循环，target 冻结） | **A**。B 不是 TTUR 而是数据重用，且违反 `ASC_PPO_derivation.md` §6.3 |
| 5 | **SGLang patch 部署** | 档 0 影子 / 档 1 未打 patch 的 JSON / 档 2 重建镜像 | **先档 0（Stage 3d）→ 档 1（3f）→ 档 2（4.5 前）**。档 0 能在零风险下验证整条 SC 管道 |
| 6 | **参数命名** | 对齐 PR（`--use-score-centering`/`--pg-loss-type`）+ 自定义 SAO 参数 / 统一 `--algorithm-type` 枚举 | **正交开关**（§2.1 四条理由）。`--algorithm-preset` 仅作只读诊断 |

---

## 附录 A：改动文件总清单（预估）

| 文件 | 类型 | 预估行数 | Stage |
|---|---|---|---|
| `slime/utils/arguments.py` | 改 | +80 / −2 | 1 |
| `slime/utils/dual_algo_validate.py` | **新** | ~180 | 1 |
| `slime/utils/pg_objectives.py` | **新** | ~120 | 2 |
| `slime/utils/score_centering.py` | **新**（PR 移植，去 DiskTensorRef） | ~170 | 2 |
| `slime/utils/ppo_utils.py` | 改（+`get_pg_loss_type`/`importance_weights`/`calculate_topk_log_probs`/2 helper） | +150 | 2 |
| `slime/backends/megatron_utils/sc_terms.py` | **新** | ~110 | 2 |
| `slime_plugins/harbor/sao_gae.py` | **新** | ~200 | 2 |
| `slime_plugins/harbor/train_loss.py` | 改（3 处挂载） | +12 | 2 |
| `slime/backends/megatron_utils/loss.py` | 改（import + 2 处调用 + custom adv 已有） | +10 | 2 |
| `slime/utils/types.py` | 改（2 个 Sample 字段） | +4 | 3 |
| `slime/ray/rollout.py` | 改（SC gate + DP key） | +14 | 3 |
| `slime/backends/megatron_utils/actor.py` | 改（head CP-slice） | +12 | 3 |
| `slime/backends/megatron_utils/model.py` | 改（forward_step keys） | +2 | 3 |
| `slime/backends/megatron_utils/data.py` | 改（log skip list） | +2 | 3 |
| `slime/utils/http_utils.py` | 改（条件 kwarg） | +30 | 3 |
| `slime_plugins/harbor/llm.py` | 改（SC 请求 + 解析） | +35 | 3 |
| `slime_plugins/harbor/atif_adapter.py` | 改（**5 处 head 搬运 + 断言**） | +90 | 3 |
| `train_async.py` | 改（TTUR if/else） | +8 | 3 |
| `docker/patch/latest/sglang-top_p.patch` | 改（PR 移植 + 注释） | +60 | 4 |
| `tests/test_*`（12 个新测试） | **新** | ~1100 | 2 |
| `tests/tools/assert_gated_ast.py` | **新** | ~80 | 1 |
| `examples/harbor_terminal_rl/sao_critic_frozen_attn.yaml` | **新** | ~10 | 4 |

**合计**：新建 10 文件（~2250 行，其中测试 1180 行），修改 12 文件（+521/−2）。**SAO 路径（Stage 1+2 的 SAO 部分 + 4.1–4.4）不依赖任何 SGLang 改动，可独立完成。**

---

**报告结束。等待人工审核。未修改任何现有代码，未创建 commit，未 apply patch。**
