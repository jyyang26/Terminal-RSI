# TB4 Run #3 改动说明：Dense Reward 终局方案、TB-GAE、max_turns

初版日期：2026-09-23；本次工程复核与启用设计更新：**2026-09-27**。

> **后续实施状态（2026-09-27）：**第 9–10 节定义的首期 TB-GAE 已完成代码接入，
> 201/201 离线测试通过（TB 专项 117 + 既有回归 84）。实际文件、测试结果、启用范例与支持边界
> 见 `TB_GAE_IMPLEMENTATION_REPORT_20260927.md`。以下“当前缺口/待实现/只修改文档”描述保留为
> 前一轮审查时的历史记录，不代表实施后的代码状态。完整生产上线仍需可信真实终止输入；
> 多 chunk/外部截断 bootstrap/BSHD 按首期协议显式拒绝，未宣称已支持。

> **本次范围：只更新本文档。** 第 0–7 节保留历史实验背景，其中“已改/实测”指此前工作，
> 不代表本次修改了业务代码。当前目标是 `launch_27b_fresh_base_critic_warmup2.sh`；
> 最新源码核查见第 8 节，**默认关闭、仅显式 True 开启的规范见第 9 节，后续改造清单见第 10 节**。
> 目前不能仅靠参数直接启用有效 S3/S4；所有新增参数/YAML 示例均为待实现设计。

初版作者备注：本文记录**此前一轮实际改了什么**、**下次启动可以改什么**、以及**哪些结论尚未被验证**。
初版数据来自对磁盘产物的只读扫描；本次另补 CPU 复现，均未执行训练。

---

## 0. TL;DR

| # | 改动 | 文件 | 状态 |
|---|---|---|---|
| 1 | `max_turns` 60 → **500** | `examples/harbor_terminal_rl/harbor_mixed3000_..._run2_ctrf82.yaml` | **已改，下次启动生效** |
| 2 | 新增终局 Dense Reward hook | `slime_plugins/harbor/outcome_dense_reward.py`（新文件） | **已写，需显式切换才生效** |
| 3 | 新增 TB-GAE 实现 + 不变量自测 | `slime_plugins/harbor/tb_gae.py`（新文件） | **已写并通过自测，但缺管线无法启用** |
| 4 | Run #2 诊断与方案推导 | `TB4_27B_hyperparameter.html` 第 10 节 | 已写入 |

以上状态记录的是初版工作；本次仅修改 Markdown，未改变任何训练任务的代码或配置。
不能从历史 run #2 的记录推断当前 fresh_base 任务参数；后者见第 8.1 节。

---

## 1. 最重要的诊断结论：轮次「持续下降」在数据上不成立

原始假设是 `harbor/avg_turns_per_trajectory` 持续下滑、源于短程捷径坍缩。
我取了 run #2 全部 **23 个 rollout** 做趋势检验，**结论相反**：

```
取值范围        46.8 – 55.2      首 46.8 → 末 53.7（末值更高）
线性斜率        +0.013 轮/步      23 步累计 +0.28 轮
Pearson r       +0.049           r² = 0.002
标准差 / CV     1.69 / 3.2%
去掉 step0 后    −0.057 轮/步      22 步累计 −1.19 轮（仍在噪声内）
最后 8 步        +0.188 轮/步      方向向上
```

同时 `harbor/reward_mean` **在上升**（r = **+0.49**，0.096 → 0.147，+31%）。

### 真正的机制：指标被 `max_turns=60` 右删失，且删失在加重

直接读 `rollout_dumps` 的逐轨迹 `harbor_num_turns`：

| dump | 轨迹 | `turns == 60` | 占比 | 均值 | 中位 | solve rate |
|---|---|---|---|---|---|---|
| `0.pt` | 256 | 119 | 46.5% | 46.8 | 57 | 16.4% |
| `10.pt` | 256 | 193 | 75.4% | 51.6 | **60** | 21.9% |
| `20.pt` | 256 | 196 | **76.6%** | 52.2 | **60** | 23.8% |

- **76.6% 的轨迹恰好停在 60 轮**，中位数就是 60 —— 均值 ~52 是「76% 顶在天花板 + 24% 早停」的混合量，**不是行为选择**。
- 撞顶轨迹**失败率 84%**：它们不是做完了，是被切断了。
- 删失从 46.5% 涨到 76.6%，而 solve rate 同时上升 —— 模型在把**更多**任务推到上限，不是浅尝辄止。

评测侧同一结论（terminal-bench-2，89 题，`eval_19.pt`）：
**成功题 turns 中位 10，失败题中位 60**（= 上限本身）；`turns == 60` 的 61 个样本**全部失败**。

> **注意因果方向**：>31 轮失败率 90.6%，但被上限切断的任务必然计入失败。
> **删失与「盲目重试 churning」在当前数据里无法区分** —— 要区分必须先放开上限。

---

## 2. 改动 1：`max_turns` 60 → 500（已生效）

**文件**：`examples/harbor_terminal_rl/harbor_mixed3000_128gpu_27b_dense_daytona_ppo_cp4_256k_run2_ctrf82.yaml`
**位置**：`harbor_trial_defaults.agent.kwargs.max_turns`

```yaml
      # >>CHANGED 60 -> 500  （完整理由见 YAML 内注释）
      max_turns: 500
```

- **目的是解除删失**，不是让模型跑 500 轮。
- 真正先起作用的约束仍在：agent `override_timeout_sec = 3600`、`proactive_summarization_threshold = 229376`。500 是上界不是目标。
- **风险**：`perf/rollout_time` 可能上升（当前 ~2200 s/步）。

**上线后必须盯的两个指标**：
1. `harbor/avg_turns_per_trajectory` —— 应当**散开**、不再堆在单一值上。
2. `perf/rollout_time` —— 若翻倍，退到 **120–200** 再评估。

已用 `yaml.safe_load` 验证该文件仍可正常解析。

---

## 3. 改动 2：终局 Dense Reward（新文件，需显式切换）

**文件**：`slime_plugins/harbor/outcome_dense_reward.py`
**入口**：`outcome_dense_rewards(args, samples, **kwargs)`

### 为什么是新文件而不是改 `partial_reward.py`

`partial_reward.py` 是正在跑的 run #2 的奖励路径。原地改会在 worker 重新 import 的瞬间
改变在途 run 的奖励。放在新文件里，**切换和回滚都只是一行配置**。

### 四支柱

```
支柱 1 成功优先   B(τ) = Y + α(1−Y)·q_H,  α = 0.30
支柱 2 加权       W(c) = ρ(type)·ι(c)·δ^depth
                    ρ: leaf 1.0 | list_elem 0.6 | keys 0.25 | length 0.25
                    ι(c) = max(0.10, H₂(p_c))   恒过项→0.10，未见项→1.0
                    δ = 0.90
                  q_H = Σ_{passed} W / Σ_{all} W
支柱 3 反作弊     通过项不含任何 leaf → q_H := 0
支柱 4 有界成本   R = clip(B − min(0.10, T·0.001), 0, 1)
```

### 权重取值的实测依据（run #1，2087 trial / 82,302 检查项）

| 类型 | 项数 | 占全部 | 通过率 | 占通过项 | ρ |
|---|---|---|---|---|---|
| leaf | 26,480 | 32.2% | 91.3% | 31.3% | 1.00 |
| list_elem | 33,931 | 41.2% | 93.3% | 41.0% | 0.60 |
| keys | 13,487 | 16.4% | **98.7%** | 17.2% | **0.25** |
| length | 8,404 | 10.2% | **96.4%** | 10.5% | **0.25** |

纯结构项占通过项 **27.7%**、通过率 **97.8%** —— 约 28% 的「成绩」几乎无条件获得。
深度分布峰值在 **depth 6（22.7%）**，来自嵌套数组展开，深≠难，故 δ=0.90（depth6 权重降到 0.531）。

### 在真实数据上的验证结果（随机 1200 trial，其中 226 有逐项 tests）

```
成功轨迹 R（52 轮）        恒 = 0.948
失败轨迹 R 最大值          0.248
成功 − 失败 断层           0.700      <- 与 1−α 严格一致
加权 q_H vs 未加权         0.691 vs 0.732（压低 5.6%，114 个部分通过样本）
反作弊闸门触发率           3.5%
```

**失败轨迹无论刷多少分子项都不可能超过成功轨迹**，数值倒挂被封堵。

### 怎么启用

```bash
export CUSTOM_REWARD_POST_PROCESS_PATH=slime_plugins.harbor.outcome_dense_reward.outcome_dense_rewards
```

或在 YAML 里加（**推荐**，因为 env 变量会被 `ray job submit` 路径丢掉）：

```yaml
harbor_outcome_alpha: 0.30      # 失败上界
harbor_depth_decay: 0.90        # δ
harbor_step_cost: 0.001
harbor_cost_cap: 0.10
harbor_iota_floor: 0.10
harbor_l1_partial: 0.05         # 仅缺 reconstruct.py 时的渐进补偿
# 可选：历史通过率表，用于 ι(c)。缺失时所有检查 ι=1.0，方案依然可用
# harbor_check_passrate_table: /path/to/passrate.json
```

### 切换时**必须同步重标**的三个参数

新奖励落在 **[0, 1]**，而当前配置是按 per-family 的 **1.00–3.00** 尺度配的：

| 参数 | 当前 | 建议 | 原因 |
|---|---|---|---|
| `value_clip` | 0.2（wandb 实测值） | **0.05–0.1** | 新尺度只有原来的 1/3，0.2 会变成过松的约束 |
| critic `clip_grad` | 见 `megatron_roles_*.yaml` | 重测 `critic-grad_norm` 后再定 | 目标尺度变小，梯度范数会同步变小 |
| `num_critic_only_steps` | 4 | **保留 ≥4** | 回归目标换了尺度，value head 需要重新对齐 |

---

## 4. 改动 3：TB-GAE（新文件，通过自测但**暂时无法启用**）

**文件**：`slime_plugins/harbor/tb_gae.py`

### 为什么需要它

本次源码与启动日志确认目标任务当前是**原生 token-level GAE**（`get_advantages_and_returns_batch`
的签名里**没有** `loss_masks`）；历史数据记录中 **62.9%** 的 response token 是 observation，
这一比例未在本次 fresh_base 数据上重新统计。

### 不变量测试已实际运行并通过

```
$ python3 slime_plugins/harbor/tb_gae.py
[1] all-action vs standard GAE: max|dadv|=0.00e+00 max|dret|=0.00e+00
[2] telescoping with observations skipped: max|returns - R|=1.43e-06
[3] observation positions zeroed: max|adv|=0.00e+00 max|ret|=0.00e+00
[4] lambda_turn=0.95:  50 轮后终局权重 0.0769   <- 禁用
    lambda_turn=0.99:  0.6050
    lambda_turn=0.999: 0.9512                   <- 推荐
    lambda_turn=1.0:   1.0000                   <- 负对照
PASS
```

测 [1] 只证明 **all-action、单卡、特定固定输入**的等价；含 observation 时应在相同奖励与终止语义下
比较有效 action 位。它不覆盖入口 wiring、CP、真实 chunk/截断，也不等于生产零回归已验收。详见第 9.7 节。

上面历史输出的第 [4] 项只展示 `lambda_turn**50`；S4 同时设置 gamma 与 lambda，
不能把该数字直接当作 S4 的完整终局奖励系数，修正见第 9.5 节。

### 为什么现在还不能开

当前 adapter 不生成 `turn_end_mask/is_real_done`，convert 不提升、DP 白名单不传递；
此外还有 critic hook 被清空、标量奖励丢失、CP wrapper 未接入、full mask/local value 形状不匹配、
CP scatter 错位、hard cap 未同步字段，以及真实终止/跨 chunk bootstrap 缺口。

**不能仅设置 custom advantage hook 启用。** 第 8 节逐项给出当前源码证据；
第 9 节规定统一的 `enable_tb_gae`，仅 True 才建立数据和优势估计路径，
False 恒保留当前原生 GAE。后续代码清单以第 10 节为准。

### CP=4 接入现状

现有 wrapper 虽调用 slime 的 gather helper，但自行计算 scatter 时用了错误的 offset，
入口还没有调用 wrapper。**当前不具备 CP=4 正确性保证**；
修复需统一 response/logit 坐标，并把 THD/BSHD 的支持边界明确化，见第 8.5 节。

---

## 5. 下次启动的建议顺序（不要一次全开）

| 阶段 | 开什么 | 判据 |
|---|---|---|
| **S1** | 只开 `max_turns=500`，奖励不变 | `avg_turns` 分布是否散开；`perf/rollout_time` 涨幅可接受 |
| **S2** | 切 `outcome_dense_rewards` + 重标 `value_clip` + `num_critic_only_steps≥4` | `rollout/rewards` 落在 [0,1]；EV 是否保持 ≥0（run #2 已爬到 +0.013） |
| **S3** | 完成门控与管线，显式 True，轮次 γ=λ=1.0 | 同一输入、奖励与终止语义下，action 位 targets 与对应原生基线比较（第 9.7 节） |
| **S4** | 保持显式 True，仅轮次 γ=λ 改到 0.999 | 与 S3 比较分桶 advantage、returns、任务效果与开销；不预设优势一定增大 |

**一次只改一件事**。S2 同时改了尺度和公式，所以 S1 的 `max_turns` 效果必须先单独测出来，
否则两者的贡献无法归因。

> **2026-09-27 复核：参数能透传，但代码尚不支持直接运行 S3/S4。**
> 第 8 节给出调用链缺口，第 9–10 节给出本次要求的可选开关设计和落地顺序。

---

## 6. 尚未验证 / 本方案**没有**解决的问题

这一节是刻意写的，避免把未证实的数字当成结论用。

1. **L2 首错 `break` 仍在**。新奖励读的还是同一份 `ctrf.json`，分母依然随候选存活深度漂移
   （实测：**45.7%** 的 trial 只暴露 1 个场景，**20.2%** 走完 3 个）。
   修这个要改 verifier 侧 `judge.py`，属独立工程。

2. **数值容差穿透（`abs_tol=1.0, rel_tol=0.25`）未修，且我没有独立复核**。
   「19.7% 的数值检查填 0 即过」这个数字需要对每个 leaf 重跑 oracle 比较才能确认，
   本轮没做，所以**文档里没有把它当已证实的数字使用**。
   反作弊闸门只能保证「至少命中一个 leaf」，**不能保证那个 leaf 不是被容差放过的 0**。

3. **判别力贫血（4.2% 判别力 / 97.9% 常数偏移）未复核**。该统计要按 (task, check) 跨多次采样配对聚合，
   扫描到交付时仍未跑完，**文档中没有引用这两个数字作为依据**。
   ι(c) 的熵权重设计**不依赖**它们成立（它在任何 pass-rate 分布上都做正确的事），
   但**收益大小**要等该统计出来才能量化。

4. **「短程捷径坍缩」假说既未证实也未否证**。当前删失掩盖了轮次信号，
   必须先放开 `max_turns`、让分布散开，才能重新检验「模型是否偏好小任务」。

5. **`eval` 走的仍是 terminal-bench-2（89 题），不是 TB-science**。
   若关心 TB-science 涨点，需要改 `harbor_eval_tasks_dir`；参见 `TB_science.html`。

6. **以下是历史 run #2 的观察，不应套用到当前 fresh_base 启动任务**：
   - `value_clip` wandb 实测 **0.2**，而 TB4 文档第 5 节记作 4.0。
   - `num_critic_only_steps` 实测 **4**、`num_rollout` 实测 **36**，
     而当时启动脚本默认写的是 2 和 3（被外部环境变量覆盖）。
   当前 fresh_base 是 warmup=2、num_rollout=100；critic `value_clip=4.0` 来自角色 override，
   顶层 args dump 的 0.2 不代表 critic 实际值。详见第 8.1 节。

---

## 7. 回滚

| 改动 | 回滚方式 |
|---|---|
| `max_turns` | 把 YAML 改回 `60`（备份：`/tmp/y.bak`） |
| Dense Reward | 不设 `CUSTOM_REWARD_POST_PROCESS_PATH`，自动回到 `test_count_progress_rewards` |
| TB-GAE | 当前未接通；未来以公共 `enable_tb_gae=False`（CLI 可用 `--no-enable-tb-gae`）回滚，并移除手工 TB hook；见第 9.3 节 |
| 两个新 `.py` | 历史版本未默认启用；本次不删除。未来接入后的回滚以开关为准，不在线删除模块 |

---

## 8. S3 / S4 当前调用链复核（2026-09-27）

**本节及第 9–10 节是本次工程设计的准确信息；取代旧版第 8 节的实现建议。**
本次只编辑本文档。核查依据是**工作区当前文件（包含已有未提交改动）**，而非上游版本；
只运行无 Ray、无 CUDA 的 CPU 小例子和索引复算，没有启动训练、加载模型、运行启动脚本或更改运行配置。
旧版中的 H20 微基准数字属于此前记录，本次未重测，不作为生产性能承诺。

### 8.1 明确结论与实际启动参数

**现有启动脚本不能仅靠参数或环境变量直接启动有效的 S3/S4。**
字符串可以传进 argparse，但端到端能力没有接通；新增一个 YAML 布尔字段也不能使尚未实现的逻辑自动工作。

已核查的调用链：

```text
launch_27b_fresh_base_critic_warmup2.sh:196
  -> run_27b_dense_128gpu_ppo_256k.sh:538–541
  -> run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest_partial_reward.sh:510
  -> run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest.sh:733
  -> run_qwen35_9b_terminal_bench2_64h20.sh:811–824
  -> train_async.py
  -> critic.train_critic / actor.train_actor
  -> loss.compute_advantages_and_returns
```

默认布局从实际生产启动日志核实，不能以测试里的 Namespace 代替生产证据：

| 配置 | 本任务值 | 工作区证据 |
|---|---|---|
| 训练入口 | `train_async.py` | 最下游脚本 :811；`run_fresh_base_warmup2.log:141` |
| TP / CP / PP | 4 / **4** / 2；64 训练 GPU，DP=2 | 同一日志的最终 argv；27B 脚本 :149 |
| QKV / CP 模式 | **`thd` / `allgather_cp=False`** | 日志 :813 / :183 |
| GAE / KL | `gamma=1.0, lambd=1.0, kl_coef=0.0` | 日志 :430 / :556 / :550 |
| 当前 hook / SC | `custom_advantage_function_path=None`；SC=False | 日志 :256 / :348 |
| warmup / rollout | `num_critic_only_steps=2, start_rollout_id=0, num_rollout=100` | 启动脚本 :121 / :140 / :169；日志 :735 / :746 |
| 轨迹适配 | `trajectory_retokenized` | fresh_base 任务 YAML :179 |
| 奖励 | `partial_reward.test_count_progress_rewards` | 启动脚本 :150；**本任务并未自动切到 S2 奖励** |
| critic 参数 | `value_clip=4.0, clip_grad=100.0` | `megatron_roles_27b_fresh_base.yaml` critic overrides |

`EXTRA_MISC_ARGS` 在顶层 :143 保留外部输入，下游 :683 按空白拆成 `MISC_ARGS`，
最终进入训练 argv。**直接给最外层 shell 脚本追加位置参数并不等价**：它没有解析或沿链传递 `"$@"`。
现有可用参数入口是环境变量 `EXTRA_MISC_ARGS`，任务配置入口是 `CONFIG_TEMPLATE`，
角色配置入口是 `MEGATRON_CONFIG_PATH`。独立的 `ENABLE_TB_GAE=True` 环境变量目前无人读取。

现状的参数陷阱：

- `--custom-advantage-function-path` 已注册（`arguments.py:942`），会优先于原生 PPO 分支。
- `--gamma/--lambd` 是原生 token GAE 参数；`tb_gae.py:221–222` 不读它们，
  而读取 `tb_gae_gamma_turn/tb_gae_lambda_turn`，缺省落到 **0.999**。
- `--enable-tb-gae` 和两项 `--tb-gae-*-turn` **均未注册**。
  当前 Megatron 解析调用带 `ignore_unknown_args=True`（`megatron_utils/arguments.py:181`），
  不能把“命令没有报 unknown argument”视为生效。
- 当前任务 YAML 在 `arguments.py:1964–1971` 通过 `setattr` 注入，
  **现有行为是 YAML 覆盖已解析 CLI**。它能注入上述浮点属性，但解决不了生产、传输、计算缺口。
- 角色 YAML 的 `overrides` 只作用于角色副本，不会反向更新 RolloutManager 的全局 args。
  只在 actor/critic 下写 `enable_tb_gae: true`，无法保证 rollout 生产字段。

### 8.2 直接挂 hook 的具体阻断与时间点

`slime_plugins/harbor/tb_gae.py:212–219` 是精确阻断位置：

```python
te = rollout_data.get("turn_end_mask")
done = rollout_data.get("is_real_done")
if te is None or done is None:
    raise RuntimeError(
        "TB-GAE needs per-token `turn_end_mask` and `is_real_done`. They are "
        "not forwarded by slime/ray/rollout.py::_split_train_data_by_dp yet -- "
        "see TB4_REWARD_TBGAE_PATCH.md 'TB-GAE plumbing' before enabling this."
    )
```

这是**显式条件守卫 + RuntimeError，不是 Python `assert`**。
触发前提是已进入 hook 且 `values is not None`；
`values is None` 在 :210–211 提前返回，非最后 PP stage 在 `loss.py:709–711` 更早返回。

在目标脚本默认配置、未发生更早的环境/训练错误且运行到 actor 首次训练的条件下：

1. 角色 YAML 使 `_apply_megatron_role_overrides` 执行；
   `arguments.py:1603–1607` **无条件清空 critic 的 custom advantage hook**。
2. `train_async.py:74–80` 的 rollout_id **0、1 只训练 critic**，
   `actor.py:456` 调用的是原生 GAE；因此不会在第一步就遇到 TB-GAE 缺字段错误。
3. rollout_id **2** 首次训练 actor，`actor.py:554 → loss.py:728–731 → tb_gae.py:215` 才抛该错误。
   warmup/start/训练入口若被覆盖，时间点随之改变；不能承诺墙钟“约 1.8 小时”。
4. 默认 `SAVE_INTERVAL=5`，不能沿用旧文“前两步必写完整 checkpoint”的描述。

即使先把字段接上，也会出现 critic 原生 returns、actor TB advantages 的不一致；
所以**仅放开 hook 不构成有效实现**。第 9 节采用独立的布尔分派，让两角色同时走 TB 分支，
保留 critic 清空通用 actor hook 的现有规则。

### 8.3 数据管线：逐层核查，包含旧文漏项

| 层 / 文件与位置 | 当前事实 | 开启 TB-GAE 所需补齐 |
|---|---|---|
| `atif_adapter.py:72,514,743` | 入口没有 TB 开关；不生产两个字段 | 向适配入口传入同一个已解析布尔值，只在 True 分支构造 |
| 同文件 :841–904 | 记录 action output 与 observation 的拼接位置；split 时生成多个 chunk | 与实际追加/切片同步标记 turn 末尾；split 重置，不能推测 token 边界 |
| `Sample`，`types.py:9–54,153–179` | 没有声明两字段；但 metadata 是自由 dict，to_dict/from_dict 也保留动态属性 | **无需无条件新增两个 dataclass 默认字段**；建议使用仅开启才存在的 metadata 键 |
| `sample_postprocess.py:107–178` | **生产已启用** response hard cap；会同时改 tokens、response_length、loss_mask、logprobs | 两掩码必须使用完全同一 `response_drop:response_drop+response_length` 切片 |
| `rollout.py::_convert_samples_to_train_data:930–1063` | 显式构造键，不提升两个 metadata 字段 | True 分支验证所有保留样本，提升为顶层逐样本数组；不能只查 samples[0] |
| `rollout.py::_split_train_data_by_dp:1094–1128` | 固定白名单不含两键 | True 分支按**同一个 partition**显式分发；现状即使上游加键也会静默丢弃 |
| `slime/utils/data.py:285–296` | Ray get 后只额外处理 total_lengths；不再过滤其它键 | 不必改；不能误认为这里还要重分两掩码 |
| `actor.py::_get_rollout_data:226–300` | 只转换已有 token/loss_mask 等；loss_masks 是**完整 response**，values 是 **CP 局部片段** | 不能直接把完整掩码与局部 values 配对；本设计在 TB helper 内按需转换 |
| `megatron_utils/data.py:244–257` / `model.py:507–525` | get_next 按请求键取值，model 只取已有 returns/advantages | 两掩码在计算 advantages 时消费，**不必进入每个训练 microbatch 的 get_batch** |
| `data.py:305–376` | 日志不跳过两字段 | 保留字段时必须跳过：bool Tensor 的 `mean()` 可能报 dtype 错；list[list] 的 `sum` 也不合法；float Tensor 才是无意义均值 |
| `sample_postprocess.py:197` | actor 的晚期回调在 GAE 后运行，仍可尝试裁 response | True 时仅检查 cap 已满足；禁止 GAE 后重裁，发现则显式报错 |

因此，“types.py/data.py 没放行”要细分：**Sample 不是强制白名单屏障**，
真正的传输屏障是 convert 未提升和 split 未转发；data.py 的风险主要是日志消费类型。
`RolloutBatch` 是类型别名（`types.py:225`），并不在运行时自动校验或搬运 tensor。

额外需要接开关的调用者是 `slime_plugins/harbor/rollout.py::_run_one_trial:1283–1295`。
adapter 没有全局 args，单改 adapter 内部 `getattr(args, ...)` 是无法工作的。

### 8.4 算法入口仍有三项硬缺陷，补白名单并不够

1. **标量奖励被丢失。** `loss.py:702` 的 `rewards` 是 `list[float]`。
   原生 PPO 在 :740–748 把标量 outcome 加入 token reward；custom hook 在此之前分派，绕过了这段。
   现有 `tb_gae.py:227` 对非 tensor reward 直接 `zeros_like(v)`，既不放入 outcome，
   也不使用已算好的 `rollout_data["kl"]`。本次 CPU 复现：reward=0.7、两 action token、
   γ=λ=1，补齐 masks 后 returns 得到 **[0.0, 0.0]**。
2. **入口没有调用 CP wrapper。** `tb_gae_advantage:229` 调的是 `compute_tb_gae`，
   `compute_tb_gae_cp` 没有被入口使用。仅修改 wrapper 不能修复生产入口。
3. **形状与 tensor 转换缺失。** 完整 response 的 loss_masks 与 CP 局部 values 长度不同；
   CPU 模拟 `values=(1,2), action_mask=(1,8)`，
   会在 `compute_tb_gae:60–61` 抛 `ValueError`。
   如果上游仅传 Python list，`te[i].bool()` 还会先触发 AttributeError。

### 8.5 CP=4 / THD / BSHD 兼容性

**当前 THD 也不安全，不是只在切 BSHD 时才有风险。**
TB-GAE 消费的是 response-aligned values/rewards，而不是直接重排 Q/K/V；
问题在 CP tensor 的布局契约与 shifted-logit 索引。

#### 8.5.1 已复现的重新切片错误

`cp_utils.py:50` 返回：

```python
chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1)
```

现有 `tb_gae.py:177–183` 取 `[1]`（未裁剪的 chunk），再减 `prompt`；
正确的 `slice_log_prob_with_cp:360–366` 取 `[2]`（logits），减 `prompt-1`。
本次直接抽取工作区 offset 函数，用 CPU 复算 total=4096、response=2048、CP=4：

| CP rank | 正确 response 索引（半开区间） | 当前 TB scatter 索引 | 长度正确 / 当前 |
|---|---|---|---|
| 0 | [1537,2048) | [1536,2048) | 511 / 512 |
| 1 | [1025,1537) | [1024,1536) | 512 / 512 |
| 2 | [513,1025) | [512,1024) | 512 / 512 |
| 3 | [0,1) + [1,513) | [0,512) | 513 / 512 |

`k=min(...)` 会掩盖长度问题。应删除自写 scatter 和静默截断，
直接调用 `slice_log_prob_with_cp(full[:response_len], total_len, response_len, qkv_format, max_seq_len)`，
逐样本断言输出长度等于原始 CP local values 长度。

#### 8.5.2 BSHD 的风险也在 gather 侧

`compute_tb_gae_cp` 没有 `qkv_format/max_seq_lens` 参数；
更关键的是 **`all_gather_with_cp(tensor,total_length,response_length)` 自身也没有这两个参数**，
内部 :258 固定按默认 THD 计算 offset。

因此当前 BSHD **不保证直接触发“max_seq_len must be provided”**：
因为参数根本没传进去，它仍可能按 THD 错算，随后在 :264 的 shard 长度断言失败，
或恰好长度一致而静默错位。只有显式按 BSHD 调 offset 且缺 max_seq_len 时才触发 :26 的断言。
只给 scatter 加 qkv_format 仍然不够。原生 batch GAE 也调用该 THD helper，
**本次 TB 设计不能借机修改关闭分支的原生 helper 行为**。

第一版支持范围明确为：**THD、普通 zigzag CP（allgather_cp=False）、CP=1/4**。
True + BSHD 或另一 CP 模式应在启动前报不支持；
后续实现独立的 TB gather，使用相同 qkv_format/max_seq_len 的两向映射并通过 BSHD 测试后再开放。
不能声称“模型支持 BSHD，所以 TB-GAE 自然兼容”。

#### 8.5.3 显存与性能

- `all_gather_with_cp` 内部用 `requires_grad=True` 的 padding 和可微 all_reduce；
  TB 入口当前没有 no_grad。应在 **True 分支整体 no_grad + detach**，不把 target 连回图。
- 无充分证据保证“无内存泄漏”。可确认的是 full-response gather、pad 和保留计算图
  会增加显存峰值；需针对长序列多轮测量。本次没有做 CUDA 泄漏测试。
- `compute_tb_gae:87–100` 是逐 action 的 Python 循环；`bool(te[j])/bool(dn[j])`
  在 CUDA tensor 上还可能造成逐 token 主机同步，不能仅归因于 kernel launch。
- 旧文 H20 1k/4k action 耗时 1.61s/6.21s 仅为历史记录，无本次复测；
  `global_batch_size=256` 统计的是 group，切 chunk 后 sample 数不一定是每 DP 128。
  不再用它推导本任务“固定增加 20 分钟”。
- 完整 masks 已由 Ray 发到各 CP rank，无需再对它们各做一次 CP gather。
  推荐只 gather values/token rewards；mask 留在 CPU，按样本使用时转 bool tensor。
  FP32 scan、ragged/chunked 处理和性能验收是上线必要条件。

### 8.6 终止与 chunk：两个掩码必要，但并不充分

`harbor_finish_reason_type` 来自 SGLang 单次生成的 `meta_info.finish_reason`
（`llm.py:185 → atif_adapter.py:1535`），**不是环境终止字段**。
一次 stop/EOS 可以只是工具调用结束；`extract_status:42–49` 主要检查异常，
“无异常/COMPLETED”也不能证明不是 max_turns 耗尽。
`_trial_metadata:1587–1596` 没有导出明确 termination/truncation 原因。

旁路核查相邻 Harbor 源码也支持该判断：
`../harbor/src/harbor/agents/terminus_2/terminus_2.py:1283,1558–1566,1662–1669`
在循环耗尽和确认 task_complete 返回后，未在所读 metadata 中统一写出显式终止枚举。
此源码仅作接口缺口证据；实际运行包的版本和路径仍需落地时核对。

还存在独立的 bootstrap 问题：当前 `compute_tb_gae:96` 在数组尾部总是 `next_v=0`、
`last=0`，**把 chunk 尾的 done 写成 False 并不会产生下一 chunk 的 value 或 advantage**。
而 adapter :912 把 trial reward 赋给每个 chunk；
现有 DP 调度只保证同 group 在同一训练 step，
`dp_schedule.py:253–301` 仍可能把 sibling chunks 的 microbatch 分到不同 DP rank。
所以不能假设共享 group_id 即可本地拼回完整轨迹。

第 9.6 节明确单 chunk 首期支持范围、拒绝条件，以及跨 chunk 的后续完整设计。
**未完成后续边界工程之前，不能对包含 split/未知终止的完整生产数据宣称 S3/S4 已可用。**

---

## 9. TB-GAE 可选开关与 Zero Regression 工程设计（待实现）

### 9.1 不可违反的开关语义

`enable_tb_gae` 是**算法与数据通路共用的唯一开关**，最终类型必须是 Python bool，
默认/缺省**恒为 False**；仅 `args.enable_tb_gae is True` 开启。
S3 是**开启 TB-GAE 且轮次 γ=λ=1.0**，不能把 S3 偷换成“开关关闭”；
S4 是**开启且 γ=λ=0.999**。

| 最终开关 | 数据 / 分派契约 |
|---|---|
| False / 缺省 | 原生 Token-level PPO GAE；仍调用现有 `get_advantages_and_returns_batch(..., args.gamma, args.lambd)`；原 rewards 构造、CP、归一化、数据顺序不变 |
| True + S3 | 生产并传递两掩码，调用 TB 函数；轮次 γ=λ=1.0 |
| True + S4 | 同一 TB 数据与调度路径；仅轮次 γ=λ 改为 0.999 |

关闭路径的“零影响”必须落实为：

- 不构造 turn span 补充结构、两个 masks、空列表占位、None 占位、调试 hash 或 TB sample metadata。
- 不为每个 Sample 加 `turn_end_mask=None/is_real_done=None` dataclass 字段；
  这会改变对象/序列化字典，违反“字段完全不创建”。本设计直接复用已有 metadata，
  只在 True 时添加其键；`types.py` 可不改。
- 不改变原 Ray train_data/DP payload 的键集和数值；不增加 tensor 搬运或 collective；
  不进入新的 target 算子；不新增显存分配。
- False 时不导入 TB 算法模块、扫描边界或验证 TB tensor；每层只保留必要的 O(1) 布尔分支。
- 原生 PPO 分支的代码、奖励加法及运算顺序保留，不为“共用函数”顺便重构；
  原 `gamma/lambd` 默认值也不改。轮次浮点参数即使出现也**不启用**算法。
- TB 专属配置的新增 namespace 标量与一次性解析属于控制面；数据面不增加字段/字节。
  Python 分支和参数解析不能承诺物理意义“零 CPU 指令/零纳秒”，
  验收应要求**零额外 token 工作、零额外通信/显存、相同基线数值与无可测吞吐退化**。

这里的“原生路径保证”针对目标脚本的 PPO baseline（当前没有其它 custom advantage）。
其它既有 custom hook 在 False 时仍沿用原有优先级；不能为了 TB 开关破坏其它实验。
但 **TB 专用 hook 路径不能绕过开关**：False + 显式 TB hook 应在启动期配置报错，
True + 同一 TB hook 可规范化为 None 交由独立分派；True + 其它 advantage hook 则报冲突。
禁止以 hook 非空、两个字段恰好存在、gamma 值或环境字符串 truthiness 推断启用。

“动态”指**每次启动可参数化选择**，不指训练中逐 rollout 热切换。
正在运行的 job 不在线改开关；重新启动才切换模式，避免在途样本无掩码而消费者已开启。

### 9.2 参数定义、严格解析与配置优先级

拟增加的公开参数：

| CLI | args / YAML 键 | 缺省与约束 |
|---|---|---|
| `--enable-tb-gae` | `enable_tb_gae` | 无 flag 为 False；出现 flag 代表显式 True |
| `--no-enable-tb-gae` | 同上 | 显式 False，可覆盖 YAML True |
| `--tb-gae-gamma-turn FLOAT` | `tb_gae_gamma_turn` | 仅启用后解析/校验；未指定时 1.0 |
| `--tb-gae-lambda-turn FLOAT` | `tb_gae_lambda_turn` | 仅启用后解析/校验；未指定时 1.0 |

不使用 `type=bool`，不执行 `bool("false")`。
YAML 开关必须经 safe_load 得到 **`type(value) is bool`**；
拒绝带引号的 `"true"/"false"`、数值 0/1、null。
CLI 使用正负 flag，**`--enable-tb-gae false` 不是合法语法**。
两个相反 flag 同时出现、重复冲突值、拼错的 `--tb-gae-*` 应报错，
不能因为外层 parse_known_args 就吞掉残留 token。

实现建议：独立预解析这几个 CLI 参数，用
`argparse.BooleanOptionalAction(default=argparse.SUPPRESS)` 和
`default=argparse.SUPPRESS` 的浮点参数记录“是否显式出现”，并在主 parser 注册同名参数。
内部 sentinel 只用于合并，**创建任意 worker 前将缺省物化为 bool False**。
必须保留原始 CLI 的 explicit map，不能从主 Namespace 中的 False 推断“用户显式关闭”。

**仅这几个 TB 参数使用以下新优先级，现有其它参数不改：**

```text
显式 CLI
  > --custom-config-path 指向的任务 YAML 顶层键
  > --megatron-config-path 指向的角色 YAML 顶层公共键
  > 内建默认：enable_tb_gae=False；启用时 gamma_turn=lambda_turn=1.0
```

按字段逐项合并。例如任务 YAML 开启 + CLI `--no-enable-tb-gae` 最终为 False；
角色 YAML S4 + CLI 只改 lambda 则保留该层 gamma。
开关 False 时两浮点参数是非活跃配置，不构造 TB policy、不做数据校验；
CLI 本身的词法/类型错误仍由 parser 正常报错，不能藉此隐式开启。
True 时要求有限数值、`0 < gamma_turn, lambda_turn <= 1`；
公布的 S3/S4 示例固定成对 1.0/0.999，其余值须明确标为其它实验。

集成到现有 `arguments.py` 的时序：

1. YAML setattr 循环执行前保存 TB CLI explicit map；
   TB 保留键由专用 resolver 处理，不能先被通用 YAML 循环覆盖后丢失来源。
2. 在任务 YAML 已读取、角色 YAML 已读取的共同位置调用 resolver。
   复用当前 SC 配置读取点附近的已有数据，但不要触碰 SC 的优先级和 off 路径。
3. **在 create_rollout_manager / create_training_models 之前**完成合并，
   让 RolloutManager、Harbor 转换、actor 和 critic 继承同一配置。
4. TB 参数仅允许写角色 YAML **顶层公共区**；禁止写在单角色
   `megatron[].overrides` / `args` 中，包括与公共值相同的重复写法。
   在创建任何 worker 前检查并报错，角色复制时再次确保没有局部覆盖。
   这是全链路协议，不能各角色自行选开关。
5. True 时做 TB 专属验证：PPO/`use_critic=True`、启用正常 advantage 计算、
   THD + ordinary CP、当前支持的 Harbor mode；首期要求
   `kl_coef=0`、`use_opd=False`、SC=False，以及兼容的默认 convert 路径。
   不支持的组合报错，不能静默回原生。
6. 只在开启/显式配置时记录一次最终开关、γ/λ、配置来源及布局。
   默认关闭不增加逐 step TB 日志或指标。

即使被更高优先级覆盖，显式给出的 YAML 开关仍须符合布尔 schema；
角色 overrides 中的禁用写法也不能靠 CLI 覆盖来规避检查。
首期适配仅开放本任务的 `trajectory_retokenized` 模式，
其它 mode 未实现并验收 masks 前应拒绝 True，不能进入未接门控的 adapter 分支。

`--advantage-estimator ppo` 必须保留，因为现有 `arguments.py` 由它设置 `use_critic`。
启动脚本无需重写：未来可经已有 `EXTRA_MISC_ARGS` 和两类配置入口传入。
不增加一个 worker 自行读取的 `ENABLE_TB_GAE` 环境变量；如未来提供 env 便捷方式，
只能由 driver 转为上述显式参数后统一解析。

### 9.3 完整 YAML 示例（设计稿，当前代码尚不支持）

下面三个代码块是**完整的角色配置文件示例**；使用与 fresh_base 相同的角色与优化器配置，
TB 键在 `megatron` 列表之外。示例不表示本次已新建/修改这些配置文件，
也不自动切换奖励、warmup、curriculum 等其它实验变量。

关闭（也可直接使用现有文件，完全省略三个 TB 键）：

```yaml
# 未来示例：megatron_roles_27b_tb_gae_off.yaml
enable_tb_gae: false

megatron:
  - name: q38-27b-actor
    role: actor
    overrides:
      lr: 5.0e-7
      min_lr: 5.0e-7
  - name: q38-27b-critic
    role: critic
    overrides:
      lr: 1.5e-5
      min_lr: 1.5e-5
      value_clip: 4.0
      clip_grad: 100.0
```

S3 负对照：

```yaml
# 未来示例：megatron_roles_27b_tb_gae_s3.yaml
enable_tb_gae: true
tb_gae_gamma_turn: 1.0
tb_gae_lambda_turn: 1.0

megatron:
  - name: q38-27b-actor
    role: actor
    overrides:
      lr: 5.0e-7
      min_lr: 5.0e-7
  - name: q38-27b-critic
    role: critic
    overrides:
      lr: 1.5e-5
      min_lr: 1.5e-5
      value_clip: 4.0
      clip_grad: 100.0
```

S4 轮次折扣：

```yaml
# 未来示例：megatron_roles_27b_tb_gae_s4.yaml
enable_tb_gae: true
tb_gae_gamma_turn: 0.999
tb_gae_lambda_turn: 0.999

megatron:
  - name: q38-27b-actor
    role: actor
    overrides:
      lr: 5.0e-7
      min_lr: 5.0e-7
  - name: q38-27b-critic
    role: critic
    overrides:
      lr: 1.5e-5
      min_lr: 1.5e-5
      value_clip: 4.0
      clip_grad: 100.0
```

若选择任务配置，就在现有完整 `CONFIG_TEMPLATE` 的**顶层**加入同名三项；
不放进 `harbor_trial_defaults.agent.kwargs`，也不把一个只含三项的文件替代完整任务配置。
任务层同名项按第 9.2 节覆盖角色公共项。

未来落地并通过验收后，可继续使用原启动脚本。例如 CLI S3（这里只是文档，不执行）：

```bash
EXTRA_MISC_ARGS="--no-load-optim --no-load-rng --enable-tb-gae --tb-gae-gamma-turn 1.0 --tb-gae-lambda-turn 1.0" \
  bash examples/harbor_terminal_rl/launch_27b_fresh_base_critic_warmup2.sh
```

S4 将两个 1.0 换成 0.999；YAML 路径用
`MEGATRON_CONFIG_PATH=/path/to/megatron_roles_27b_tb_gae_s4.yaml` 指向对应完整角色文件。
回到基线可显式加 `--no-enable-tb-gae`，它覆盖仍为 True 的 YAML；
或删除所有配置源中的启用项。**只删除 custom hook 不再是未来版本的回滚规范**。

### 9.4 三层硬门控及字段生命周期

以下均为拟实现伪代码；不是对当前代码已有能力的描述。

**适配层：在原逻辑外仅增加条件追加，不复制/改写关闭路径。**

```python
# slime_plugins/harbor/rollout.py::_run_one_trial
# 现有 sc_conversion_kwargs 等保持原样。
if args.enable_tb_gae is True and not _is_eval_args(args):
    conversion_kwargs["enable_tb_gae"] = True
samples = samples_from_trial_result(..., **conversion_kwargs)

# atif_adapter.py：沿三个实际入口传递 keyword-only enable_tb_gae=False。
# 仅 True 才建立新的边界记录，并按真实拼接输出构造 masks。
if enable_tb_gae is True:
    turn_end_mask, is_real_done = build_tb_masks(aligned_turn_spans, ...)
    sample.metadata["turn_end_mask"] = turn_end_mask
    sample.metadata["is_real_done"] = is_real_done
# False：不调用 helper，不创建/赋值这两个键，不建空 tensor。
```

训练数据的两个字段使用**完整 response 坐标**，长度都为
`sample.response_length == len(sample.loss_mask)`，不含 prompt/padding。
运输阶段可用紧凑的 CPU bool/uint8 数组，训练消费时才变成 `torch.bool`；
不依赖 Python list 自动拥有 `.bool()`。语义是：

- `turn_end_mask[t]`：真正一轮模型输出的末 token；observation 上恒 False。
  本实验的“一轮”明确以 adapter 中的主 agent `TurnRecord` 为单位。
  多 detail/subagent/多 step 需有独立归属协议，首期拒绝。
- `is_real_done[t]`：只在环境真实结束的最后一个保留 action token 为 True，
  成功和正常失败都可终止；其余位置 False。
- 正常有效样本满足 `done <= turn_end_mask <= action_mask`；
  对 `remove_sample=True` 被原通路置全零 loss_mask 的样本，在 True 分支同时清零 TB masks，
  保留原样本次序/权重，不偷偷改成丢样本。
- turn 输出为空时不虚构边界；最终没有有效 action 的数据应按明确策略报错或沿用原有无效样本处理。
  sample clone 不可继承旧 trial 的 TB masks。
- **边界生成在拼接时完成**，不能从 observation→action 跳变反推：
  两轮之间可以没有 observation，也可能有重分词/被屏蔽的历史输出。
- 前缀/后缀裁剪用同一实际 slice，不凭统计的 dropped-token 计数重新推导。
  若末端被截掉，不能把新末 token 伪造为真实 turn_end/done。

**样本后处理：生产 hard cap 是必改路径。**

`cap_total_tokens_allow_response_trim_in_samples` 的 True 分支把两 mask 使用
`[trim.response_drop:trim.response_drop + trim.response_length]` 同步切片。
仅裁 prompt 时 mask 不变；保留尾部而丢前缀时保留真实尾端 done。
GAE 后的 `cap_total_tokens_allow_response_trim_in_rollout_data` 在 True 时只校验 cap，
若仍超限即报错；不能修改 lengths 后再让已计算的 CP targets 继续使用。
False 时两个函数完全走原逻辑。

**Ray 层：关门的是同一布尔值，不能改成“字段存在就转发”。**

```python
# _convert_samples_to_train_data：保留原 reward/drop/remove_sample 流程。
# 此分支放在样本筛选和 loss_masks 最终确定之后。
if self.args.enable_tb_gae is True:
    from slime_plugins.harbor.tb_gae_data import validate_and_collect_tb_masks
    fields = validate_and_collect_tb_masks(samples, train_data["loss_masks"])
    train_data.update(fields)  # 仅含两个完整 response 对齐的字段

# _split_train_data_by_dp：原白名单、build_dp_schedule、partition 保留。
# 放在每个 rank 的普通字段复制后、ray.put 前。
if self.args.enable_tb_gae is True:
    for key in ("turn_end_mask", "is_real_done"):
        if key not in data:
            raise ValueError(f"TB-GAE DP input missing {key}")
        rollout_data[key] = [data[key][j] for j in partition]
```

True 时校验**全部**样本的字段存在、长度、取值、对齐和边界语义，
缺失任何一个都报错，不以全零掩码补齐；
False 时不遍历 TB 字段、不对数据新增 `None`。
现有 custom convert 提前返回会绕过默认 convert，
因此首期 True + custom convert 报不支持；未来扩展时必须在公共出口补验证，
split 仍是最后一次协议检查。调试 dump/旧缓存缺 mask 也必须在 True 时报错。

**损失/优势层：显式专用分支，与 actor-only custom hook 分离。**

```python
# loss.py::compute_advantages_and_returns
# 原有最后 PP stage return 与 KL 构造仍先执行。
if args.enable_tb_gae is True:
    from slime_plugins.harbor.tb_gae import tb_gae_advantage
    tb_gae_advantage(args, rollout_data)  # 此入口自身也检查 is True + no_grad
    advantages = rollout_data["advantages"]
    returns = rollout_data["returns"]
elif args.custom_advantage_function_path is not None:
    # 原 custom hook 分支不变。
    ...
elif args.advantage_estimator == "ppo":
    # 原有标量 reward -> token rewards 代码原样保留。
    ...
    advantages, returns = get_advantages_and_returns_batch(
        total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
    )
# 其余 estimator 分支和已有 whitening/OPD 后处理顺序保持原样。
```

critic 的 `custom_advantage_function_path=None` 仍照旧，
因为 TB 在它之前由公共开关分派，两个角色都进入 TB；
warmup 阶段的 critic 也必须从第一步使用 TB returns。
actor 使用 critic 回传的同一份 **更新前 values**（现有 `actor.py:453–472,542–548`），
不能给两角色不同 γ/λ 或一个归一化前、一个归一化后 target。

CPU masks 可保留在 rollout_data，`data.py::log_rollout_data` 在 True 时跳过这两键，
不把它们交给 tensor.mean 或 sum。get_batch 不加两键：
已经有 `advantages/returns` 足够训练。
TB helper 在最后 PP stage 按样本/小批转 GPU bool，结束即释放临时 mask；
`actor.py::_get_rollout_data` 因而不需要为每个 PP stage 提前搬两张全长 mask。

### 9.5 开启后的 reward / CP / GAE 计算契约

推荐在 TB 入口形成清晰的 full-response 契约：

1. `values`、现有 token KL 是 CP local ragged lists；两 masks 和 loss_masks 是 CPU/已有
   full-response lists。检查它们与 tokens/lengths 的关系，不能混用。
2. 只在 THD ordinary CP 的受支持分支 gather values/必要的 token rewards，
   恢复每条真实 response 顺序；两 masks 已是完整数据，**不做 CP gather**。
   所有 rank 按同样顺序调用 collective，空 local response 也必须参与。
3. 将经过现有 reward postprocess 的**标量 outcome**放到最后一个有效 action 位一次；
   完整单 chunk 末端必须可定位，不能沿用“非 tensor 就零奖励”。
   首期固定 `kl_coef=0`，与当前任务一致，不额外变更奖励公式。
   以后若开放非零 KL，需实现 action 位的 KL reward 并解决 critic 被置 kl_coef=0 的角色差异，
   同时更新负对照条件；不能悄悄漏掉 KL。
4. full action_mask 是当前有效 loss_mask 的 bool 视图；在 action 顺序递推，
   同轮内 γ=λ=1，跨真实 turn 边界才用轮次 γ/λ：
   `delta_j = r_j + g_j*(1-d_j)*V_next - V_j`，
   `A_j = delta_j + g_j*l_j*(1-d_j)*A_next`。
   正常完整终局的尾部 bootstrap=0；截断/跨 chunk 按第 9.6 节单独处理。
5. 用 FP32 累积（支持时输入/输出也保持原有 FP32 target 契约），
   action 位 returns=A+V；非 action 位可置零，由现有 loss_masks 排除。
   不根据 γ=λ=1 绕回原生，S3 也必须真实覆盖 TB 计算路径。
6. CP=1 直接返回；CP>1 使用原 `slice_log_prob_with_cp` 将
   full_adv[:response_len]/full_ret[:response_len] 切回 **logit-aligned** local values 形状。
   检查每条 length 精确相等，不 pad 后随便截短、不跨样本混排。
7. 返回 list[Tensor]，保持原 sample/partition 顺序；
   targets `requires_grad=False`、`grad_fn is None`。
   仅在该分支内优化 scan：用逐 turn 的边界摘要加块内向量化/稳定反向 scan，
   或先以 CPU 参考实现校验，再做 GPU 实现。禁止逐 CUDA token 的 Python bool/item 同步。
8. 沿用原 whitening 的时机与 mask，不加一次新的优势标准化。

第一版可直接把 `compute_tb_gae_cp` 改为“local values/rewards + full masks”的 list 接口
（见第 10 节），由 `tb_gae_advantage` 确实调用它；
避免将所有 ragged samples 强行 pad 成 256k×batch 的大张量。

S4 的折扣不保证长任务 `|advantage|` 增大。
终局奖励对更早 action 的直接递推系数随跨轮次数按 `(gamma*lambda)^n` 衰减；
γ=λ=0.999、n=50 时约 **0.9048**，并非只算 `0.999^50≈0.9512`。
完整 advantage 还含 value residual，符号与大小需测量，不能作为已知收益。

### 9.6 chunk 与真实终止的分阶段支持边界

**首期 S3/S4 的受支持输入：单 detail、单 chunk、真实终止已确认、保留最终 action 的轨迹。**
前缀 hard cap 可以在计算前同步裁剪；不把右端截断当作完整终止。
True 时遇到多 chunk、未知结束原因、外部 timeout、max_turns/长度截断且无 bootstrap，
应在 adapter/训练数据转换阶段显式失败；不能设置 done=True 蒙混通过，
不能静默丢弃这些轨迹后仍称“与完整生产基线一致”。

这是一阶段能力边界，不是建议永久过滤困难任务。
当前生产配置允许 split/timeout，因此**仅完成首期代码仍不足以对完整数据上线 S4**。
后续需补完整协议：

1. **可靠终止来源。** 在实际 Harbor agent/trial 终止点输出明确字段，如
   `termination_kind=terminated|truncated|continuing|unknown` 和 reason。
   真实完成/正常失败为 terminated；资源/时长/轮数截断为 truncated；
   sample chunk 切分为 continuing。若将某个限制定义成有限时域 MDP 的终止，
   必须单独明确任务语义，不能混同外部截断。
   adapter 读取已验证来源，而非单次 SGLang finish_reason、reward>0 或 status==COMPLETED。
   如现有运行包确有可信 ATIF 完成事件，可复用；否则必须扩展生产者协议。
2. **仅 True 才携带最小 chunk sidecar。**
   传输 episode/detail 标识、chunk_index/count、终止种类和 bootstrap 所需引用；
   每个新键都受同一开关门控。两个 per-token masks 仍必需，
   但不足以表达跨样本关系，不能把它们宣称为唯一所需信息。
3. **跨 chunk 的连续递推。**
   优先在现有 DP 调度后，于 True 路径交换小型 boundary summaries，
   保留原 microbatch/partition；不要因为 group_id 相同就假设在同 rank。
   对 action 链，每 chunk 的首优势可表示为
   `A_first = b_chunk + p_chunk*A_next_chunk_first`，
   还需下一 chunk 的首 value 来形成尾部 delta。
   gather episode/chunk 顺序及这些边界量后从后向前解出 continuation，
   再本地回填所有 action advantages。若另选整组固定到一个 DP rank，
   则调度/负载改变也必须只在 True 路径，并重做 PP 对齐和吞吐验证。
4. **奖励只分配一次且定义明确。**
   整 episode 终局奖励只落在最后真实结束的 chunk；
   当前“每 chunk 都拿 trial reward”的原生行为保留在 False 路径。
   不能在新路径既逐 chunk 重复 reward，又拼接为同一条 GAE 链。
5. **外部截断 bootstrap。**
   最末 chunk 没有 next chunk 时，需可验证的后继状态 value；
   为有限 rollout 设尾部优势边界=0，但不能把 value bootstrap=0 当作默认真实语义。
   如果后继状态/计算路径不可得，保持显式不支持；不能仅凭 done=False 宣称修复。
6. **避免把边界修复误当算法负对照。**
   上述 episode reward/continuation 语义与旧版独立 chunk 原生 GAE 不同，
   即使 γ=λ=1 也不必相等。
   对这部分数据另设“相同终止/奖励/continuation 语义的 token GAE 参考”，
   区分 chunk 协议修复和 S4 折扣影响；不得改 False 分支来强迫等价。

### 9.7 零回归与 S3/S4 验收矩阵

下列是**后续实现必须通过的验收**；不是本次已完成的训练测试。

| 验收层 | 必须覆盖的条件 | 判据 |
|---|---|---|
| 参数 | 全缺省、YAML false、CLI false 覆盖 YAML true、CLI true 覆盖 YAML false、仅浮点项、字符串 false、未知 flag、单角色覆盖 | 缺省/显式 False 走基线；非法类型/不支持组合及 TB hook 绕过尝试显式失败 |
| False 数据面 | 同一批生产样本、含 hard cap/remove_sample、DP 重排、serialize/deserialize | Sample 字典/metadata 无 TB 键；convert/DP payload 键集、数组和值相同；无新增 tensor 字节 |
| False 算法 | 固定输入/RNG 的 actor 与 critic，包括 warmup | spy 确认调用原函数；TB 模块未加载、helper 未调用；原 advantages/returns 在可确定环境下逐元素相同 |
| False 系统 | 相同 rollout replay 与优化器输入 | RNG 不被 TB 消耗；相同 loss/梯度/更新；通信次数/字节和 GPU allocation 无 TB 增量；吞吐差异不超测量噪声 |
| True 对齐 | 多 turn、相邻 action、重分词、空 output、prompt-only trim、response trim、remove_sample | len 与最终 response 相等；端点不伪造；action 身份与原 token/logprob 相符 |
| True 协议 | 缺任一 mask、旧 dump、custom convert 绕过、晚期 cap | 首个可验证阶段报错，不能静默补零/回原生 |
| CP | CP=1/4、变长、非 8 倍数长度、prompt 落在 shard 边缘、空 local response、turn 跨 shard | gather→TB→slice 等于单进程参考；每 rank 形状精确相等；collective 顺序一致 |
| 两角色 | 首个 critic-only step 与 actor 首次训练 | 相同 old values/reward/γ/λ 产生相同未 whitening targets；禁止半 TB |
| 内存/性能 | 长 response、不同 action 密度、多次 rollout | 无 target autograd 图、无逐 CUDA token 同步；记录峰值/稳态显存、耗时、临时 tensor 上界 |
| 扩展协议 | 多 chunk 跨 DP、terminated/truncated、terminal reward 唯一性 | 与同语义整条轨迹参考相符；不与旧独立 chunk 基线混用等价判据 |

**S3 的等价前提必须写进测试名和报告**：

- 对比同一份固定 values、相同 reward、同一组有效 tokens、同一终止边界；
  原生 `gamma=lambd=1.0`，TB `gamma_turn=lambda_turn=1.0`。
- 首期单 chunk/真实终止，KL=0，observation reward 为 0，终局 outcome 在同一个最终 action 位。
  比较 **action 位**的 unwhitened advantage/returns；
  TB 的 observation target=0，原生该位置可以不为 0，所以不声称全 response 逐元素相同。
- CPU FP32 先比解析/逐 token 参考，再与实际 `get_advantages_and_returns_batch`
  的 chunked 实现做 CP=1/4 比对；基本绝对误差门槛 1e-3，
  对长序列/奖励尺度同时报告最大绝对与相对误差，不因看到误差就无依据放宽。
- 本次小例子 action 位差 0，但全序列 returns 不相等，已验证上述区分；
  单卡内置 selftest PASS 不能代替生产 CP 测试。
- `rollout/returns` 聚合日志或两次独立在线训练曲线不能充当逐元素负对照。
  critic 原生路径“看起来相同”也不能证明 actor TB 正确。
- True 路径的目标算子、边界和 CP 都通过后，才做 S4；
  S3→S4 除两项轮次参数外，reward/warmup/数据/角色设置全部固定。
  如果以本 fresh_base reward 做 S3，请标记该基线；要接本文 S2 奖励，则先完成独立 S2，
  不能在启用 TB 时同时换奖励并宣称单变量比较。

---

## 10. 后续分步代码改造清单与交付边界

**下表仅为未来工作量估计，本次未修改任何列出的 Python、YAML、shell 文件。**
行数是新增/改写逻辑行的粗估（非精确净增）；需按边界支持范围重新评审，
旧版“6 文件约 190 行即可完整落地”不再适用。

仓库 `CLAUDE.md` 提倡 Harbor 通过插件 hook 扩展、不修改 core；当前固定 DP 白名单
无法仅靠 custom convert hook 绕过。本设计将算法保留在插件，仅规划必要的 core 门控与传输改动，
并明确其维护成本。本次用户已要求评估这些接入点，因此无需为文档设计另行请求批准；
这也不构成本次修改 core 的授权。

### 10.1 首期实现文件与函数

| 文件 | 拟改函数 / 责任 | 预计新增或改写行 |
|---|---|---|
| `slime/utils/arguments.py` | 参数注册；`parse_args/slime_validate_args` 统一 resolve；`parse_megatron_role_args` 禁止角色局部 TB override | 60–100 |
| `slime/utils/tb_gae_config.py`（新） | 严格布尔/来源/优先级解析；预启动支持范围校验；不改 SC 解析规则 | 100–160 |
| `slime_plugins/harbor/rollout.py` | `_run_one_trial` 仅训练且 True 时传 adapter keyword | 10–20 |
| `slime_plugins/harbor/atif_adapter.py` | 三个入口传参；生成真实 output spans、同步 chunks/两 masks；真实终止读取与拒绝策略 | 90–150 |
| `slime_plugins/harbor/tb_gae_data.py`（新） | masks/终止协议、检查和收集；CPU 紧凑表示；slice helper | 70–120 |
| `slime_plugins/harbor/sample_postprocess.py` | 样本 hard cap 同步两 masks；晚期 cap 在 True 时只校验 | 30–50 |
| `slime/ray/rollout.py` | convert 在原筛选后收集；split 在同一 partition 下 gated 透传；不改关闭白名单 | 25–45 |
| `slime/backends/megatron_utils/loss.py` | 在原 custom/PPO 前加 TB gated 分派；复用后处理 | 10–20 |
| `slime/backends/megatron_utils/data.py` | True 时日志 skip masks；无需扩充 model.get_batch 的字段清单 | 5–10 |
| `slime_plugins/harbor/tb_gae.py` | 开关守卫；标量 reward；真实 CP 调度、no_grad/detach、ragged full masks、稳定 scan、shape 校验 | 140–240 |
| `slime/utils/types.py` | metadata 方案不新增 dataclass 实例字段；按需补类型注释 | **0–10** |
| `slime/backends/megatron_utils/actor.py` / `model.py` / `cp_utils.py` | 首期沿用已有数据接口与 THD helper，不提前搬 mask，不改原生 GAE helper | **0** |
| `tests/test_tb_gae_config.py`（新） | 解析与优先级、角色一致、hook 绕过 | 100–160 |
| `tests/test_tb_gae_pipeline.py`（新） | off/on、cap、DP、Sample 序列化、两角色入口 | 150–250 |
| `tests/test_tb_gae_math_cp.py`（新） | reward、CPU 参考、CP 模拟、真实 CP=4、小型分布式 worker | 150–250 |

首期业务逻辑约 **540–925 行**，测试约 **400–660 行**；
真实终止生产者未就绪时只能验证已知终止的固定 fixtures，不能据此宣布生产就绪。
不需要改实际 `launch_27b_fresh_base_critic_warmup2.sh`，
日后配置文件由原入口选择即可；本次不新增任何示例文件。

关键签名/接口约定（新函数完整签名，现有函数保留原参数并追加 keyword）：

```python
# 新：slime/utils/tb_gae_config.py
def resolve_tb_gae_config(
    args, *, cli_explicit: dict, task_config: dict, role_config: dict
) -> None:
    """物化 bool False/True；仅 True 物化轮次超参并校验支持范围。"""

def validate_tb_gae_config(args) -> None:
    """在 worker 创建前验证角色一致、PPO、布局、hook 等约束。"""

# atif_adapter.py 现有三个函数分别追加 keyword-only 参数：
# samples_from_trial_result(..., enable_tb_gae: bool = False) -> list[Sample]
# samples_from_rollout_details(..., enable_tb_gae: bool = False) -> list[Sample]
# _normalized_trajectory_samples_from_detail(..., enable_tb_gae: bool = False) -> list[Sample]
# 参数追加在已有 require_routing_replay/score_centering_top_k 后，原调用兼容。

# 新：slime_plugins/harbor/tb_gae_data.py
def build_tb_masks(
    response_length: int,
    output_spans: list[tuple[int, int]],
    action_mask,
    *,
    termination_kind: str,
    is_last_chunk: bool,
) -> tuple[list[bool], list[bool]]:
    """spans 使用最终 response 半开区间；终止种类必须来自可信上游。"""

def validate_and_collect_tb_masks(
    samples: list[Sample], final_loss_masks
) -> dict:
    """返回 turn_end_mask/is_real_done 的逐样本 CPU 数组；逐条验证。"""

def slice_tb_masks(metadata: dict, *, start: int, stop: int) -> None:
    """仅 True 调用；按与 response 相同的 slice 原位更新两个字段。"""

# 签名不变、内部加 gated 分派：
# RolloutManager._convert_samples_to_train_data(self, samples)
# RolloutManager._split_train_data_by_dp(self, data)
# compute_advantages_and_returns(args, rollout_data)
# log_rollout_data(rollout_id, args, rollout_data)
# cap_total_tokens_allow_response_trim_in_samples(args, all_samples, data_source=None)
# cap_total_tokens_allow_response_trim_in_rollout_data(args, rollout_id, rollout_data)

# tb_gae.py：采用 ragged local values/rewards + full masks 的统一 CP 接口。
@torch.no_grad()
def compute_tb_gae_cp(
    values_shards: list[torch.Tensor],
    token_reward_shards: list[torch.Tensor],
    action_masks_full,
    turn_end_masks_full,
    is_real_done_full,
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    gamma_turn: float,
    lambda_turn: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """首期 THD/ordinary CP；True 入口统一调用，输出与 local values 对齐。"""

@torch.no_grad()
def tb_gae_advantage(args, rollout_data: dict) -> None:
    """检查 enable_tb_gae is True；构造 token rewards、调度 CP、写 detached targets。"""

# 完整 episode 扩展时，不把 continuing 当 terminal：
def compute_tb_gae(
    values, rewards, action_mask, turn_end_mask, is_real_done,
    *,
    gamma_turn: float,
    lambda_turn: float,
    bootstrap_values=None,
    continuation_advantages=None,
):
    """非终止尾部必须显式有边界值；不得默认为 0 冒充 bootstrap。"""
```

`token_reward_shards` 的构造可先在 full response 上放置 outcome，再按既有 CP slice
映射到各 rank，或精确判断哪个 local logit slot 持有该 action；
不能无条件 `rank0[-1] += reward` 后假定其永远等于最后一个有效 action。

### 10.2 完整生产支持的追加工作

| 扩展 | 可能涉及文件 / 签名 | 预计新增或改写行 |
|---|---|---|
| 真实终止生产者 | 实际 Harbor 包 `agents/terminus_2/terminus_2.py`、必要的 result/context schema；显式写 termination_kind/reason；adapter 读入 | 50–120，需先确认运行包版本 |
| 跨 chunk 传输 | `tb_gae_data.py`、`atif_adapter.py`、`slime/ray/rollout.py`；gated episode/chunk sidecar | 60–120 |
| 跨 DP continuation / bootstrap | `tb_gae.py` 或新增 `tb_gae_boundary.py`；`resolve_chunk_continuations(boundaries, *, dp_group)`；需要时增加后继状态 value 前向 | 150–300，后继状态协议复杂时另估 |
| BSHD | TB 独立 gather helper：`gather_tb_response(local, total, response, *, qkv_format, max_seq_len)`；不改 off helper | 60–100 |
| 扩展验收 | 多 chunk/跨 DP/真终止/截断/奖励唯一性/BSHD tests | 200–350 |

这些追加项不能隐入“+2 个字段”的工时估计；
尤其是跨 chunk 连续 GAE、外部截断 bootstrap 和 BSHD 都要有独立验收报告。

### 10.3 分步执行顺序与回滚

1. **冻结 baseline。** 用当前代码和固定 rollout 输入保存必要的 Sample/DP payload、
   PPO targets、loss/梯度参考及参数来源。确认 dirty 工作区来源，不覆盖已有改动。
2. **先落开关/解析/独立分派。** 缺省 False，不修改原 PPO 分支；
   True 在数据协议未就绪时明确失败。先通过 off 的数据面和数值回归。
3. **接适配、hard cap、Ray。** 每个新增键都从 True 来源生成，沿同一个 partition 传递；
   在此阶段验证 field completeness、未知终止/多 chunk 的拒绝边界。
4. **修算法入口与 CP。** 保留 CPU 参考，修 reward 丢失、调用 wrapper、shape/off-by-one，
   no_grad 与 scan 性能；验证 critic warmup 与 actor 相同配置。
5. **跑单 chunk S3 验收。** 固定输入双计算后再用独立的小型 CP=4 smoke job；
   不要运行生产启动脚本来验证报错。负对照通过才进入 S4。
6. **补全生产边界再上线。** 真实终止来源、split/外部截断协议完成并验收后，
   才把完整生产数据称为支持；若要 BSHD，单独验证后开放。
7. **S4 只改两项轮次超参。** 比较按 turn/长度分桶的 advantage、returns、value loss、
   explained variance、任务 reward 和训练耗时，不预设涨点或优势必然增大。
8. **回滚通过公共 False。** 新 job 显式 `--no-enable-tb-gae`，
   或从所有配置源移除启用项，并移除手工 TB hook；
   默认启动脚本、数据通路和原生 GAE 必须恢复到既定 baseline。
   不在运行中的 job 热切开关或复用缺乏模式校验的 TB 训练缓存。

### 10.4 本次文档更新的验证记录

本次只运行了 CPU/只读验证：

- 对工作区 `get_logits_and_tokens_offset_with_cp` 直接抽取执行，复现第 8.5.1 节四个 rank 的索引差异。
- 运行现有 `tb_gae.py` 的 CPU 内置 selftest：PASS；范围仅是单卡核心数学函数。
- 直接调用当前 hook，复现缺字段 RuntimeError、补字段后 reward=0.7 却得到零 returns、
  以及 full mask/local values 的 shape ValueError。
- 同一小型含 observation 轨迹的 S3 对照：action 位 advantage/returns 最大差 0；
  全 response returns 不相等，证明需限定比较域。
- 文档中的 YAML 示例做 safe_load 和 bool/角色结构检查；Markdown 代码围栏配对检查。
- 对本仓库 `slime/slime_plugins/examples/tests/scripts` 中的 Python、shell、YAML 文件
  加两个训练入口（合计 676 个文件）在编辑前后核对哈希；本次未写这些源文件。
  工作区已有大量改动且本文原为未跟踪文件，因此不能把整份 git status 当作本次改动清单。

**没有执行真实 CP=4 collective、GPU 长序列基准、端到端训练或上线验收。**
以上设计满足“可选启用、默认原生”的工程约束；它不是已上线功能声明。
