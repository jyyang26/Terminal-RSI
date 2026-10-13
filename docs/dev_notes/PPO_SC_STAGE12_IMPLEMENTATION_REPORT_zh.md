# PPO-SC Stage 1/2 实施与 27B warmcritic 适配报告

日期：2026-09-26。范围：本地代码、离线 CPU 单测、CPU/gloo 多进程验证。未启动训练、未构建或修改 SGLang 影子环境、未修改 live SGLang、未 git commit/push。

## 1. 实施范围

已按 `PPO_SC_INTEGRATION_PLAN.md` 的 83 条 SC-SPEC 记录核验原文件 SHA256，重建并落地 15 个拟议模块；在此基础上补充了本轮要求的公共角色配置、评估 worker 隔离及实施中发现的防御检查。

| 层 | 文件 | 行为 |
|---|---|---|
| 参数 | `slime/utils/arguments.py` | canonical/legacy CLI 共用 destination；custom YAML 后统一 SC 配置；多角色覆盖校验 |
| 公共角色配置 | `slime/utils/score_centering_config.py` | 纯 Python，无 torch/Ray/Megatron import；只处理 SC keys，不改 LR 或 checkpoint |
| 校验 | `slime/utils/score_centering_validate.py` | SC-off 立即返回；检查公式、分母、采样设置、budget、已适配 hook |
| 算法 | `slime/utils/score_centering.py` | score/ppo_weighted 两公式、detach、epsilon=1e-8、CPU codec、完整词表 TP gather |
| 批次算法 | `slime/backends/megatron_utils/sc_terms.py` | packed rows、CP布局、mask、old-policy head、每rollout挂载检测 |
| 损失 | `slime/utils/pg_objectives.py`、Harbor `train_loss.py`、core `loss.py` | 保留 PPO clip；SC gate 后加项；scale0无SC反向图、不做loss加法 |
| 通信 | Harbor `llm.py`、`rollout.py`、`trial_worker.py` | request/top-k解码、JSON-safe extra、预算；SC作业eval worker禁用heads |
| Token | Harbor `atif_adapter.py`、`sample_postprocess.py` | prompt/token/head身份、真实hard-cap同步六轨；禁止训练晚期裁切 |
| 数据 | `slime/ray/rollout.py`、Megatron `actor.py/model.py/data.py` | 三字段同DP分区、CPU head、old-actor生产者、microbatch取数、日志跳过矩阵 |

`ppo_utils.py` 中的原 PPO 算子、`types.py` 的 Sample 定义、`placement_group.py` 的 warm critic checkpoint 隔离保持原字节。未修改既有 reward、GAE、value loss 或启动脚本。

与审阅规格相比的明确补充：

1. 增加角色 YAML **顶层 `score_centering` 公共块**，使 Driver/RolloutManager/TrialWorker/Actor 在创建前共享同一配置；只在 actor.overrides 设置 enable 不足以打通采样端。
2. 允许 Critic 显式 `enable_score_centering: false`，不再沿用规格中“所有角色SC键都拒绝”的过严规则。Actor 可重复声明与公共值一致的SC设置；不同值直接报错，防止训练与rollout失配。
3. Critic init 的关闭守卫提前到 Megatron 初始化之前；角色配置未使用的 deepcopy critic 也能隔离。
4. 真实response切片使用独立数组副本，避免小切片仍持有完整长轨迹底层buffer。
5. SC训练的远端eval增加限定的evaluation_overrides；持久TrialWorker原先保存training args，单在coordinator的eval_args关闭不足以保护远端评估。SC关闭时仍使用原remote调用形式。
6. 将历史单测更新为已审阅的新公式/契约；移除“可关闭对齐”的旧期望，但保留与真实旧源码的单独回归对照。

## 2. 27B warmcritic 的配置方式

原文件 **`examples/harbor_terminal_rl/megatron_roles_27b_criticlr30x_warmcritic.yaml` 没有改动**。原启动脚本仍使用它，SC默认关闭。

原角色参数继续为：

| 角色 | lr / min_lr | clip_grad | value_clip | SC |
|---|---:|---:|---:|---|
| Actor | 5e-7 | 顶层继承，原值1.0 | 无value head | 由公共开关决定 |
| Critic | 1.5e-5 | 100.0 | 4.0 | 强制False |

Critic:Actor学习率为30；warm `_critic` sibling发现、weights-only加载及 `critic_load_optim` 例外仍由原 `_isolate_critic_checkpoint_dirs` 处理，SC不改写load/save/no_load_optim/no_load_rng/finetune。

### 2.1 关闭：缺省或false

继续使用原文件即可，命令和环境无须改变。若需要显式写入，放在**派生角色 YAML 根层**，与 `megatron` 同级：

```yaml
score_centering:
  enable_score_centering: false
```

`sc_top_k` 等惰性参数不决定是否启用。此时LLM不请求Top-k，Sample无新字段，Ray不附加head，Actor loss无SC张量计算或collective。

### 2.2 开启：已提供的诊断派生配置

新增文件：

`examples/harbor_terminal_rl/megatron_roles_27b_criticlr30x_warmcritic_sc_shadow.yaml`

它完整保留原 `megatron` 两角色条目，仅追加：

```yaml
score_centering:
  enable_score_centering: true
  sc_top_k: 32
  sc_formula: score
  sc_tail_eps: 1.0e-8
  sc_weight_denominator: old_actor
  sc_assert_alignment: full
  sc_chunk_size: 256
  sc_wire_format: json
  sc_correction_scale: 0.0
  sc_max_response_bytes: 268435456
  sc_max_trial_bytes: 536870912
  sc_max_batch_bytes: 8589934592
```

这里文件名中的 `shadow` 表示 **scale0诊断模式**，不是SGLang影子环境：JSON读取现有引擎的标准top-logprobs；本轮没有实际请求引擎。SC=开启且scale=0仍计算/报告SC与传输heads，但不改变loss和梯度，不能当作“SC关闭零开销”。

后续批准真实小批联调时可沿现有run2入口选择该文件，无需修改任何 `.sh`：

```bash
MEGATRON_CONFIG_PATH="$PWD/examples/harbor_terminal_rl/megatron_roles_27b_criticlr30x_warmcritic_sc_shadow.yaml" \
  bash examples/harbor_terminal_rl/launch_27b_run2_warmcritic_ctrf82.sh
```

**本轮未执行该命令。** 在尚未验证SGLang真实响应、显存/主存和27B单step之前，不应直接用此命令启动全量128GPU训练。JSON诊断应先限制并发和response长度。

真实加项需在派生文件中将 `sc_correction_scale: 1.0`。公式含义：

- `sc_formula: score`：原PPO loss + 审阅任务要求的 `A(q-alpha*p)` 加项；不切REINFORCE。
- `sc_formula: ppo_weighted` + `sc_weight_denominator: old_actor`：保留当前PPO分母并中心化其有效clip权重；实际old-actor head生产者已接通，CPU/gloo验证通过。仅在需要复现该历史数学定义时选择。

两种公式不是同义词，不能把score模式说成无条件消除PPO clip漂移。tail折叠、conditioning mismatch和生产成本仍有规格书所述前提。

### 2.3 配置优先级与角色规则

参数优先级为 parser/CLI默认 → 角色文件顶层公共SC块 → `--custom-config-path`中的SC键，沿用现有custom YAML优先约定。CLI显式enable和YAML显式false冲突时报错，避免误认为已启用。

CLI别名：`--use-score-centering` / `--enable-score-centering`；`--score-centering-top-k` / `--sc-top-k`。内部只有 `enable_score_centering/sc_top_k` 一组状态，YAML别名归一化后删除，避免critic副本被alias重新开启。

可以在Critic的overrides补写：

```yaml
      enable_score_centering: false
```

不必写，Critic自动关闭。Critic true报错；Actor单独把false改true或单独改k报错，应改公共块。其原因是rollout已经使用公共args，Actor-only改动无法让引擎同步产出相应heads。

## 3. 验证方法与证据

实施前快照：`/tmp/ppo-sc-stage12-46ux60iv/before`。包含当时dirty工作区源码，不以Git HEAD替代用户已有修改。参数/数学64项、损失23项、Harbor与角色55项在实施前通过。

新增统一离线入口：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  python -B tools/test_ppo_sc_offline.py \
  --baseline /tmp/ppo-sc-stage12-46ux60iv/before \
  --output-dir /tmp/ppo-sc-stage12-46ux60iv/final
```

各组在独立进程中运行，拒绝把skip算通过，避开历史mpu/torch全局桩污染。`--baseline`是本轮原源码快照；未来复测应保留这份快照或重新捕获实施前版本，不能拿实施后源码当baseline。日志与JSON结果位于output-dir，本轮最终结果记录在下一节。

覆盖：

- FP32/FP64解析梯度、detach边界、PPO clip临界点、两种公式的全词表/比例tail参考、k1/empty/all-mask、masked NaN/Inf、TP候选越界、chunk不变性。
- binary/json codec、错误shape/bytes/base64/ID/概率/digest、Top-k tail合法miss、hard cap和R3行长度、Sample往返、HTTP gate和预算。
- warmcritic公共开关true/false、角色覆盖、真实slime_validate_args入口、old-policy forward_only回调、动态microbatch逆排序、Ray DP分区、actor/critic/PP-last head归属。
- 旧源码与新SC-off的Harbor/core loss、entropy/KL组合、全部logits梯度、AdamW参数/状态、非SC parser defaults、角色参数、HTTP请求字节及LLMResponse、Token/cap结果逐bit或逐字段相同。
- CPU/gloo `TP2×CP2`与`TP1×CP4`，两种公式、THD zigzag/allgather及BSHD zigzag，与单机全词表的前向和反向比对；空CP有效行、mask与old-head重建。

原有脚本/YAML/patch 378个文件的SHA256全部保持不变。warmcritic原文件SHA256：
`5c9a51ddb35e38d0b94a0c7cad36bcb4bebf9a4afc42d5cdaacf4f276bdd085c`。

## 4. 最终验收结果与限制

**302项离线测试通过，0失败、0跳过；另有两个四进程gloo任务通过。**

| 测试组 | 通过 | 失败 | 跳过 |
|---|---:|---:|---:|
| 参数与数学 | 104 | 0 | 0 |
| 通信、Token、Harbor、角色与checkpoint | 87 | 0 | 0 |
| Harbor/core损失集成 | 23 | 0 | 0 |
| 真实解析、old-policy回调、DP及actor数据 | 4 | 0 | 0 |
| 冻结原源码直接回归 | 12 | 0 | 0 |
| 既有异步rollout、reward、entropy、DP等 | 72 | 0 | 0 |
| TP2×CP2 四进程 | PASS | 0 | 0 |
| TP1×CP4 四进程 | PASS | 0 | 0 |

机器可读结果：`PPO_SC_STAGE12_TEST_RESULTS.json`。逐组日志：`/tmp/ppo-sc-stage12-46ux60iv/final/`。统一runner退出码0。

最后静态核验：25个新增/修改Python模块可compile；原有378个.sh/.yaml/.patch文件SHA256不变；原compute_advantages_and_returns、get_log_probs_and_entropy、value_loss_function、_apply_megatron_role_overrides的AST不变。原规格书未改写，本文记录实施增量与实际测试结果。

已验证的是本地代码与离线数值/配置/数据契约；本阶段未执行真实GPU训练启动、27B/122B单step、NCCL/PP/VPP集群联调、SGLang canary或长上下文内存峰值测量。不能把CPU logits/optimizer对照夸大成全部生产模型参数与显存峰值已实测一致。

SC-off保留原tensor算子路径，但新增Python分支/配置读取并不等于解释器指令字节完全相同。SC-on额外head副本、完整logits梯度buffer、base64、old-head与并发trial预算也不意味着整个job内存硬上限，生产扩量仍须Stage3/4容量验收。

本轮没有改动或构建SGLang环境，也没有应用规格书中的SGLang最小patch。binary模式本地代码已具备，但必须待后续SGLang响应能力验收后才能使用。当前诊断派生文件显式用json，不静默fallback。
