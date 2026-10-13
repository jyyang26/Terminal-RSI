# 1. 架构现状与断层诊断

**交付日期：2026-09-26。状态：仅规格书；本轮只允许写本文件，任何代码块均未应用到源码，也未提交 Git。**

本方案以目标工作区的实际内容为基线，而不是把 HEAD 当作干净源码。目标 HEAD 为 `2cf4f3066ff661fb376dde8dc9f157fff8f017fe`（2026-08-18），其中已经有早期 SC 模块、双损失挂载以及许多生产定制；不得 reset、checkout、覆盖或 cherry-pick 整个上游文件。第 4 章 Before 来自本轮读取的真实工作区，源码 SHA-256 是实施前的漂移门禁。

上游参考锁定用户指定目录 `slime_latest/slime`，HEAD `8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b`，提交日期 2026-09-23。它包含 `4c4adbbf3d1bab34d5123e93d1ad112a2bc597e2`（#2405）以及 `86c97a30`（#2406，top-p replay SC），最后一个提交是 #2407。**因此“最新参考”不能仅等同 #2405。** 本轮尝试公开网页查询、GitHub API 与 `git ls-remote`，未获得有效远端结果，网络连接超时；没有 fetch/pull，也不能证明该快照就是 2026-09-26 远端最新 HEAD。实施 Stage 1 必须再次只读核实；新提交须单独审阅，不自动更新本规格。

已完整通读全部五份指定历史文档。证据优先级：当前源码及本轮可重复测试 > 本轮 Git 元数据 > 历史实测记录 > 历史推断。本文替代旧计划，不修改历史报告。

## 1.1 调用链与文件级差异

两个基准脚本共享启动链：27B 脚本 → partial_reward 脚本 → strongest 脚本 → 64h20 启动器 → `train_async.py`。strongest 的实际默认值为 PPO、keep_old_actor=1、use_rollout_logprobs=0、Harbor custom_loss。27B 为 THD、TP4×PP2×CP4、log_probs_chunk_size=1024；122B 路径启用 R3。SC 不更改这些默认值。

训练路径是 `SlimeSGLangLLM.call → LLMResponse.extra → harbor.llms.chat.Chat.rollout_details → atif_adapter → sample_postprocess → RolloutManager → DP partition/Ray → actor → model.get_batch → Harbor custom_loss`。所有连线必须同时实现。通用 `http_utils.post` 不在 Harbor 生成链上，无需修改。Critic 使用 value_loss，不能继承 SC logits 收集。

下表行数为本轮实际读取值；“+/-”以**上游 → 本地当前工作区**的整文件 `git diff --no-index --numstat`为方向，不是拟议补丁量：

| 相对路径 | 本地行数 | 上游行数 | 上游→本地 + / − |
|---|---:|---:|---|
| `slime/utils/arguments.py` | 2050 | 2130 | 294 / 374 |
| `slime/utils/score_centering.py` | 189 | 302 | 176 / 289 |
| `slime/utils/tensor_store.py` | 不存在 | 139 | 架构缺失 |
| `slime/utils/ppo_utils.py` | 821 | 939 | 346 / 464 |
| `slime/utils/types.py` | 245 | 596 | 41 / 392 |
| `slime/utils/http_utils.py` | 301 | 332 | 34 / 65 |
| `slime/ray/rollout.py` | 1615 | 610 | 1268 / 263 |
| `slime/backends/megatron_utils/actor.py` | 725 | 712 | 228 / 215 |
| `slime/backends/megatron_utils/model.py` | 967 | 1018 | 111 / 162 |
| `slime/backends/megatron_utils/loss.py` | 1369 | 1552 | 276 / 459 |
| `slime/backends/megatron_utils/data.py` | 567 | 227 | 391 / 51 |
| `slime_plugins/harbor/llm.py` | 431 | 不存在 | 架构缺失 |
| `slime_plugins/harbor/atif_adapter.py` | 1564 | 不存在 | 架构缺失 |
| `slime_plugins/harbor/sample_postprocess.py` | 325 | 不存在 | 架构缺失 |
| `slime_plugins/harbor/train_loss.py` | 247 | 不存在 | 架构缺失 |
| `slime/utils/pg_objectives.py` | 55 | 不存在 | 架构缺失 |
| `slime/backends/megatron_utils/sc_terms.py` | 271 | 不存在 | 架构缺失 |

上游新增的 observability、DiskTensorRef 生命周期、routed_experts disk spill 和 Sample.append_response_tokens 在本地没有对应调用链。不移植这些前置设施。保留本地 R3 内存数组路径、group_mask_sums、动态 DP 调度和 Harbor 奖励函数。SC 以 CPU NumPy 数组经 Ray 对象存储传输，不使用业务层 disk spill；Ray 自身是否 spill 由集群配置决定，本文不能声称 Ray 永不落盘。

## 1.2 历史阻碍复核及新增发现

| 项 | 本轮结论 | 本方案落点 |
|---|---|---|
| B1 | PPO 的真实梯度权重是 r×未进入平坦区的指示量，不是 clamp(r) | 两种公式明确分开；weighted 公式使用真实权重 |
| B2 | PPO 分母为 old-actor 重算，不是 SGLang q | weighted 模式必须真实 old head，完整生产者见 4.6；不再默认近似替代 |
| B3/B8 | FP64 不得降精度；stop-gradient 应测试冻结系数解析梯度 | 独立数学测试；不以跨 detach 的普通有限差分作为证明 |
| B4 | 重复 head ID 属非法分布，不能“取第一项后继续” | 解码/打包阶段硬拒绝重复 |
| B5 | group reducer 不是普通 batch mean | 指标逐 token，复用同一 group_mask_sums 归约 |
| B6 | YAML 合并在 slime_validate_args 内；角色覆盖还可能更晚 | YAML 后校验；actor role 后复核；critic 强制隔离 |
| B7 | 历史测试桩污染确实有风险 | 子进程隔离真实 torch；不把 skip 算 Pass |
| B9/C1 | top-logprobs 分支有 `top_logprobs_num > 0` 守卫；省略字段时不进入 | 默认请求字典不新增 top-k 字段 |
| B10/B14 | 早期损失已存在，但生产数据尚未接通 | 本章列出的全链路补丁；Sample 动态字段避免关闭态序列化新增 None |
| B11 | 本轮重现 `corrupt patch at line 1176`；文件缺尾换行；本轮 splitlines=1176，历史为1175 | 不应用整包 top_p.patch，使用本轮精确最小替换 |
| B12/B15/B16 | SGLang HEAD `5a15cde858ea09b77116212a39356f2fc51b8584`；41 文件、+1480/-168，本轮是已暂存修改，不能仅备份 unstaged diff | 完整包影子 + git diff --binary HEAD；不重建或清理 live checkout |
| B17 | zigzag、allgather logits 行序不同；现有 loss_mask 是完整 response | CPU 全 heads，loss 显式布局映射；allgather 优势重建与 addend 回分布 |
| 新增 N1 | normalized 构建器的 drop counters 只是历史投影匹配；真实输出全部保留 | 禁止按 drop counter 切 head；按真实 sample.tokens 验证 |
| 新增 N2 | strongest:515–516 启用 sample_postprocess 的 response hard-cap | 实际裁切时六轨同步；训练后期裁切改为 SC-only fail-fast |
| 新增 N3 | Harbor Chat 对空 completion 不累积 IDs，却会累积非空 extra | SC 空 completion 在 LLM 边界立即报错，防止轮数错位 |
| 新增 N4 | 旧 _APPLIED 是进程全局粘滞标记，且非 PP-last 不计算 loss | 每 rollout reset；仅物理 PP-last 检查，避免正常 PP 被误杀 |
| 新增 N5 | DP partition 是不相交分配，不是每个 DP 都复制全 batch | 内存按分区及 TP/CP/PP 物理副本分别计算 |
| 新增 N6 | 分块 softmax workspace 不等于峰值显存；backward 还返回完整 [T,Vp] 梯度 | 单次 packed gather，明确完整梯度内存与 empty-row 路径 |
| 新增 N7 | DRY_RUN=1 在完整 TRAIN_CMD 之前退出；partial_reward 会先 mkdir、写配置 | Stage 1 的双层快照方案，不以旧 dry-run 输出冒充 argv |

## 1.3 支持边界

本规格完整支持用户两条基线依赖的 Harbor normalized/retokenized、PPO、Megatron、R3、partial reward、THD zigzag CP。也给出 THD allgather-CP 以及 BSHD zigzag/CP1 的行布局。采样固定 temperature=1、top_p=1、top_k=-1、min_p=0、无 penalty/constraint；既是两条基线现状，也避开 #2406 的完整 top-p support 与本地引擎的概率语义差异。开启 SC 时遇到未适配的配置明确失败，关闭时不引入这些限制。

`trajectory_retokenized` 可以使训练条件上下文不同于原采样 prompt；保留该基线行为、保留原始 q，并沿用 metadata 中的 conditioning_match 标志。Token 对齐正确并不等于训练与采样条件相同，也不能证明重分词造成的所有偏差已被消除。

# 2. PPO-SC 数学公式与张量契约

## 2.1 必须解决的公式歧义

设 p=πθ，q 是 SGLang 实际采样分布，o=πold 是冻结 old-actor。最小化的原生 PPO 是：

\[
\ell_{\mathrm{PPO},t}=\max(-A_t r_{t,y},-A_t\,\mathrm{clip}(r_{t,y},1-\epsilon_l,1+\epsilon_h)),\quad r_{t,v}=p_t(v)/o_t(v).
\]

它保持原有 `compute_policy_loss` 调用，既不改成 REINFORCE，也不把 old_log_probs 换成 rollout_log_probs。

用户指定的
\[
\nabla\ell_{\mathrm{SC},t}=A_t\sum_{v\in K_t}(q_t(v)-\alpha_t p_t(v))\nabla\log p_t(v)
\]
对应**未加 PPO 有效权重的 score-centering 加项**。历史 PPO_SC.md 的“中心化 PPO 截断算子”则包含 w̄(v)。两者一般不相等，不能同时宣称为同一个损失。为完整满足明确公式并保留历史可复现实验，新增唯一公式选择参数：

* `--sc-formula score`（默认）：严格实现本任务给出的 q−αp 加项，叠加原生 PPO。以下“公式严格对齐 Pass”指此模式。
* `--sc-formula ppo_weighted`：实现历史意图的 PPO 有效权重中心化；必须同时选实际分母。本地基线必须 `--sc-weight-denominator old_actor`，不是 silently sampler。

上游 #2405 的默认整体目标是 REINFORCE-SC；本文只复用其冻结系数标量化、head 数据契约和 TP 原理，**本文 PPO+score 不是上游整体目标的等价替换**。#2406 的 top-p 扩展不在此固定 top-k port 范围内。

## 2.2 标量损失、尾部与 detach

默认 score 模式：
\[
P_K=\sum_Kp,\ Q_K=\sum_Kq,\quad
\alpha=\frac{\max(1-Q_K,0)}{\max(1-P_K,10^{-8})},\quad
c_v=\mathrm{sg}(q_v-\alpha p_v).
\]
\[
\ell_{\mathrm{SC},t}=\mathrm{sg}(A_t)\sum_K c_v\log p_t(v),\quad
L=L_{\mathrm{PPO}}+\lambda\,\mathrm{Reduce}_{mask,group}(\ell_{\mathrm{SC}}).
\]

用 `(1-P_K).clamp_min(1e-8)`，**不能**用 FP32 的 `clamp(max=1-1e-8)`冒充严格正尾质量：1−1e−8 在 FP32 可舍入为1。q 的真实零尾部保持零，不人为灌入 epsilon 质量。实数中 q=p 且尾质量≥epsilon 时系数为零；浮点乘法/除法不保证逐 bit 零，不能沿用旧报告的“恒为逐 bit 零”承诺。

所有 p、q、α、w̄、old head 和优势系数均在 no_grad 内或显式 detach；最后只有 live logp 参与梯度。故默认模式精确得到所需梯度，不多出 ∇α、∇p 系数或 ∇A。梯度下降时正加项抵消对应正向漂移，不应再加负号。

weighted 模式：
\[
\bar w(v)=r_v\,1[\neg(A>0\land r_v>1+\epsilon_h)\land\neg(A<0\land r_v<1-\epsilon_l)],
\]
\[
\rho=\frac{q_{tail}}{\max(p_{tail},\epsilon)},\
r_{tail}=\frac{\max(p_{tail},\epsilon)}{\max(o_{tail},\epsilon)},\
\alpha=\rho\bar w(r_{tail}),\
c_v=\mathrm{sg}(q_v\bar w(v)-\alpha p_v).
\]
clip 边界采用与 torch clamp/maximum 在相等分支处一致的未截断值；dual-clip 另有第三段，不支持，校验拒绝。

Top-k 折叠依赖 tail 中 q 与 p 成比例；weighted 模式还依赖 o 的对应尾部比例。head 全覆盖时无折叠近似。epsilon 附近是数值正则化，不能声称仍严格等于未经正则化的全词表式。

“E[A·centered_score]=0”只有 A 对当前 action 条件固定时成立。真实优势依赖 action，保留的协方差正是学习信号；weighted 算子又依赖优势符号，不能把含 action-dependent 权重的式子不加条件地提出期望。本文只证明所写标量的梯度等式，不承诺 PPO+score 或 truncated weighted 在任意分布下零漂移。

## 2.3 张量与跨进程契约

| Name | Dtype/Shape | Device/阶段 | 约束及序列化 |
|---|---|---|---|
| SGLang output_topk_token_ids/log_probs | little-endian int32/float32，[C,k] | HTTP base64 | 每个生成 token 一行，全词表归一化，不能在 head 内重归一化 |
| score_centering_head | JSON-safe dict | LLMResponse.extra / Harbor Pydantic / Chat | IDs、logps、行 token base64；prompt+tokens+head SHA256，version=1 |
| rollout_topk_token_ids | np.int32 [R,k] | Sample/DP Ray object | **完整 response**，与环境 token dummy 行一起；SC-off 属性不存在 |
| rollout_topk_log_probs | np.float32 [R,k] | Sample/DP Ray object | 同上；Ray ndarrays 不写入、不转 Python 嵌套浮点列表 |
| sc_response_token_ids | np.int32 [R] | Sample/DP Ray object | 独立行身份；最终逐元素比对 sample.tokens 的 response suffix |
| 上三项 | torch CPU [R,k]/[R] | actor → iterator | PP-last actor 复制为 writable CPU；不在 actor CP-slice |
| sc_old_head_log_probs | torch.float32 CPU [R,k] | old-policy forward → rollout_data | 只在 weighted/old_actor 收集；CP all-reduce 重建后 CPU 保存；不是来自 ref/teacher/critic |
| logits | float32 [B,T,V/TP] | GPU | packed/padded；不修改原存储，不生成每 sample 大 logits 副本 |
| packed row IDs / candidate IDs | int64 [N] / [N,k] | 当前 microbatch GPU | rows 唯一、有界，IDs 对全 TP 词表有界 |
| local advantages/mask | float32/bool [N] | GPU | CP 布局显式映射；masked 行先排除计算 |
| correction/addend/metrics | float32 [N] | GPU | score 乘 detached A；allgather 最后回 zigzag；metrics detach |
| loss_masks/group_mask_sums | 既有 dtype | 既有路径 | 原 group 归约及分布式缩放不改变 |

Padding 没有生成对应 head 行，布局映射只选真实 response 的预测 logits，注意 prediction 行是 token 行减1。response 内工具 token 则必须有有限 dummy head，mask=0；SC 在 softmax 前排除这些行，标量层再次使用 where 防止 0×NaN。Prompt 为空、response 为空、head 缺失、shape/token 不匹配全部 SC-only fail-fast。

Sample 不新增默认 dataclass 字段。已有 `from_dict`会把非 dataclass keys 通过 setattr 恢复，所以动态 SC-only 字段已经可往返；这比新增两个 None 字段更严格地维持关闭态 __dict__/to_dict/Ray 业务 payload。

# 3. 统一超参数配置与零回归证明

## 3.1 CLI 与 YAML

统一公开名 `--use-score-centering` 与 `--score-centering-top-k`，旧名为同一 argparse action 的别名，内部继续沿用 enable_score_centering/sc_top_k，避免两个 enable 状态失配。YAML canonical key 在 custom_config_path 合并后映射，冲突立即失败。

| 参数 | 类型 | 新默认 | 位置/用途 |
|---|---|---|---|
| --use-score-centering / --enable-score-centering | store_true | False | arguments.add_algo_arguments；唯一总开关 |
| --score-centering-top-k / --sc-top-k | int | 32 | [1,1024] 且不超过实际词表 |
| --sc-formula | score / ppo_weighted | score | 精确区分本任务公式与历史加权公式 |
| --sc-tail-eps | float | 1e-8 | 固定数值保护；偏离本规格时拒绝 |
| --sc-weight-denominator | old_actor / sampler | old_actor | weighted 才消费；必须等于真实 PPO 分母 |
| --sc-correction-scale | float | 1.0 | 有限非负；0 为 no_grad 影子诊断且不执行 loss 加法 |
| --sc-assert-alignment | full | full | 生产强制逐 token 身份验证；取消历史 off/sample 的假安全 |
| --sc-chunk-size | int | 256 | SC 自有 workspace 分块；不改原 log_probs_chunk_size |
| --sc-wire-format | binary / json | binary | JSON 仅小批联调；不静默 fallback |
| --sc-max-response-bytes | int | 268435456 | HTTP 单次生成与解码预算 |
| --sc-max-batch-bytes | int | 8589934592 | 全 batch raw heads 预算，打包 Ray 前硬检查 |
| --sc-max-trial-bytes | int | 536870912 | 单 LLM 实例请求前累计预留 raw head 字节 |

store_true 的关闭方式是不传参数，`--use-score-centering=False` 不是合法 argparse 拼写；YAML `use_score_centering: false`合法。旧六个 SC namespace 默认中 epsilon、denominator、alignment 有明确更新，它们在关闭态不参与任何训练计算。参数快照比较需仅白名单这些 SC 键，不能把它们伪装为“所有 namespace 字节均相同”。

校验除 parse/YAML 之外在 Harbor training trial config 处再调用，角色 override 不得单独更改 SC 配置；critic 最终初始化也关掉 SC。实际每次 LLM 请求再次检查 sampling_params，避免配置值与请求值不同。

## 3.2 零回归的可证明命题与不能承诺的命题

令 G=enable_score_centering=False。在同一现有工作区、环境、输入、RNG、拓扑与调度的条件下，按以下投影比较：

1. 两条 .sh 及其父脚本均无拟议修改；既有 EXTRA_MISC_ARGS 足够启用 SC。G=False 时 resolved TRAIN_CMD 和 runtime_env 相同。
2. LLM 不创建 SC 属性、不修改 payload、不读取/解码 Top-k。Harbor extra 中无 SC envelope。
3. atif/sample_postprocess 的 SC 分支不执行；动态 Sample 字段不存在，原 token/mask/logprob/R3 数据保持原值。
4. Ray 打包不添加三项 SC keys；DP 分区与 microbatch schedule 原样。actor 不分配 heads；model 的新 key 只在 G=True 选择，关闭态 get_next 的实际 key 序列保持原值。
5. PPO、GAE、reward、critic、entropy、KL 原函数体和算子调用保持原语义；SC gate=False 不进入 gather/exp/reduce/all-reduce，也不构造新的 autograd 分支。原生 `compute_policy_loss`文件片段完全不改。
6. SC-only 新模块定义与 CPU 控制分支不等于 GPU 计算；关闭态没有 Top-k tensor 读取/分配和 SC collective；既有 Harbor loss 返回值、梯度及报告 key 集合不变。

**这能证明业务有效输入、浮点张量算子路径与 payload 字段结构等价，不能证明整个 Python 程序机器指令“100%逐字节等价”。** 添加 gate 必然添加条件判断；导入源码、pickle 类定义、日志参数快照和 Ray ObjectID 也可能变化。跨两次异步分布式训练存在调度与浮点非确定性，未实测前也不能声称 checkpoint 全 bit 一致。验收要求是受控单步 bitwise、请求 bytes/结构、无新增 SC tensor/collective；完整训练曲线另以配对实验评估，不能用文字形式证明代替。

## 3.3 不改变脚本的启用方式（仅未来实施后）

保留已有 EXTRA_MISC_ARGS 的模型特定值，在作业命令环境追加 `--use-score-centering --score-centering-top-k 32 --sc-formula score`。weighted 则额外 `--sc-formula ppo_weighted --sc-weight-denominator old_actor`。不要重设 USE_ROLLOUT_LOGPROBS 或 KEEP_OLD_ACTOR 来规避 old head 生产者。

binary 模式在 job 提交前把完整 SC SGLang shadow 根目录置于 PYTHONPATH；现有启动器的 runtime_env 已透传 PYTHONPATH。传入的 shadow 路径必须在 driver、actor、SGLang child 上通过 __file__ 和 provenance 全部核验；DIRECT_RUN 情况也验证，不能只相信环境变量。默认关闭作业无需 shadow，使用原 runtime。

# 4. 全模块落地修改规格与完整代码实现

以下 Python 块全部是完整模块、精确 Before/After 局部替换，或完整新增函数。**本阶段不得运行任何 apply/write 工具落地它们。** HTML SC-SPEC 标记供第 6 章只读审计器逐块提取、顺序重建并 compile，不会写源文件。源码 hash 锁定本文审阅基线；执行前若不匹配须重新 diff/review，不能强行应用。局部修改仅列出被替换的连续完整片段；未出现的原代码原样保留，不存在占位实现。


## 4.1 配置层

### slime/utils/arguments.py： 替换 add_algo_arguments 中既有 SC 参数区

**Before**

<!-- SC-SPEC {"path":"slime/utils/arguments.py","kind":"before","id":"slime/utils/arguments.py#0","sha256":"63001174b1c5ca8c1955bac4918bae7ca5f8df34f7e65f3aa8cb2286180d3aaf"} -->
```python
            # ---- PPO + Score Centering (PPO_SC.md) -------------------------------
            # Additive drift correction layered ON TOP of the PPO clip: the clipped
            # surrogate operator is untouched, and a separate centering term removes
            # the systematic drift that clipping reintroduces. See
            # PPO_SC_INTEGRATION_PLAN.md for the derivation and the rollout plan.
            #
            # There is deliberately NO separate SC epsilon: the centering term must
            # subtract the expectation of the *same* weight the optimizer applies, so
            # it reads --eps-clip / --eps-clip-high. A private epsilon would center a
            # different operator and the zero-drift identity would not hold.
            parser.add_argument(
                "--enable-score-centering",
                action="store_true",
                default=False,
                help=(
                    "Enable PPO + Score Centering (additive drift correction, PPO_SC.md). "
                    "The PPO clipped surrogate is preserved as-is; a detached top-k "
                    "correction term is added to the policy loss. Off by default."
                ),
            )
            parser.add_argument(
                "--sc-top-k",
                type=int,
                default=32,
                help=(
                    "Size of the score-centering head set K. Only read when "
                    "--enable-score-centering is set. k=32 matches full-vocabulary "
                    "centering in the paper's ablation at a quarter of the host memory."
                ),
            )
            parser.add_argument(
                "--sc-tail-eps",
                type=float,
                default=1e-6,
                help="Floor for head/tail probability masses in the tail-rescaling ratio rho.",
            )
            parser.add_argument(
                "--sc-weight-denominator",
                type=str,
                choices=["sampler", "old_actor"],
                default="sampler",
                help=(
                    "Denominator of the head importance ratios. 'sampler' treats the "
                    "logged sampler distribution as pi_old; 'old_actor' uses the real "
                    "old-actor head log-probs (requires --keep-old-actor and an extra "
                    "head gather in the old-actor forward)."
                ),
            )
            parser.add_argument(
                "--sc-correction-scale",
                type=float,
                default=1.0,
                help=(
                    "Multiplier on the centering term. 0.0 computes and reports the "
                    "correction without changing the gradient (shadow mode)."
                ),
            )
            parser.add_argument(
                "--sc-assert-alignment",
                type=str,
                choices=["off", "sample", "full"],
                default="sample",
                help=(
                    "Strength of the head/sampled-token cross-check that catches "
                    "misaligned top-k rows. 'sample' checks a small fraction per rollout."
                ),
            )

```

**After**

<!-- SC-SPEC {"path":"slime/utils/arguments.py","kind":"after","id":"slime/utils/arguments.py#0","sha256":"63001174b1c5ca8c1955bac4918bae7ca5f8df34f7e65f3aa8cb2286180d3aaf"} -->
```python
            # Canonical CLI names share legacy destinations; no double enable state.
            parser.add_argument("--use-score-centering", "--enable-score-centering",
                                dest="enable_score_centering", action="store_true", default=False)
            parser.add_argument("--score-centering-top-k", "--sc-top-k",
                                dest="sc_top_k", type=int, default=32)
            parser.add_argument("--sc-tail-eps", type=float, default=1e-8)
            parser.add_argument("--sc-formula", choices=["score", "ppo_weighted"], default="score")
            parser.add_argument("--sc-weight-denominator", choices=["sampler", "old_actor"], default="old_actor")
            parser.add_argument("--sc-correction-scale", type=float, default=1.0)
            parser.add_argument("--sc-assert-alignment", choices=["full"], default="full")
            parser.add_argument("--sc-chunk-size", type=int, default=256)
            parser.add_argument("--sc-wire-format", choices=["binary", "json"], default="binary")
            parser.add_argument("--sc-max-response-bytes", type=int, default=268435456)
            parser.add_argument("--sc-max-batch-bytes", type=int, default=8589934592)

```

### slime/utils/arguments.py： slime_validate_args，custom_config_path 合并后

**Before**

<!-- SC-SPEC {"path":"slime/utils/arguments.py","kind":"before","id":"slime/utils/arguments.py#1","sha256":"63001174b1c5ca8c1955bac4918bae7ca5f8df34f7e65f3aa8cb2286180d3aaf"} -->
```python
    if getattr(args, "enable_score_centering", False):
        from slime.utils.score_centering_validate import validate_score_centering_args

        validate_score_centering_args(args)
```

**After**

<!-- SC-SPEC {"path":"slime/utils/arguments.py","kind":"after","id":"slime/utils/arguments.py#1","sha256":"63001174b1c5ca8c1955bac4918bae7ca5f8df34f7e65f3aa8cb2286180d3aaf"} -->
```python
    # Accept canonical YAML keys only when present; reject conflicting switches.
    if hasattr(args, "use_score_centering"):
        if not isinstance(args.use_score_centering, bool):
            raise ValueError("YAML use_score_centering must be a boolean")
        if args.enable_score_centering and not args.use_score_centering:
            raise ValueError("Conflicting CLI/YAML score-centering switches")
        args.enable_score_centering = args.use_score_centering
    if hasattr(args, "score_centering_top_k"):
        args.sc_top_k = args.score_centering_top_k
    if not isinstance(args.enable_score_centering, bool):
        raise ValueError("YAML enable_score_centering must be a boolean")
    if getattr(args, "enable_score_centering", False):
        from slime.utils.score_centering_validate import validate_score_centering_args
        validate_score_centering_args(args)
```

### slime/utils/arguments.py： 角色覆盖后二次校验并关闭 critic SC

**Before**

<!-- SC-SPEC {"path":"slime/utils/arguments.py","kind":"before","id":"slime/utils/arguments.py#2","sha256":"63001174b1c5ca8c1955bac4918bae7ca5f8df34f7e65f3aa8cb2286180d3aaf"} -->
```python
    role_args = _apply_megatron_role_overrides(base_args, overrides, role)
    logger.info(
```

**After**

<!-- SC-SPEC {"path":"slime/utils/arguments.py","kind":"after","id":"slime/utils/arguments.py#2","sha256":"63001174b1c5ca8c1955bac4918bae7ca5f8df34f7e65f3aa8cb2286180d3aaf"} -->
```python
    role_args = _apply_megatron_role_overrides(base_args, overrides, role)
    if getattr(base_args, "enable_score_centering", False) or getattr(role_args, "enable_score_centering", False):
        for name in overrides:
            if name.startswith("sc_") or name in {
                "use_score_centering", "enable_score_centering", "score_centering_top_k"
            }:
                raise ValueError("SC configuration belongs to the common rollout/actor config")
        if role == "critic":
            role_args.enable_score_centering = False
            if hasattr(role_args, "use_score_centering"):
                role_args.use_score_centering = False
        else:
            from slime.utils.score_centering_validate import validate_score_centering_args
            validate_score_centering_args(role_args)
    logger.info(
```

### slime/utils/arguments.py： 新增单 trial 累计 head 预算

**Before**

<!-- SC-SPEC {"path":"slime/utils/arguments.py","kind":"before","id":"slime/utils/arguments.py#3","sha256":"63001174b1c5ca8c1955bac4918bae7ca5f8df34f7e65f3aa8cb2286180d3aaf"} -->
```python
            parser.add_argument("--sc-max-batch-bytes", type=int, default=8589934592)
```

**After**

<!-- SC-SPEC {"path":"slime/utils/arguments.py","kind":"after","id":"slime/utils/arguments.py#3","sha256":"63001174b1c5ca8c1955bac4918bae7ca5f8df34f7e65f3aa8cb2286180d3aaf"} -->
```python
            parser.add_argument("--sc-max-batch-bytes", type=int, default=8589934592)
            parser.add_argument("--sc-max-trial-bytes", type=int, default=536870912)
```

文件：`slime/utils/score_centering_validate.py`。以下完整替换模块（已有早期版本，不是新增空文件）；不保留旧函数的并行实现。

<!-- SC-SPEC {"path":"slime/utils/score_centering_validate.py","kind":"full","id":"slime/utils/score_centering_validate.py#full","sha256":"a6060f2471781380760aa2a834497b95293c05acf9148567f44612e3e3d71f6c"} -->
```python
"""Validation runs only after SC is explicitly enabled."""
import math


def validate_score_centering_args(args):
    if not getattr(args, "enable_score_centering", False):
        return
    if getattr(args, "use_score_centering", True) is not True:
        raise ValueError("Conflicting SC switches")
    if args.sc_formula not in {"score", "ppo_weighted"}:
        raise ValueError("Unknown SC formula")
    if isinstance(args.sc_top_k, bool) or not isinstance(args.sc_top_k, int) or not 1 <= args.sc_top_k <= 1024:
        raise ValueError("SC top-k must lie in [1,1024]")
    vocab = getattr(args, "vocab_size", None)
    if vocab is not None and args.sc_top_k > vocab:
        raise ValueError("SC top-k exceeds vocabulary size")
    if args.sc_tail_eps != 1e-8:
        raise ValueError("This integration fixes SC tail epsilon at 1e-8")
    if not math.isfinite(args.sc_correction_scale) or args.sc_correction_scale < 0:
        raise ValueError("SC scale must be finite and nonnegative")
    if not isinstance(args.sc_chunk_size, int) or args.sc_chunk_size < 1:
        raise ValueError("SC chunk size must be positive")
    for name in ("sc_max_response_bytes", "sc_max_batch_bytes", "sc_max_trial_bytes"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if args.sc_wire_format not in {"binary", "json"}:
        raise ValueError("SC wire format must be binary or json")
    if args.sc_assert_alignment != "full":
        raise ValueError("Production SC requires full alignment checks")
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("SC requires Megatron")
    if args.advantage_estimator != "ppo":
        raise ValueError("This integration requires PPO advantages")
    if args.qkv_format not in {"thd", "bshd"}:
        raise ValueError("Unknown QKV layout")
    if args.allgather_cp and args.qkv_format != "thd":
        raise ValueError("SC allgather-CP requires THD, matching get_responses")
    if args.loss_type == "custom_loss":
        expected = "slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled"
        if args.custom_loss_function_path != expected:
            raise ValueError("SC custom loss must use the audited Harbor mount")
    elif args.loss_type != "policy_loss":
        raise ValueError("SC requires a policy loss")
    if getattr(args, "rollout_function_path", None) != "slime_plugins.harbor.rollout.generate_rollout":
        raise ValueError("SC data transport in this port requires Harbor rollout")
    if getattr(args, "harbor_sample_mode", "turn") not in {
        "trajectory_normalized", "trajectory_retokenized"
    }:
        raise ValueError("SC requires an audited normalized Harbor trajectory builder")
    if not getattr(args, "harbor_fail_on_conversion_error", True):
        raise ValueError("SC conversion errors must terminate the rollout")
    for name in ("use_tis", "get_mismatch_metrics", "use_opsm", "use_opd", "use_unbiased_kl", "enable_mtp_training"):
        if getattr(args, name, False):
            raise ValueError(f"SC does not support {name}")
    for name in ("custom_tis_function_path", "custom_pg_loss_reducer_function_path",
                 "custom_convert_samples_to_train_data_path", "rollout_logprob_server_url",
                 "rollout_sample_filter_path", "rollout_sample_postprocess_path"):
        if getattr(args, name, None):
            raise ValueError(f"SC requires an audited adapter for {name}")
    permitted_hooks = {
        "rollout_data_postprocess_path":
            "slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data",
        "rollout_all_samples_process_path":
            "slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples",
    }
    for name, supported in permitted_hooks.items():
        if getattr(args, name, None) not in (None, supported):
            raise ValueError(f"SC only supports the audited token-cap hook for {name}")
    if not getattr(args, "compute_advantages_and_returns", True):
        raise ValueError("SC requires the standard actor advantage/old-policy lifecycle")
    if getattr(args, "eps_clip_c", None) is not None:
        raise ValueError("SC does not support dual-clip")
    high = args.eps_clip if args.eps_clip_high is None else args.eps_clip_high
    if not (math.isfinite(args.eps_clip) and 0 <= args.eps_clip < 1):
        raise ValueError("SC requires finite 0 <= eps_clip < 1")
    if not math.isfinite(high) or high < 0:
        raise ValueError("SC requires finite nonnegative eps_clip_high")
    if args.rollout_temperature != 1 or args.rollout_top_p != 1 or args.rollout_top_k != -1:
        raise ValueError("This SC transport requires temperature=1, top_p=1, top_k=-1")
    if args.sc_formula == "ppo_weighted":
        if args.use_rollout_logprobs:
            if args.sc_weight_denominator != "sampler":
                raise ValueError("PPO denominator is sampler; select --sc-weight-denominator sampler")
        elif args.sc_weight_denominator != "old_actor" or not args.keep_old_actor:
            raise ValueError("PPO old-actor denominator requires old_actor SC and --keep-old-actor")
```

## 4.2 纯算法工具层（无 DiskTensorRef）

文件：`slime/utils/score_centering.py`。以下完整替换模块（已有早期版本，不是新增空文件）；不保留旧函数的并行实现。

<!-- SC-SPEC {"path":"slime/utils/score_centering.py","kind":"full","id":"slime/utils/score_centering.py#full","sha256":"5a432ee0436a64270d66ebee0e32f3d6467c3d84df79ee49f3c439a49e6f0b18"} -->
```python
"""PPO plus score centering; no disk-backed tensor dependency.

All sampler/old-policy coefficients are stop-gradient. The top-k log-probability
operator normalizes over the complete TP-sharded vocabulary.
"""
from __future__ import annotations

import base64
import hashlib
import math

import numpy as np
import torch
import torch.distributed as dist


HEAD_KEYS = (
    "rollout_topk_token_ids",
    "rollout_topk_log_probs",
    "sc_response_token_ids",
)
EPS = 1e-8


def _fp(tensor):
    return tensor if tensor.dtype == torch.float64 else tensor.float()


def validate_head(ids, logps, tokens, *, k, vocab_size=None):
    ids, logps, tokens = np.asarray(ids), np.asarray(logps), np.asarray(tokens)
    if ids.dtype != np.dtype("<i4") or logps.dtype != np.dtype("<f4"):
        raise ValueError("SC heads require little-endian int32/float32")
    if tokens.dtype != np.dtype("<i4") or tokens.ndim != 1:
        raise ValueError("SC row token IDs require a one-dimensional int32 array")
    if ids.shape != (len(tokens), k) or logps.shape != ids.shape:
        raise ValueError("SC head rows/width do not match the response token IDs")
    if k < 1:
        raise ValueError("SC k must be positive")
    for start in range(0, len(tokens), 1024):
        a = ids[start:start + 1024]
        b = logps[start:start + 1024].astype(np.float64)
        t = tokens[start:start + 1024]
        if (a < 0).any() or (t < 0).any():
            raise ValueError("SC token IDs must be nonnegative")
        if vocab_size is not None and ((a >= vocab_size).any() or (t >= vocab_size).any()):
            raise ValueError("SC token ID exceeds the actual vocabulary")
        if (np.diff(np.sort(a, axis=-1), axis=-1) == 0).any():
            raise ValueError("SC head contains duplicate token IDs")
        if not np.isfinite(b).all() or (b > 0).any():
            raise ValueError("SC head requires finite, nonpositive log probabilities")
        if (np.exp(b).sum(-1) > 1.0001).any():
            raise ValueError("SC head is not a probability subdistribution")
    return ids, logps, tokens


def _decode(encoded, dtype, shape):
    if not isinstance(encoded, str):
        raise ValueError("SC binary field must be a base64 string")
    size = math.prod(shape) * np.dtype(dtype).itemsize
    if len(encoded) != 4 * ((size + 2) // 3):
        raise ValueError("SC base64 length does not match declared shape")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != size:
        raise ValueError("SC decoded byte length does not match declared shape")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def decode_sampler_head(meta, tokens, *, k, wire_format, max_bytes, vocab_size):
    tokens = np.asarray(tokens, dtype="<i4")
    count = len(tokens)
    if count * (8 * k + 4) > max_bytes:
        raise ValueError("SC response exceeds the configured head byte budget")
    binary = ("output_topk_token_ids", "output_topk_log_probs", "output_topk_shape")
    if wire_format == "binary":
        if any(key not in meta for key in binary):
            raise ValueError("SC binary response missing; use the verified SC SGLang runtime")
        if meta[binary[2]] != [count, k]:
            raise ValueError("SGLang SC shape differs from output_token_logprobs")
        ids = _decode(meta[binary[0]], "<i4", (count, k))
        logps = _decode(meta[binary[1]], "<f4", (count, k))
    elif wire_format == "json":
        rows = meta.get("output_top_logprobs")
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError("SGLang JSON head row count mismatch")
        ids = np.empty((count, k), dtype="<i4")
        logps = np.empty((count, k), dtype="<f4")
        for i, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != k:
                raise ValueError(f"SGLang JSON head width mismatch at row {i}")
            for j, item in enumerate(row):
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    raise ValueError("Invalid SGLang JSON top-logprob entry")
                lp, token = item[:2]
                if isinstance(token, bool) or not isinstance(token, int):
                    raise ValueError("SGLang head token IDs must be integers")
                if not 0 <= token < vocab_size:
                    raise ValueError("SGLang head token ID is outside the vocabulary")
                ids[i, j], logps[i, j] = token, lp
    else:
        raise ValueError("SC wire format must be binary or json")
    return validate_head(ids, logps, tokens, k=k, vocab_size=vocab_size)


def assert_sampled_logps(ids, logps, tokens, sampled_logps, mask=None):
    expected = np.asarray(sampled_logps, dtype=np.float64)
    if expected.shape != (len(tokens),) or not np.isfinite(expected).all():
        raise ValueError("SC requires finite sampled-token logprobs for every response row")
    trainable = np.ones(len(tokens), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if trainable.shape != expected.shape:
        raise ValueError("SC loss mask shape mismatch")
    for start in range(0, len(tokens), 1024):
        stop = start + 1024
        matches = ids[start:stop] == tokens[start:stop, None]
        hit = matches.any(-1) & trainable[start:stop]
        if hit.any():
            column = matches.argmax(-1)
            actual = np.take_along_axis(logps[start:stop], column[:, None], axis=1)[:, 0]
            if not np.allclose(actual[hit], expected[start:stop][hit], rtol=0, atol=1e-4):
                raise ValueError("SC head/sampled-logprob cross-check failed")


def encode_turn_head(ids, logps, tokens, prompt_ids):
    def b64(value):
        return base64.b64encode(np.ascontiguousarray(value).tobytes()).decode("ascii")
    prompt = np.asarray(prompt_ids, dtype="<i4")
    digest = hashlib.sha256()
    for value in (prompt, tokens, ids, logps):
        digest.update(np.ascontiguousarray(value).tobytes())
    return {
        "version": 1,
        "shape": list(ids.shape),
        "ids": b64(ids),
        "logps": b64(logps),
        "tokens": b64(tokens),
        "digest": digest.hexdigest(),
    }


def decode_turn_head(envelope, prompt_ids, output_ids, sampled_logps, *, k):
    if not isinstance(envelope, dict) or envelope.get("version") != 1:
        raise ValueError("Harbor turn is missing its SC envelope")
    count = len(output_ids)
    if envelope.get("shape") != [count, k]:
        raise ValueError("Harbor SC envelope shape mismatch")
    ids = _decode(envelope.get("ids"), "<i4", (count, k))
    logps = _decode(envelope.get("logps"), "<f4", (count, k))
    tokens = _decode(envelope.get("tokens"), "<i4", (count,))
    validate_head(ids, logps, tokens, k=k)
    if not np.array_equal(tokens, np.asarray(output_ids, dtype="<i4")):
        raise ValueError("Harbor SC row token IDs differ from completion_token_ids")
    expected = encode_turn_head(ids, logps, tokens, prompt_ids)["digest"]
    if envelope.get("digest") != expected:
        raise ValueError("Harbor SC prompt/token/head digest mismatch")
    assert_sampled_logps(ids, logps, tokens, sampled_logps)
    return ids, logps, tokens


def score_centering_request(*, k, sampling_params, max_bytes):
    if k < 1:
        raise ValueError("SC k must be positive")
    required = {
        "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
        "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
    }
    for key, value in required.items():
        if sampling_params.get(key, value) != value:
            raise ValueError(f"SC requires actual request {key}={value}")
    for key in ("json_schema", "regex", "ebnf", "structural_tag", "logit_bias", "custom_params"):
        if sampling_params.get(key):
            raise ValueError(f"SC does not support constrained sampling: {key}")
    count = sampling_params.get("max_new_tokens")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("SC requires a positive integer max_new_tokens")
    if count * (8 * k + 4) > max_bytes:
        raise ValueError("SC request exceeds the configured head byte budget")
    return {"top_logprobs_num": k, "return_text_in_logprobs": False}


def ppo_effective_weight(ratio, adv_sign, eps_clip_low, eps_clip_high):
    killed = ((adv_sign > 0) & (ratio > 1 + eps_clip_high)) | (
        (adv_sign < 0) & (ratio < 1 - eps_clip_low)
    )
    return torch.where(killed, torch.zeros_like(ratio), ratio)


def score_centering_correction(
    *, train_head_logp, sampler_head_logp, advantages, formula="score",
    old_head_logp=None, eps_clip_low=0.2, eps_clip_high=0.2,
    tail_eps=EPS, loss_mask=None,
):
    if formula not in {"score", "ppo_weighted"}:
        raise ValueError("Unknown SC formula")
    if not math.isfinite(tail_eps) or not 0 < tail_eps < 1:
        raise ValueError("SC tail epsilon must lie in (0, 1)")
    lp = _fp(train_head_logp)
    if lp.ndim != 2 or sampler_head_logp.shape != lp.shape:
        raise ValueError("SC requires matching [tokens, k] head tensors")
    if advantages.shape != lp.shape[:1]:
        raise ValueError("SC advantage shape mismatch")
    valid = torch.ones(lp.size(0), dtype=torch.bool, device=lp.device)
    if loss_mask is not None:
        if loss_mask.shape != advantages.shape:
            raise ValueError("SC mask shape mismatch")
        valid = loss_mask.to(device=lp.device, dtype=torch.bool)
    # Sanitize masked rows BEFORE exp, ratio, and multiplication.
    safe_lp = torch.where(valid[:, None], lp, torch.zeros_like(lp))
    with torch.no_grad():
        qlp = torch.where(valid[:, None], _fp(sampler_head_logp.detach()), torch.zeros_like(lp))
        if not torch.isfinite(safe_lp).all() or not torch.isfinite(qlp).all():
            raise ValueError("Nonfinite unmasked SC head")
        p, q = safe_lp.detach().exp(), qlp.exp()
        p = torch.where(valid[:, None], p, torch.zeros_like(p))
        q = torch.where(valid[:, None], q, torch.zeros_like(q))
        pm, qm = p.sum(-1), q.sum(-1)
        if ((pm > 1.0001) | (qm > 1.0001) | (p > 1.0001).any(-1) | (q > 1.0001).any(-1)).any():
            raise ValueError("SC inputs must be full-vocabulary-normalized logprobs")
        pt = (1 - pm).clamp_min(tail_eps)
        qt = (1 - qm).clamp_min(0)
        rho = qt / pt
        alpha = rho
        weight = torch.ones_like(p)
        if formula == "ppo_weighted":
            denominator = qlp if old_head_logp is None else _fp(old_head_logp.detach())
            if denominator.shape != lp.shape:
                raise ValueError("SC old-policy head shape mismatch")
            denominator = torch.where(valid[:, None], denominator, torch.zeros_like(denominator))
            if not torch.isfinite(denominator).all():
                raise ValueError("Nonfinite old-policy SC head")
            old = denominator.exp()
            old = torch.where(valid[:, None], old, torch.zeros_like(old))
            if ((old.sum(-1) > 1.0001) | (old > 1.0001).any(-1)).any():
                raise ValueError("Invalid old-policy probability subdistribution")
            ratio = (safe_lp.detach() - denominator).exp()
            sign = advantages.detach().sign()
            weight = ppo_effective_weight(ratio, sign[:, None], eps_clip_low, eps_clip_high)
            old_tail = (1 - old.sum(-1)).clamp_min(tail_eps)
            tail_weight = ppo_effective_weight(pt / old_tail, sign, eps_clip_low, eps_clip_high)
            alpha = rho * tail_weight
        coefficient = (q * weight - alpha[:, None] * p).detach()
        coefficient = torch.where(valid[:, None], coefficient, torch.zeros_like(coefficient)).detach()
        if not torch.isfinite(coefficient).all():
            raise ValueError("SC ratio/coefficient overflow; abort rather than silently clip it")
        metrics = {
            "sc_sampler_head_mass": qm.detach(),
            "sc_train_head_mass": pm.detach(),
            "sc_alpha": torch.where(valid, alpha, torch.zeros_like(alpha)).detach(),
            "sc_coefficient_l1": coefficient.abs().sum(-1).detach(),
        }
    correction = (coefficient.detach() * safe_lp).sum(-1)
    return correction, metrics


def _tp_info(group):
    if group is None:
        return 0, 1
    if not dist.is_initialized():
        raise RuntimeError("A TP group was supplied before distributed initialization")
    return dist.get_rank(group), dist.get_world_size(group)


def _reduce(tensor, operation, group):
    if group is not None and dist.get_world_size(group) > 1:
        dist.all_reduce(tensor, op=operation, group=group)


class _HeadLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, row_ids, token_ids, group, chunk_size):
        rank, world = _tp_info(group)
        n, width = token_ids.shape
        vp = logits.size(1)
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        result = torch.empty((n, width), dtype=dtype, device=logits.device)
        maximums = torch.empty((n, 1), dtype=dtype, device=logits.device)
        denominators = torch.empty_like(maximums)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            work = logits.index_select(0, row_ids[start:end]).to(dtype=dtype)
            maximum = work.max(-1, keepdim=True).values
            _reduce(maximum, dist.ReduceOp.MAX, group)
            ids = token_ids[start:end]
            local = (ids >= rank * vp) & (ids < (rank + 1) * vp)
            relative = (ids - rank * vp).clamp(0, vp - 1)
            selected = work.gather(-1, relative).masked_fill(~local, 0)
            _reduce(selected, dist.ReduceOp.SUM, group)
            work.sub_(maximum).exp_()
            denominator = work.sum(-1, keepdim=True)
            _reduce(denominator, dist.ReduceOp.SUM, group)
            result[start:end] = selected - maximum - denominator.log()
            maximums[start:end], denominators[start:end] = maximum, denominator
        ctx.save_for_backward(logits, row_ids, token_ids, maximums, denominators)
        ctx.chunk_size, ctx.vocab_start = chunk_size, rank * vp
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        logits, row_ids, token_ids, maximums, denominators = ctx.saved_tensors
        grad_input = torch.zeros_like(logits)
        vp = logits.size(1)
        dtype = maximums.dtype
        for start in range(0, len(row_ids), ctx.chunk_size):
            end = min(start + ctx.chunk_size, len(row_ids))
            index = row_ids[start:end]
            work = logits.index_select(0, index).to(dtype=dtype)
            work.sub_(maximums[start:end]).exp_().div_(denominators[start:end])
            grad = grad_output[start:end].to(dtype=dtype)
            work.mul_(-grad.sum(-1, keepdim=True))
            ids = token_ids[start:end]
            local = (ids >= ctx.vocab_start) & (ids < ctx.vocab_start + vp)
            relative = (ids - ctx.vocab_start).clamp(0, vp - 1)
            work.scatter_add_(-1, relative, grad.masked_fill(~local, 0))
            grad_input.index_copy_(0, index, work.to(dtype=logits.dtype))
        # The loss is replicated over TP; another all-reduce would multiply its gradient.
        return grad_input, None, None, None, None


def calculate_head_log_probs(logits, row_ids, token_ids, group=None, chunk_size=256):
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("SC expects flattened floating logits [T, V/TP]")
    if row_ids.dtype != torch.long or row_ids.ndim != 1:
        raise ValueError("SC packed row indices must be int64 [N]")
    if token_ids.dtype != torch.long or token_ids.ndim != 2 or token_ids.size(0) != len(row_ids):
        raise ValueError("SC candidate IDs must be int64 [N,k]")
    if row_ids.device != logits.device or token_ids.device != logits.device:
        raise ValueError("SC gather indices and logits must share a device")
    if chunk_size < 1:
        raise ValueError("SC chunk size must be positive")
    rank, world = _tp_info(group)
    if ((row_ids < 0) | (row_ids >= logits.size(0))).any():
        raise ValueError("SC packed row index out of bounds")
    if torch.unique(row_ids).numel() != row_ids.numel():
        raise ValueError("SC packed rows must be unique")
    if ((token_ids < 0) | (token_ids >= world * logits.size(1))).any():
        raise ValueError("SC global candidate token ID out of bounds")
    return _HeadLogProbs.apply(logits, row_ids, token_ids, group, chunk_size)
```

旧 `slime/utils/ppo_utils.py` 中已存在的早期 calculate_topk_log_probs 保留原样，新SC调用本文calculate_head_log_probs。后者一次作用于packed logits，避免逐sample get_responses产生大词表tensor副本；没有要求重写baseline PPO工具。

## 4.3 通信、SGLang 与采样层

### slime_plugins/harbor/llm.py： SlimeSGLangLLM.__init__ 签名

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"before","id":"slime_plugins/harbor/llm.py#0","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
        timeout_sec: float = 600.0,
        **_: Any,
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"after","id":"slime_plugins/harbor/llm.py#0","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
        timeout_sec: float = 600.0,
        score_centering_top_k: int | None = None,
        score_centering_wire_format: str = "binary",
        score_centering_max_response_bytes: int = 268435456,
        **_: Any,
```

### slime_plugins/harbor/llm.py： SlimeSGLangLLM.__init__ SC-only 属性

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"before","id":"slime_plugins/harbor/llm.py#1","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
        self._timeout_sec = timeout_sec
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"after","id":"slime_plugins/harbor/llm.py#1","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
        self._timeout_sec = timeout_sec
        if score_centering_top_k is not None:
            if not isinstance(score_centering_top_k, int) or score_centering_top_k < 1:
                raise ValueError("score_centering_top_k must be positive")
            self._score_centering_top_k = score_centering_top_k
            self._score_centering_wire_format = score_centering_wire_format
            self._score_centering_max_response_bytes = score_centering_max_response_bytes
```

### slime_plugins/harbor/llm.py： SlimeSGLangLLM.call 请求注入

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"before","id":"slime_plugins/harbor/llm.py#2","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
        if self._use_rollout_routing_replay:
            payload["return_routed_experts"] = True
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"after","id":"slime_plugins/harbor/llm.py#2","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
        if hasattr(self, "_score_centering_top_k"):
            from slime.utils.score_centering import score_centering_request
            payload.update(score_centering_request(
                k=self._score_centering_top_k, sampling_params=sampling_params,
                max_bytes=self._score_centering_max_response_bytes,
            ))
        if self._use_rollout_routing_replay:
            payload["return_routed_experts"] = True
```

### slime_plugins/harbor/llm.py： SlimeSGLangLLM.call 响应解码与 JSON-safe extra

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"before","id":"slime_plugins/harbor/llm.py#3","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
        extra = {"finish_reason": meta_info.get("finish_reason")}
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"after","id":"slime_plugins/harbor/llm.py#3","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
        extra = {"finish_reason": meta_info.get("finish_reason")}
        if hasattr(self, "_score_centering_top_k"):
            from slime.utils.score_centering import (
                assert_sampled_logps, decode_sampler_head, encode_turn_head,
            )
            if not completion_token_ids:
                raise RuntimeError("SC rejects empty LLM completions before Harbor loses turn alignment")
            ids, head_logps, row_tokens = decode_sampler_head(
                meta_info, completion_token_ids, k=self._score_centering_top_k,
                wire_format=self._score_centering_wire_format,
                max_bytes=self._score_centering_max_response_bytes,
                vocab_size=len(self._tokenizer),
            )
            assert_sampled_logps(ids, head_logps, row_tokens, logprobs)
            extra["score_centering_head"] = encode_turn_head(
                ids, head_logps, row_tokens, prompt_token_ids
            )
```

### slime_plugins/harbor/llm.py： SlimeSGLangLLM 单实例累计预算配置

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"before","id":"slime_plugins/harbor/llm.py#4","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
        score_centering_max_response_bytes: int = 268435456,
        **_: Any,
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"after","id":"slime_plugins/harbor/llm.py#4","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
        score_centering_max_response_bytes: int = 268435456,
        score_centering_max_trial_bytes: int = 536870912,
        **_: Any,
```

### slime_plugins/harbor/llm.py： SlimeSGLangLLM.__init__ SC-only 计数器

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"before","id":"slime_plugins/harbor/llm.py#5","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
            self._score_centering_max_response_bytes = score_centering_max_response_bytes
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"after","id":"slime_plugins/harbor/llm.py#5","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
            self._score_centering_max_response_bytes = score_centering_max_response_bytes
            self._score_centering_max_trial_bytes = score_centering_max_trial_bytes
            self._score_centering_reserved_bytes = 0
```

### slime_plugins/harbor/llm.py： SlimeSGLangLLM.call 请求前累计预算

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"before","id":"slime_plugins/harbor/llm.py#6","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
            payload.update(score_centering_request(
                k=self._score_centering_top_k, sampling_params=sampling_params,
                max_bytes=self._score_centering_max_response_bytes,
            ))
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/llm.py","kind":"after","id":"slime_plugins/harbor/llm.py#6","sha256":"2bb92513b6374f107fcf4f43443156b6f5e3e0f18ab709041f4d5e778a94fe67"} -->
```python
            payload.update(score_centering_request(
                k=self._score_centering_top_k, sampling_params=sampling_params,
                max_bytes=self._score_centering_max_response_bytes,
            ))
            reserve = sampling_params["max_new_tokens"] * (8 * self._score_centering_top_k + 4)
            if self._score_centering_reserved_bytes + reserve > self._score_centering_max_trial_bytes:
                raise RuntimeError("SC trial head budget exceeded before generation")
            # Reserve before the first await. Failed/cancelled calls conservatively keep their reservation.
            self._score_centering_reserved_bytes += reserve
```

### slime_plugins/harbor/rollout.py： _set_default_agent_config 注入唯一配置来源

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/rollout.py","kind":"before","id":"slime_plugins/harbor/rollout.py#0","sha256":"920f584cc12b8eb67db298d7e1bb5b7ecb7f35808faa763f25b7f6b8a08d92a6"} -->
```python
    llm_kwargs.setdefault("router_url", _router_url(args))
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/rollout.py","kind":"after","id":"slime_plugins/harbor/rollout.py#0","sha256":"920f584cc12b8eb67db298d7e1bb5b7ecb7f35808faa763f25b7f6b8a08d92a6"} -->
```python
    llm_kwargs.setdefault("router_url", _router_url(args))
    if getattr(args, "enable_score_centering", False) and not _is_eval_args(args):
        from slime.utils.score_centering_validate import validate_score_centering_args
        validate_score_centering_args(args)
        required_sc = {
            "score_centering_top_k": args.sc_top_k,
            "score_centering_wire_format": args.sc_wire_format,
            "score_centering_max_response_bytes": args.sc_max_response_bytes,
        }
        for key, value in required_sc.items():
            if key in llm_kwargs and llm_kwargs[key] != value:
                raise ValueError(f"Harbor llm_kwargs conflicts with SC configuration: {key}")
            llm_kwargs[key] = value
        if kwargs.get("collect_rollout_details") is not True:
            raise ValueError("SC requires collect_rollout_details=True")
        if kwargs.get("llm_backend") != "slime_sglang":
            raise ValueError("SC requires Harbor slime_sglang backend")
```

### slime_plugins/harbor/rollout.py： _run_one_trial 条件转换参数

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/rollout.py","kind":"before","id":"slime_plugins/harbor/rollout.py#1","sha256":"920f584cc12b8eb67db298d7e1bb5b7ecb7f35808faa763f25b7f6b8a08d92a6"} -->
```python
    try:
        samples = samples_from_trial_result(
            result,
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/rollout.py","kind":"after","id":"slime_plugins/harbor/rollout.py#1","sha256":"920f584cc12b8eb67db298d7e1bb5b7ecb7f35808faa763f25b7f6b8a08d92a6"} -->
```python
    sc_conversion_kwargs = {}
    if getattr(args, "enable_score_centering", False) and not _is_eval_args(args):
        sc_conversion_kwargs["score_centering_top_k"] = args.sc_top_k
    try:
        samples = samples_from_trial_result(
            result,
```

### slime_plugins/harbor/rollout.py： _run_one_trial 传递转换开关

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/rollout.py","kind":"before","id":"slime_plugins/harbor/rollout.py#2","sha256":"920f584cc12b8eb67db298d7e1bb5b7ecb7f35808faa763f25b7f6b8a08d92a6"} -->
```python
            require_routing_replay=bool(getattr(args, "use_rollout_routing_replay", False)),
        )
    except RolloutConversionError:
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/rollout.py","kind":"after","id":"slime_plugins/harbor/rollout.py#2","sha256":"920f584cc12b8eb67db298d7e1bb5b7ecb7f35808faa763f25b7f6b8a08d92a6"} -->
```python
            require_routing_replay=bool(getattr(args, "use_rollout_routing_replay", False)),
            **sc_conversion_kwargs,
        )
    except RolloutConversionError:
```

### slime_plugins/harbor/rollout.py： _run_one_group SC 数据错误禁止被通用异常吞掉

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/rollout.py","kind":"before","id":"slime_plugins/harbor/rollout.py#3","sha256":"920f584cc12b8eb67db298d7e1bb5b7ecb7f35808faa763f25b7f6b8a08d92a6"} -->
```python
                except Exception as exc:
                    stats["failed"] += 1
                    logger.exception("Harbor trial failed for sample index %s", base_sample.index)
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/rollout.py","kind":"after","id":"slime_plugins/harbor/rollout.py#3","sha256":"920f584cc12b8eb67db298d7e1bb5b7ecb7f35808faa763f25b7f6b8a08d92a6"} -->
```python
                except Exception as exc:
                    if getattr(args, "enable_score_centering", False):
                        # A missing/malformed head must not silently turn into dropped training data.
                        raise
                    stats["failed"] += 1
                    logger.exception("Harbor trial failed for sample index %s", base_sample.index)
```

### slime_plugins/harbor/rollout.py： _set_default_agent_config 透传 trial 预算

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/rollout.py","kind":"before","id":"slime_plugins/harbor/rollout.py#4","sha256":"920f584cc12b8eb67db298d7e1bb5b7ecb7f35808faa763f25b7f6b8a08d92a6"} -->
```python
            "score_centering_max_response_bytes": args.sc_max_response_bytes,
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/rollout.py","kind":"after","id":"slime_plugins/harbor/rollout.py#4","sha256":"920f584cc12b8eb67db298d7e1bb5b7ecb7f35808faa763f25b7f6b8a08d92a6"} -->
```python
            "score_centering_max_response_bytes": args.sc_max_response_bytes,
            "score_centering_max_trial_bytes": args.sc_max_trial_bytes,
```

### 当前 SGLang 的最小完整替换

目标为**未来完整shadow内**的 `sglang/srt/managers/tokenizer_manager.py`；live路径 `/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py`。修改其top-logprobs处理函数内以下连续块。当前文件SHA256：e3d7144afbe26f1415d7b0126ac1deaf04b99cf2608f750be8fa166f5f59565e。import numpy仅SC响应分支发生，复用原pybase64；不修改top_p.patch、不改采样器。

**Before**

```python
        # 2. Handle top logprobs
        if top_logprobs_num > 0:
            if len(state.input_top_logprobs_val) > len(state.input_top_logprobs):
                state.input_top_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.input_top_logprobs_val[len(state.input_top_logprobs) :],
                        state.input_top_logprobs_idx[len(state.input_top_logprobs) :],
                        return_text_in_logprobs,
                    )
                )
            if len(state.output_top_logprobs_val) > len(state.output_top_logprobs):
                state.output_top_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.output_top_logprobs_val[len(state.output_top_logprobs) :],
                        state.output_top_logprobs_idx[len(state.output_top_logprobs) :],
                        return_text_in_logprobs,
                    )
                )

            meta_info["input_top_logprobs"] = state.input_top_logprobs
            meta_info["output_top_logprobs"] = state.output_top_logprobs

```

**After**

```python
        # 2. Handle top logprobs
        if top_logprobs_num > 0:
            if len(state.input_top_logprobs_val) > len(state.input_top_logprobs):
                state.input_top_logprobs.extend(
                    self.detokenize_top_logprobs_tokens(
                        state.input_top_logprobs_val[len(state.input_top_logprobs) :],
                        state.input_top_logprobs_idx[len(state.input_top_logprobs) :],
                        return_text_in_logprobs,
                    )
                )
            meta_info["input_top_logprobs"] = state.input_top_logprobs
            if not return_text_in_logprobs and not state.obj.stream:
                import numpy as np
                ids = np.asarray(state.output_top_logprobs_idx, dtype="<i4").reshape(-1, top_logprobs_num)
                logps = np.asarray(state.output_top_logprobs_val, dtype="<f4").reshape(-1, top_logprobs_num)
                if ids.shape != logps.shape:
                    raise ValueError("SC output head shapes differ")
                meta_info["output_topk_token_ids"] = pybase64.b64encode(ids.tobytes()).decode("ascii")
                meta_info["output_topk_log_probs"] = pybase64.b64encode(logps.tobytes()).decode("ascii")
                meta_info["output_topk_shape"] = list(ids.shape)
            else:
                if len(state.output_top_logprobs_val) > len(state.output_top_logprobs):
                    state.output_top_logprobs.extend(
                        self.detokenize_top_logprobs_tokens(
                            state.output_top_logprobs_val[len(state.output_top_logprobs) :],
                            state.output_top_logprobs_idx[len(state.output_top_logprobs) :],
                            return_text_in_logprobs,
                        )
                    )
                meta_info["output_top_logprobs"] = state.output_top_logprobs

```

### 未来影子构建工具完整源码

未来相对路径 `tools/build_sglang_sc_shadow.py`；本阶段只存在于本文。replacement参数指向上方After块的UTF-8文本文件，包含原8空格缩进。脚本备份diff HEAD并复制完整包；不修改live或重建镜像。

```python
"""Build an immutable complete SGLang shadow; never patch the running checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/sgl-workspace/sglang")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--replacement", required=True,
                        help="UTF-8 text containing the complete After block from section 4.3")
    args = parser.parse_args()
    source, destination = Path(args.source).resolve(), Path(args.destination).resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise SystemExit("Destination must be a new directory outside the live checkout")
    package = source / "python/sglang"
    target = package / "srt/managers/tokenizer_manager.py"
    original = target.read_text()
    expected = "e3d7144afbe26f1415d7b0126ac1deaf04b99cf2608f750be8fa166f5f59565e"
    if sha(target.read_bytes()) != expected:
        raise SystemExit("Live SGLang source drifted; regenerate the reviewed replacement")
    begin = original.index("        # 2. Handle top logprobs\n")
    end = original.index("        # 3. Handle token_ids_logprob\n", begin)
    replacement = Path(args.replacement).read_text()
    if not replacement.startswith("        # 2. Handle top logprobs\n"):
        raise SystemExit("Replacement must contain the complete reviewed block")
    proposed = original[:begin] + replacement + original[end:]
    compile(proposed, str(target), "exec")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + f".building-{os.getpid()}")
    if staging.exists():
        raise SystemExit("Staging directory already exists")
    staging.mkdir()
    diff = subprocess.check_output(["git", "-C", str(source), "diff", "--binary", "HEAD"])
    (staging / "live-local-changes.patch").write_bytes(diff)
    # Copy tracked, staged and untracked package files; follow symlinks so the
    # shadow never quietly imports a Python submodule from the live checkout.
    shutil.copytree(package, staging / "sglang",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), symlinks=False)
    (staging / "sglang/srt/managers/tokenizer_manager.py").write_text(proposed)
    hashes = {
        str(path.relative_to(staging / "sglang")): sha(path.read_bytes())
        for path in sorted((staging / "sglang").rglob("*")) if path.is_file()
    }
    provenance = {
        "source": str(source),
        "source_head": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "source_tokenizer_sha256": expected,
        "local_changes_sha256": sha(diff),
        "files": hashes, "built_at_unix": time.time(),
    }
    (staging / ".provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    if sha(target.read_bytes()) != expected:
        raise SystemExit("Live source changed during shadow build")
    os.rename(staging, destination)
    print(destination)


if __name__ == "__main__":
    main()
```

## 4.4 数据适配层：真实Token流水与裁切

### slime_plugins/harbor/atif_adapter.py： samples_from_trial_result 签名

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"before","id":"slime_plugins/harbor/atif_adapter.py#0","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
def samples_from_trial_result(
    result: Any,
    base_sample: Sample,
    *,
    reward_key: str = "reward",
    mode: str = "turn",
    include_subagents: bool = False,
    metadata: dict[str, Any] | None = None,
    require_routing_replay: bool = False,
) -> list[Sample]:
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"after","id":"slime_plugins/harbor/atif_adapter.py#0","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
def samples_from_trial_result(
    result: Any,
    base_sample: Sample,
    *,
    reward_key: str = "reward",
    mode: str = "turn",
    include_subagents: bool = False,
    metadata: dict[str, Any] | None = None,
    require_routing_replay: bool = False,
    score_centering_top_k: int | None = None,
) -> list[Sample]:
```

### slime_plugins/harbor/atif_adapter.py： samples_from_rollout_details 签名

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"before","id":"slime_plugins/harbor/atif_adapter.py#1","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
def samples_from_rollout_details(
    details: Iterable[dict[str, Any]],
    base_sample: Sample,
    *,
    reward: float,
    status: Sample.Status,
    mode: str = "turn",
    metadata: dict[str, Any] | None = None,
    require_routing_replay: bool = False,
) -> list[Sample]:
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"after","id":"slime_plugins/harbor/atif_adapter.py#1","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
def samples_from_rollout_details(
    details: Iterable[dict[str, Any]],
    base_sample: Sample,
    *,
    reward: float,
    status: Sample.Status,
    mode: str = "turn",
    metadata: dict[str, Any] | None = None,
    require_routing_replay: bool = False,
    score_centering_top_k: int | None = None,
) -> list[Sample]:
```

### slime_plugins/harbor/atif_adapter.py： _normalized_trajectory_samples_from_detail 签名

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"before","id":"slime_plugins/harbor/atif_adapter.py#2","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
def _normalized_trajectory_samples_from_detail(
    detail: dict[str, Any],
    base_sample: Sample,
    *,
    reward: float,
    status: Sample.Status,
    detail_index: int,
    metadata: dict[str, Any] | None,
    sample_mode: str,
    require_routing_replay: bool = False,
) -> list[Sample]:
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"after","id":"slime_plugins/harbor/atif_adapter.py#2","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
def _normalized_trajectory_samples_from_detail(
    detail: dict[str, Any],
    base_sample: Sample,
    *,
    reward: float,
    status: Sample.Status,
    detail_index: int,
    metadata: dict[str, Any] | None,
    sample_mode: str,
    require_routing_replay: bool = False,
    score_centering_top_k: int | None = None,
) -> list[Sample]:
```

### slime_plugins/harbor/atif_adapter.py： samples_from_trial_result 条件关键字

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"before","id":"slime_plugins/harbor/atif_adapter.py#3","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
    details = extract_rollout_details(result, include_subagents=include_subagents)
    return samples_from_rollout_details(
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"after","id":"slime_plugins/harbor/atif_adapter.py#3","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
    details = extract_rollout_details(result, include_subagents=include_subagents)
    sc_kwargs = {}
    if score_centering_top_k is not None:
        sc_kwargs["score_centering_top_k"] = score_centering_top_k
    return samples_from_rollout_details(
```

### slime_plugins/harbor/atif_adapter.py： samples_from_trial_result 透传

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"before","id":"slime_plugins/harbor/atif_adapter.py#4","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
        require_routing_replay=require_routing_replay,
    )


def explain_trial_result_conversion
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"after","id":"slime_plugins/harbor/atif_adapter.py#4","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
        require_routing_replay=require_routing_replay,
        **sc_kwargs,
    )


def explain_trial_result_conversion
```

### slime_plugins/harbor/atif_adapter.py： samples_from_rollout_details 模式守卫

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"before","id":"slime_plugins/harbor/atif_adapter.py#5","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
    converted: list[Sample] = []
    details = list(details)
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"after","id":"slime_plugins/harbor/atif_adapter.py#5","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
    converted: list[Sample] = []
    sc_kwargs = {}
    if score_centering_top_k is not None:
        if mode not in _NORMALIZED_TRAJECTORY_SAMPLE_MODES:
            raise RolloutConversionError("SC only supports normalized trajectory modes")
        sc_kwargs["score_centering_top_k"] = score_centering_top_k
    details = list(details)
```

### slime_plugins/harbor/atif_adapter.py： samples_from_rollout_details 生产构建器透传

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"before","id":"slime_plugins/harbor/atif_adapter.py#6","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
                    sample_mode=mode,
                    require_routing_replay=require_routing_replay,
                )
            )
        else:
            raise RolloutConversionError
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"after","id":"slime_plugins/harbor/atif_adapter.py#6","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
                    sample_mode=mode,
                    require_routing_replay=require_routing_replay,
                    **sc_kwargs,
                )
            )
        else:
            raise RolloutConversionError
```

### slime_plugins/harbor/atif_adapter.py： _normalized_trajectory_samples_from_detail 验证逐轮 heads

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"before","id":"slime_plugins/harbor/atif_adapter.py#7","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
    finish_reason_types = _finish_reason_types(detail)
    length_finish_count = sum(1 for value in finish_reason_types if value == "length")
    total_output_tokens = sum(len(turn.output_ids) for turn in turns)
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"after","id":"slime_plugins/harbor/atif_adapter.py#7","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
    sc_turn_heads = None
    if score_centering_top_k is not None:
        from slime.utils.score_centering import decode_turn_head
        envelopes = _detail_extra_list(detail, "score_centering_head")
        if len(envelopes) != len(turns):
            raise RolloutConversionError("SC envelope count differs from Harbor turn count")
        sc_turn_heads = [
            decode_turn_head(
                envelope, turn.prompt_ids, turn.output_ids, turn.output_log_probs,
                k=score_centering_top_k,
            )
            for envelope, turn in zip(envelopes, turns, strict=True)
        ]
    finish_reason_types = _finish_reason_types(detail)
    length_finish_count = sum(1 for value in finish_reason_types if value == "length")
    total_output_tokens = sum(len(turn.output_ids) for turn in turns)
```

### slime_plugins/harbor/atif_adapter.py： _normalized_trajectory_samples_from_detail 出口挂载

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"before","id":"slime_plugins/harbor/atif_adapter.py#8","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
        samples.append(sample)

    return samples


def _new_normalized_chunk_stats
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"after","id":"slime_plugins/harbor/atif_adapter.py#8","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python
        if score_centering_top_k is not None:
            _attach_normalized_sc_head(
                sample, chunk, turns, transitions, sc_turn_heads, k=score_centering_top_k
            )
        samples.append(sample)

    return samples


def _new_normalized_chunk_stats
```

### slime_plugins/harbor/atif_adapter.py： 新增完整 _slice_sc_tracks / _attach_normalized_sc_head

<!-- SC-SPEC {"path":"slime_plugins/harbor/atif_adapter.py","kind":"append","id":"slime_plugins/harbor/atif_adapter.py#9","sha256":"41c4f924acc2dd2d07414a6616ee79f7f1daef887c70cfa97b730ad75f82d11d"} -->
```python

def _slice_sc_tracks(tokens, mask, sampled_logps, head_ids, head_logps, row_tokens, start, stop):
    """One slice for every response-aligned track; never infer a slice from drop counters."""
    from slime.utils.score_centering import validate_head
    tokens = np.asarray(tokens, dtype="<i4")
    mask = np.asarray(mask, dtype=np.int32)
    sampled_logps = np.asarray(sampled_logps, dtype=np.float32)
    if not 0 <= start <= stop <= len(tokens):
        raise RolloutConversionError("Invalid SC response slice")
    validate_head(head_ids, head_logps, row_tokens, k=head_ids.shape[1])
    if not np.array_equal(tokens, row_tokens):
        raise RolloutConversionError("SC token identity differs before slicing")
    if mask.shape != tokens.shape or sampled_logps.shape != tokens.shape:
        raise RolloutConversionError("SC parallel response tracks differ in length")
    selection = slice(start, stop)
    return tuple(np.ascontiguousarray(value[selection]) for value in (
        tokens, mask, sampled_logps, head_ids, head_logps, row_tokens
    ))


def _attach_normalized_sc_head(sample, chunk, turns, transitions, turn_heads, *, k):
    """Mirror actual response construction; matching-projection drops are not response drops."""
    from slime.utils.score_centering import assert_sampled_logps, validate_head
    count = sample.response_length
    ids = np.empty((count, k), dtype="<i4")
    logps = np.empty((count, k), dtype="<f4")
    row_tokens = np.empty(count, dtype="<i4")
    cursor = 0
    for turn_index in range(chunk["start_turn"], chunk["end_turn"] + 1):
        if turn_index > chunk["start_turn"]:
            transition = transitions[turn_index - 1]
            if transition["strategy"] == "split":
                raise RolloutConversionError("SC chunk crosses a split boundary")
            context = transition["context_ids"]
            end = cursor + len(context)
            if list(chunk["response_ids"][cursor:end]) != list(context):
                raise RolloutConversionError("SC context slice differs from the constructed response")
            ids[cursor:end] = np.arange(k, dtype="<i4")
            logps[cursor:end] = -np.log(float(k))
            row_tokens[cursor:end] = context
            if any(chunk["loss_mask"][cursor:end]):
                raise RolloutConversionError("Tool/context rows must be masked")
            cursor = end
        turn = turns[turn_index]
        hi, hp, rt = turn_heads[turn_index]
        values = _slice_sc_tracks(
            turn.output_ids, [1] * len(turn.output_ids), turn.output_log_probs,
            hi, hp, rt, 0, len(turn.output_ids),
        )
        tokens, mask, sampled, hi, hp, rt = values
        end = cursor + len(tokens)
        if end > count or list(chunk["response_ids"][cursor:end]) != tokens.tolist():
            raise RolloutConversionError("SC action token slice differs from the constructed response")
        if list(chunk["loss_mask"][cursor:end]) != mask.tolist():
            raise RolloutConversionError("SC action mask differs from the constructed response")
        if not np.allclose(chunk["rollout_log_probs"][cursor:end], sampled, rtol=0, atol=1e-6):
            raise RolloutConversionError("SC sampled logprobs differ from the constructed response")
        ids[cursor:end], logps[cursor:end], row_tokens[cursor:end] = hi, hp, rt
        cursor = end
    if cursor != count:
        raise RolloutConversionError("SC response was only partially filled")
    expected = np.asarray(sample.tokens[len(sample.tokens)-count:], dtype="<i4")
    if not np.array_equal(expected, row_tokens):
        raise RolloutConversionError("SC final response token identity mismatch")
    validate_head(ids, logps, row_tokens, k=k)
    assert_sampled_logps(ids, logps, row_tokens, sample.rollout_log_probs, sample.loss_mask)
    # Dynamic fields preserve the SC-off dataclass __dict__ and pickle payload.
    sample.rollout_topk_token_ids = ids
    sample.rollout_topk_log_probs = logps
    sample.sc_response_token_ids = row_tokens
```

### slime_plugins/harbor/sample_postprocess.py： cap_total_tokens_allow_response_trim_in_samples 同步真正 response 丢弃

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/sample_postprocess.py","kind":"before","id":"slime_plugins/harbor/sample_postprocess.py#0","sha256":"d0fcd4507670768017aa72c72cd80274bb25fb4ed618822ac98f6fd4e34eba74"} -->
```python
            sample.response_length = trim.response_length
            sample.loss_mask = list(loss_mask[trim.response_drop : trim.response_drop + trim.response_length])
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/sample_postprocess.py","kind":"after","id":"slime_plugins/harbor/sample_postprocess.py#0","sha256":"d0fcd4507670768017aa72c72cd80274bb25fb4ed618822ac98f6fd4e34eba74"} -->
```python
            if getattr(args, "enable_score_centering", False):
                from slime_plugins.harbor.atif_adapter import _slice_sc_tracks
                tracks = _slice_sc_tracks(
                    tokens[len(tokens)-response_length:], loss_mask, logprobs,
                    sample.rollout_topk_token_ids, sample.rollout_topk_log_probs,
                    sample.sc_response_token_ids,
                    trim.response_drop, trim.response_drop + trim.response_length,
                )
                _, _, _, sample.rollout_topk_token_ids, sample.rollout_topk_log_probs, sample.sc_response_token_ids = tracks
                if not (sample.sc_response_token_ids == sample.tokens[-trim.response_length:]).all():
                    raise ValueError("SC head/token identity failed after the hard response cap")
            sample.response_length = trim.response_length
            sample.loss_mask = list(loss_mask[trim.response_drop : trim.response_drop + trim.response_length])
```

### slime_plugins/harbor/sample_postprocess.py： cap_total_tokens_allow_response_trim_in_rollout_data 拒绝晚期破坏性裁切

**Before**

<!-- SC-SPEC {"path":"slime_plugins/harbor/sample_postprocess.py","kind":"before","id":"slime_plugins/harbor/sample_postprocess.py#1","sha256":"d0fcd4507670768017aa72c72cd80274bb25fb4ed618822ac98f6fd4e34eba74"} -->
```python
    """Backstop hard cap for converted train_data dictionaries."""

    cap = _get_cap(args)
```

**After**

<!-- SC-SPEC {"path":"slime_plugins/harbor/sample_postprocess.py","kind":"after","id":"slime_plugins/harbor/sample_postprocess.py#1","sha256":"d0fcd4507670768017aa72c72cd80274bb25fb4ed618822ac98f6fd4e34eba74"} -->
```python
    """Backstop hard cap for converted train_data dictionaries."""

    cap = _get_cap(args)
    if getattr(args, "enable_score_centering", False):
        # This callback runs AFTER old-policy/critic/GAE and CP slicing.
        # A late trim would invalidate logprobs, values, heads and group denominators.
        if cap is not None and any(int(length) > cap for length in rollout_data["total_lengths"]):
            raise ValueError("SC token cap must run on Samples before DP scheduling and old-policy forward")
        return
```

### Sample类型与序列化：无需修改types.py

当前Sample.from_dict在恢复dataclass字段后，会对其他key调用setattr；to_dict复制__dict__。本文只SC-on附加三个动态字段，所以不用增加默认None字段、不改变SC-off Sample序列化schema。Ray直接pickle和Sample.from_dict往返均承载这些字段。禁止把SC ndarray放入metadata，避免日志或JSON_dump将其误展开。

## 4.5 Harbor训练与损失挂载

`slime_plugins/harbor/train_loss.py` 全文件Before/After如下。保留entropy监控、group reducer及KL等现有定制。当前工作区已经有早期apply_score_centering调用，本方案把它置于明确SC gate。pg_objectives的完整实现是共同入口。

**Before（完整现有模块）**

<!-- SC-SPEC {"path":"slime_plugins/harbor/train_loss.py","kind":"before","id":"harbor-loss-full","sha256":"125b855e6e16cd5d514c0cc35ef527e7066a9441d629ba275bcb229f523efa4e"} -->
```python
from __future__ import annotations

import os


def _should_report_entropy(args) -> bool:
    """Whether to compute policy entropy for logging when entropy_coef == 0.

    Default ON: entropy is a cheap (no_grad) but critical health metric. Disable
    by setting ``args.harbor_report_entropy`` or env ``HARBOR_REPORT_ENTROPY`` to
    a falsey value ("0", "false", "no", "off").
    """

    value = getattr(args, "harbor_report_entropy", None)
    if value is None:
        value = os.environ.get("HARBOR_REPORT_ENTROPY", "1")
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def policy_loss_no_entropy_when_disabled(args, batch, logits, sum_of_sample_mean):
    """Policy loss variant that skips entropy graph construction when entropy_coef is zero."""

    import torch

    from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, vanilla_tis_function
    from slime.utils.misc import load_function
    from slime.utils.pg_objectives import apply_score_centering
    from slime.utils.ppo_utils import (
        compute_approx_kl,
        compute_gspo_kl,
        compute_opsm_mask,
        compute_policy_loss,
    )

    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    with_entropy = float(getattr(args, "entropy_coef", 0.0) or 0.0) != 0.0

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=with_entropy,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    if not args.use_rollout_logprobs and not old_log_probs:
        old_log_probs = [log_prob.detach() for log_prob in log_probs]
    train_log_probs_for_tis = batch.get("log_probs")
    if not train_log_probs_for_tis:
        train_log_probs_for_tis = [log_prob.detach() for log_prob in log_probs]

    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"
    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    # PPO + Score Centering (PPO_SC.md). The clipped surrogate above is left
    # untouched; this only adds a detached top-k correction term. No-op unless
    # --enable-score-centering is set, in which case `pg_loss` is returned as
    # the same object. Mounted here as well as in core policy_loss_function
    # because the production recipe routes through this fork, not core.
    pg_loss, sc_metrics = apply_score_centering(args, batch, logits, advantages=advantages, pg_loss=pg_loss)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    if args.get_mismatch_metrics or args.use_tis:
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"
        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": train_log_probs_for_tis,
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # NOTE: positional args must match get_sum_of_sample_mean's signature
        # (total_lengths, response_lengths, loss_masks, sample_denoms,
        #  calculate_per_token_loss, qkv_format, max_seq_lens). The previous call
        # omitted batch["group_mask_sums"] for the 4th (sample_denoms) slot, so
        # calculate_per_token_loss (False=0) landed in sample_denoms -> denom
        # clamped to 1 -> pg_loss / entropy / train_rollout_logprob_abs_diff were
        # left un-normalized (inflated ~mask-token-count, ~1000x) whenever
        # get_mismatch_metrics or use_tis rebuilt the reducer. Pass group_mask_sums
        # to match the core loss.py:1055 call and restore correct per-rollout means.
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            batch["group_mask_sums"],
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    if with_entropy:
        entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        # entropy_coef == 0: entropy is NOT part of the optimization objective, but
        # we still want it as a monitoring signal (entropy collapse is the canonical
        # RL-degeneration tell). Compute it under torch.no_grad() so NO backward
        # graph / activations are retained -- this is what makes it memory-safe on
        # the 27B (the whole reason this custom loss skips the grad-enabled entropy).
        # Gated by HARBOR_REPORT_ENTROPY (default on); set to 0 to fully skip.
        entropy_loss = logits.new_zeros(())
        if _should_report_entropy(args):
            with torch.no_grad():
                _, entropy_only = get_log_probs_and_entropy(
                    logits,
                    args=args,
                    unconcat_tokens=batch["unconcat_tokens"],
                    total_lengths=total_lengths,
                    response_lengths=response_lengths,
                    with_entropy=True,
                    max_seq_lens=max_seq_lens,
                )
                entropy_report = torch.cat(entropy_only["entropy"], dim=0)
                # Reuse the same per-sample-mean reducer so the reported number is
                # directly comparable to the entropy_coef != 0 path.
                entropy_loss = sum_of_sample_mean(entropy_report).detach()

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        for metric_key, metric_value in tis_metrics.items():
            reported_loss[f"{metric_key}"] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Empty unless score centering is on, so this loop is a no-op on the baseline.
    for metric_key, metric_value in sc_metrics.items():
        reported_loss[metric_key] = sum_of_sample_mean(metric_value.detach()).clone().detach()

    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss
```

**After（完整替换模块）**

<!-- SC-SPEC {"path":"slime_plugins/harbor/train_loss.py","kind":"after","id":"harbor-loss-full","sha256":"125b855e6e16cd5d514c0cc35ef527e7066a9441d629ba275bcb229f523efa4e"} -->
```python
from __future__ import annotations

import os


def _should_report_entropy(args) -> bool:
    """Whether to compute policy entropy for logging when entropy_coef == 0.

    Default ON: entropy is a cheap (no_grad) but critical health metric. Disable
    by setting ``args.harbor_report_entropy`` or env ``HARBOR_REPORT_ENTROPY`` to
    a falsey value ("0", "false", "no", "off").
    """

    value = getattr(args, "harbor_report_entropy", None)
    if value is None:
        value = os.environ.get("HARBOR_REPORT_ENTROPY", "1")
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def policy_loss_no_entropy_when_disabled(args, batch, logits, sum_of_sample_mean):
    """Policy loss variant that skips entropy graph construction when entropy_coef is zero."""

    import torch

    from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, vanilla_tis_function
    from slime.utils.misc import load_function
    from slime.utils.pg_objectives import apply_score_centering
    from slime.utils.ppo_utils import (
        compute_approx_kl,
        compute_gspo_kl,
        compute_opsm_mask,
        compute_policy_loss,
    )

    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    with_entropy = float(getattr(args, "entropy_coef", 0.0) or 0.0) != 0.0

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=with_entropy,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    if not args.use_rollout_logprobs and not old_log_probs:
        old_log_probs = [log_prob.detach() for log_prob in log_probs]
    train_log_probs_for_tis = batch.get("log_probs")
    if not train_log_probs_for_tis:
        train_log_probs_for_tis = [log_prob.detach() for log_prob in log_probs]

    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"
    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    # PPO + Score Centering (PPO_SC.md). The clipped surrogate above is left
    # untouched; this only adds a detached top-k correction term. No-op unless
    # --enable-score-centering is set, in which case `pg_loss` is returned as
    # the same object. Mounted here as well as in core policy_loss_function
    # because the production recipe routes through this fork, not core.
    sc_metrics = {}
    if getattr(args, "enable_score_centering", False):
        pg_loss, sc_metrics = apply_score_centering(args, batch, logits, advantages=advantages, pg_loss=pg_loss)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    if args.get_mismatch_metrics or args.use_tis:
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"
        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": train_log_probs_for_tis,
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # NOTE: positional args must match get_sum_of_sample_mean's signature
        # (total_lengths, response_lengths, loss_masks, sample_denoms,
        #  calculate_per_token_loss, qkv_format, max_seq_lens). The previous call
        # omitted batch["group_mask_sums"] for the 4th (sample_denoms) slot, so
        # calculate_per_token_loss (False=0) landed in sample_denoms -> denom
        # clamped to 1 -> pg_loss / entropy / train_rollout_logprob_abs_diff were
        # left un-normalized (inflated ~mask-token-count, ~1000x) whenever
        # get_mismatch_metrics or use_tis rebuilt the reducer. Pass group_mask_sums
        # to match the core loss.py:1055 call and restore correct per-rollout means.
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            batch["group_mask_sums"],
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    if with_entropy:
        entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        # entropy_coef == 0: entropy is NOT part of the optimization objective, but
        # we still want it as a monitoring signal (entropy collapse is the canonical
        # RL-degeneration tell). Compute it under torch.no_grad() so NO backward
        # graph / activations are retained -- this is what makes it memory-safe on
        # the 27B (the whole reason this custom loss skips the grad-enabled entropy).
        # Gated by HARBOR_REPORT_ENTROPY (default on); set to 0 to fully skip.
        entropy_loss = logits.new_zeros(())
        if _should_report_entropy(args):
            with torch.no_grad():
                _, entropy_only = get_log_probs_and_entropy(
                    logits,
                    args=args,
                    unconcat_tokens=batch["unconcat_tokens"],
                    total_lengths=total_lengths,
                    response_lengths=response_lengths,
                    with_entropy=True,
                    max_seq_lens=max_seq_lens,
                )
                entropy_report = torch.cat(entropy_only["entropy"], dim=0)
                # Reuse the same per-sample-mean reducer so the reported number is
                # directly comparable to the entropy_coef != 0 path.
                entropy_loss = sum_of_sample_mean(entropy_report).detach()

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        for metric_key, metric_value in tis_metrics.items():
            reported_loss[f"{metric_key}"] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Empty unless score centering is on, so this loop is a no-op on the baseline.
    for metric_key, metric_value in sc_metrics.items():
        reported_loss[metric_key] = sum_of_sample_mean(metric_value.detach()).clone().detach()

    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss
```

文件：`slime/utils/pg_objectives.py`。以下完整替换模块（已有早期版本，不是新增空文件）；不保留旧函数的并行实现。

<!-- SC-SPEC {"path":"slime/utils/pg_objectives.py","kind":"full","id":"slime/utils/pg_objectives.py#full","sha256":"4e81708cefc718cea06dc77c5158a012a23be193a03246895d48681354e08520"} -->
```python
"""Shared optional SC mount; preserve the original PPO tensor when disabled."""
import torch


def apply_score_centering(args, batch, logits, *, advantages, pg_loss):
    if not getattr(args, "enable_score_centering", False):
        return pg_loss, {}
    from slime.backends.megatron_utils.sc_terms import get_score_centering_terms
    if args.sc_correction_scale == 0:
        with torch.no_grad():
            addend, metrics = get_score_centering_terms(args, batch, logits, advantages=advantages)
    else:
        addend, metrics = get_score_centering_terms(args, batch, logits, advantages=advantages)
        pg_loss = pg_loss + args.sc_correction_scale * addend
    metrics["sc_applied"] = torch.full_like(addend.detach(), float(args.sc_correction_scale != 0))
    return pg_loss, metrics
```

### core对称挂载

### slime/backends/megatron_utils/loss.py： PPO 原算子后唯一 SC gate

**Before**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/loss.py","kind":"before","id":"slime/backends/megatron_utils/loss.py#0","sha256":"86077a2a39a0a8ce2e7b9ef8b87767fd98278f6db4abb86ce1b0d8e4118a9f8e"} -->
```python
    pg_loss, sc_metrics = apply_score_centering(args, batch, logits, advantages=advantages, pg_loss=pg_loss)
```

**After**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/loss.py","kind":"after","id":"slime/backends/megatron_utils/loss.py#0","sha256":"86077a2a39a0a8ce2e7b9ef8b87767fd98278f6db4abb86ce1b0d8e4118a9f8e"} -->
```python
    sc_metrics = {}
    if getattr(args, "enable_score_centering", False):
        pg_loss, sc_metrics = apply_score_centering(args, batch, logits, advantages=advantages, pg_loss=pg_loss)
```

## 4.6 必需数据通道、old-policy前向、CP/TP与日志

这些不是可选附录；缺少任一生产者/消费者改造都不能称为全链路合入。old-policy callback遵守本地forward_only返回 `(empty_tensor, dict[list[Tensor]])` 的真实约定。

文件：`slime/backends/megatron_utils/sc_terms.py`。以下完整替换模块（已有早期版本，不是新增空文件）；不保留旧函数的并行实现。

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/sc_terms.py","kind":"full","id":"slime/backends/megatron_utils/sc_terms.py#full","sha256":"d6b70e3a9a39e2244bce2d4e0b71a7497aa08c72c8b02272a080af825e37ff68"} -->
```python
"""SC batching with one gather over packed logits, full CPU heads and explicit CP maps."""
from __future__ import annotations

import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.utils.score_centering import calculate_head_log_probs, score_centering_correction

_APPLIED = False


def score_centering_was_applied():
    return _APPLIED


def reset_score_centering_applied():
    global _APPLIED
    _APPLIED = False


def assert_score_centering_was_applied(args):
    if getattr(args, "enable_score_centering", False) and not _APPLIED:
        raise RuntimeError("SC enabled but the selected policy loss never computed its correction")


def response_row_layout(args, batch, logits):
    from .cp_utils import get_logits_and_tokens_offset_with_cp
    flat = logits.reshape(-1, logits.size(-1))
    cp = mpu.get_context_parallel_world_size()
    rank = mpu.get_context_parallel_rank()
    if logits.dtype != torch.float32 or logits.ndim != 3:
        raise ValueError("SC expects float32 [B,T,V/TP] policy logits")
    layouts = []
    packed_offset = 0
    global_offset = 0
    for i, (total, response) in enumerate(zip(batch["total_lengths"], batch["response_lengths"], strict=True)):
        if not 0 < response < total:
            raise ValueError("SC requires nonempty prompt and response")
        prompt = total - response
        maximum = batch.get("max_seq_lens")
        maximum = maximum[i] if maximum is not None else None
        if cp == 1:
            base = i * maximum if args.qkv_format == "bshd" else global_offset
            rows = torch.arange(base + prompt - 1, base + total - 1, device=flat.device)
            response_rows = torch.arange(response, device=flat.device)
        elif args.allgather_cp:
            lo = max(global_offset + prompt - 1, rank * flat.size(0))
            hi = min(global_offset + total - 1, (rank + 1) * flat.size(0))
            hi = max(lo, hi)
            rows = torch.arange(lo - rank * flat.size(0), hi - rank * flat.size(0), device=flat.device)
            response_rows = torch.arange(lo - (global_offset + prompt - 1), hi - (global_offset + prompt - 1), device=flat.device)
        else:
            chunk, chunks, offsets, _ = get_logits_and_tokens_offset_with_cp(
                total, response, args.qkv_format, maximum
            )
            row_parts, response_parts = [], []
            for half in (0, 1):
                lo, hi = offsets[half]
                if hi > lo:
                    row_parts.append(torch.arange(
                        packed_offset + half * chunk + lo - chunks[half][0],
                        packed_offset + half * chunk + hi - chunks[half][0], device=flat.device
                    ))
                    response_parts.append(torch.arange(lo - prompt + 1, hi - prompt + 1, device=flat.device))
            rows = torch.cat(row_parts) if row_parts else torch.empty(0, device=flat.device, dtype=torch.long)
            response_rows = torch.cat(response_parts) if response_parts else torch.empty(0, device=flat.device, dtype=torch.long)
            packed_offset += 2 * chunk
        layouts.append((rows.long(), response_rows.long()))
        global_offset += total
    return flat, layouts


def _gather_current_heads(args, batch, logits):
    flat, layouts = response_row_layout(args, batch, logits)
    candidates, packed_rows, masks, response_ids = [], [], [], []
    for i, ((rows, response_rows), total, response) in enumerate(zip(
        layouts, batch["total_lengths"], batch["response_lengths"], strict=True
    )):
        cpu_index = response_rows.cpu()
        full_ids = batch["rollout_topk_token_ids"][i]
        if tuple(full_ids.shape) != (response, args.sc_top_k):
            raise ValueError("SC expects full response heads on CPU; no actor-side CP slicing")
        ids = full_ids.index_select(0, cpu_index).to(device=flat.device, dtype=torch.long)
        token_ids = batch["unconcat_tokens"][i][total-response:total].index_select(0, response_rows)
        audit_ids = batch["sc_response_token_ids"][i].index_select(0, cpu_index).to(flat.device)
        if not torch.equal(token_ids, audit_ids.long()):
            raise ValueError("SC token identity mismatch after scheduling or CP mapping")
        masks.append(batch["loss_masks"][i].index_select(0, response_rows).bool())
        response_ids.append(token_ids)
        candidates.append(ids)
        packed_rows.append(rows)
    local_mask = torch.cat(masks)
    ids = torch.cat(candidates)
    packed = torch.cat(packed_rows)
    # Masked tool rows never enter softmax/gather and cannot cause 0 * NaN.
    valid_rows = torch.nonzero(local_mask, as_tuple=False).flatten()
    selected = calculate_head_log_probs(
        flat, packed.index_select(0, valid_rows), ids.index_select(0, valid_rows),
        mpu.get_tensor_model_parallel_group(), args.sc_chunk_size
    )
    head = selected.new_zeros((len(packed), args.sc_top_k)).index_copy(0, valid_rows, selected)
    return head, layouts, local_mask, ids, torch.cat(response_ids)


def old_policy_head_callback(callback, args, batch):
    def collect(logits):
        result, outputs = callback(logits)
        with torch.no_grad():
            head, layouts, mask, _, _ = _gather_current_heads(args, batch, logits)
            lengths = [len(rows) for rows, _ in layouts]
            values = []
            for part, (_, response_rows), response in zip(
                head.split(lengths), layouts, batch["response_lengths"], strict=True
            ):
                full = part.new_zeros((response, args.sc_top_k))
                full.index_copy_(0, response_rows, part)
                if mpu.get_context_parallel_world_size() > 1:
                    dist.all_reduce(full, group=mpu.get_context_parallel_group())
                values.append(full.cpu())
            outputs["sc_old_head_log_probs"] = values
        return result, outputs
    return collect


def _full_advantages(args, batch):
    from .cp_utils import get_logits_and_tokens_offset_with_cp
    values = []
    for i, (adv, total, response) in enumerate(zip(
        batch["advantages"], batch["total_lengths"], batch["response_lengths"], strict=True
    )):
        if mpu.get_context_parallel_world_size() == 1:
            values.append(adv.detach())
            continue
        maximum = batch.get("max_seq_lens")
        maximum = maximum[i] if maximum is not None else None
        _, _, offsets, _ = get_logits_and_tokens_offset_with_cp(total, response, args.qkv_format, maximum)
        prompt = total - response
        pieces = [
            torch.arange(lo-prompt+1, hi-prompt+1, device=adv.device)
            for lo, hi in offsets if hi > lo
        ]
        indices = torch.cat(pieces) if pieces else torch.empty(0, device=adv.device, dtype=torch.long)
        full = adv.new_zeros(response)
        full.index_copy_(0, indices.long(), adv.detach())
        dist.all_reduce(full, group=mpu.get_context_parallel_group())
        values.append(full)
    return values


def get_score_centering_terms(args, batch, logits, *, advantages):
    global _APPLIED
    for key in ("rollout_topk_token_ids", "rollout_topk_log_probs", "sc_response_token_ids", "rollout_log_probs"):
        if batch.get(key) is None:
            raise ValueError(f"SC missing training batch field: {key}")
        if len(batch[key]) != len(batch["response_lengths"]):
            raise ValueError(f"SC sample count mismatch: {key}")
    head, layouts, mask, ids, sampled = _gather_current_heads(args, batch, logits)
    local_q, local_old, local_adv, local_sampled_lp = [], [], [], []
    weighted = args.sc_formula == "ppo_weighted"
    need_old = weighted and args.sc_weight_denominator == "old_actor"
    if need_old and batch.get("sc_old_head_log_probs") is None:
        raise ValueError("Old-policy SC head producer did not run")
    full_adv = _full_advantages(args, batch) if args.allgather_cp else None
    for i, (_, response_rows) in enumerate(layouts):
        cpu_index = response_rows.cpu()
        local_q.append(batch["rollout_topk_log_probs"][i].index_select(0, cpu_index).to(logits.device))
        if need_old:
            local_old.append(batch["sc_old_head_log_probs"][i].index_select(0, cpu_index).to(logits.device))
        if full_adv is not None:
            local_adv.append(full_adv[i].index_select(0, response_rows))
        else:
            local_adv.append(batch["advantages"][i])
        # rollout_log_probs is zigzag on the actor; reconstruct only for allgather-CP.
        if args.allgather_cp:
            temporary = dict(batch)
            temporary["advantages"] = batch["rollout_log_probs"]
            # Reconstruct once below, outside this per-sample loop.
        else:
            local_sampled_lp.append(batch["rollout_log_probs"][i])
    if args.allgather_cp:
        all_lp = _full_advantages(args, temporary)
        local_sampled_lp = [
            value.index_select(0, index) for value, (_, index) in zip(all_lp, layouts, strict=True)
        ]
    q = torch.cat(local_q)
    adv = torch.cat(local_adv).detach()
    adv = torch.where(mask, adv, torch.zeros_like(adv))
    if not torch.isfinite(adv).all():
        raise ValueError("SC requires finite advantages on unmasked tokens")
    logged = torch.cat(local_sampled_lp)
    if len(adv) != len(head) or len(logged) != len(head):
        raise ValueError("SC advantages/sampled-logprobs do not match local response rows")
    hit_matrix = ids == sampled[:, None]
    hit = hit_matrix.any(-1) & mask
    first = hit_matrix.long().argmax(-1, keepdim=True)
    deviation = (q.gather(-1, first).squeeze(-1) - logged).abs()
    if hit.any() and (deviation[hit] > 1e-4).any():
        raise ValueError("SC sampled-token logprob differs from transported head")
    correction, metrics = score_centering_correction(
        train_head_logp=head, sampler_head_logp=q, advantages=adv,
        formula=args.sc_formula, old_head_logp=torch.cat(local_old) if need_old else None,
        eps_clip_low=args.eps_clip, eps_clip_high=args.eps_clip_high,
        tail_eps=args.sc_tail_eps, loss_mask=mask
    )
    metrics["sc_head_hit"] = hit.float().detach()
    metrics["sc_head_logp_deviation"] = torch.where(hit, deviation, torch.zeros_like(deviation)).detach()
    metrics["sc_correction"] = correction.detach()
    # Multiply advantages before redistribution because weighted SC uses their signs.
    scaled = adv * correction
    lengths = [len(rows) for rows, _ in layouts]
    values = {"sc_addend": list(scaled.split(lengths))}
    values.update({key: list(value.split(lengths)) for key, value in metrics.items()})
    if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
        from .loss import _allgather_cp_redistribute
        _allgather_cp_redistribute(
            values, logits_local_len=logits.size(1), args=args,
            total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens"),
        )
    addend = torch.cat(values.pop("sc_addend"))
    if addend.shape != advantages.shape:
        raise ValueError("SC final correction is not in the PPO advantage layout")
    _APPLIED = True
    return addend, {key: torch.cat(value).detach() for key, value in values.items()}
```

### slime/ray/rollout.py： _convert_samples_to_train_data 完整校验后添加 ndarray

**Before**

<!-- SC-SPEC {"path":"slime/ray/rollout.py","kind":"before","id":"slime/ray/rollout.py#0","sha256":"e940fdc5177269e204a955a44df0c92439e4d84e5f6b2152288d4856224a0e7b"} -->
```python
        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]
```

**After**

<!-- SC-SPEC {"path":"slime/ray/rollout.py","kind":"after","id":"slime/ray/rollout.py#0","sha256":"e940fdc5177269e204a955a44df0c92439e4d84e5f6b2152288d4856224a0e7b"} -->
```python
        if getattr(self.args, "enable_score_centering", False):
            import numpy as np
            from slime.utils.score_centering import HEAD_KEYS, assert_sampled_logps, validate_head
            total_bytes = 0
            for sample in samples:
                response = sample.response_length
                if not 0 < response < len(sample.tokens):
                    raise ValueError("SC requires nonempty prompt and response")
                fields = [getattr(sample, key, None) for key in HEAD_KEYS]
                if any(value is None for value in fields):
                    raise ValueError("Harbor did not attach all SC response fields")
                ids, logps, row_tokens = validate_head(*fields, k=self.args.sc_top_k)
                if not np.array_equal(row_tokens, np.asarray(sample.tokens[-response:], dtype="<i4")):
                    raise ValueError("SC sample token IDs changed after Harbor conversion")
                assert_sampled_logps(ids, logps, row_tokens, sample.rollout_log_probs, sample.loss_mask)
                total_bytes += sum(value.nbytes for value in fields)
            if total_bytes > self.args.sc_max_batch_bytes:
                raise ValueError(f"SC batch heads require {total_bytes} bytes; increase budget only after memory review")
            for key in HEAD_KEYS:
                train_data[key] = [getattr(sample, key) for sample in samples]
        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]
```

### slime/ray/rollout.py： _split_train_data_by_dp 使用既有 partition

**Before**

<!-- SC-SPEC {"path":"slime/ray/rollout.py","kind":"before","id":"slime/ray/rollout.py#1","sha256":"e940fdc5177269e204a955a44df0c92439e4d84e5f6b2152288d4856224a0e7b"} -->
```python
            # keys that need to be splited at train side
```

**After**

<!-- SC-SPEC {"path":"slime/ray/rollout.py","kind":"after","id":"slime/ray/rollout.py#1","sha256":"e940fdc5177269e204a955a44df0c92439e4d84e5f6b2152288d4856224a0e7b"} -->
```python
            if getattr(self.args, "enable_score_centering", False):
                from slime.utils.score_centering import HEAD_KEYS
                for key in HEAD_KEYS:
                    if key not in data:
                        raise ValueError(f"SC DP partition is missing {key}")
                    rollout_data[key] = [data[key][j] for j in partition]
            # keys that need to be splited at train side
```

### slime/backends/megatron_utils/actor.py： _get_rollout_data CPU heads 保留完整 response，不做 CP 切片

**Before**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/actor.py","kind":"before","id":"slime/backends/megatron_utils/actor.py#0","sha256":"bde660986aaa5d483be48d91410edee584bd20231a23ed9b17fa50bd2dc283b6"} -->
```python
        if "rollout_routed_experts" in rollout_data:
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
            ]
        return rollout_data
```

**After**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/actor.py","kind":"after","id":"slime/backends/megatron_utils/actor.py#0","sha256":"bde660986aaa5d483be48d91410edee584bd20231a23ed9b17fa50bd2dc283b6"} -->
```python
        if "rollout_routed_experts" in rollout_data:
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
            ]
        if getattr(self.args, "enable_score_centering", False) and self.role == "actor":
            from slime.utils.score_centering import HEAD_KEYS
            keep = mpu.is_pipeline_last_stage(ignore_virtual=True)
            for key in HEAD_KEYS:
                if key not in rollout_data:
                    raise ValueError(f"SC actor input missing {key}")
                if keep:
                    # Private writable CPU buffers; Ray-backed ndarrays can be read-only.
                    rollout_data[key] = [torch.from_numpy(value.copy()) for value in rollout_data[key]]
                else:
                    del rollout_data[key]
        elif self.role == "critic":
            # Keys are absent in baseline jobs; do not import or allocate SC data.
            for key in ("rollout_topk_token_ids", "rollout_topk_log_probs", "sc_response_token_ids"):
                if key in rollout_data:
                    del rollout_data[key]
        return rollout_data
```

### slime/backends/megatron_utils/actor.py： train_actor 每 rollout 重置挂载检测

**Before**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/actor.py","kind":"before","id":"slime/backends/megatron_utils/actor.py#1","sha256":"bde660986aaa5d483be48d91410edee584bd20231a23ed9b17fa50bd2dc283b6"} -->
```python
            # Train
            if self.args.use_routing_replay:
```

**After**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/actor.py","kind":"after","id":"slime/backends/megatron_utils/actor.py#1","sha256":"bde660986aaa5d483be48d91410edee584bd20231a23ed9b17fa50bd2dc283b6"} -->
```python
            if getattr(self.args, "enable_score_centering", False) and mpu.is_pipeline_last_stage(ignore_virtual=True):
                from slime.backends.megatron_utils.sc_terms import reset_score_centering_applied
                reset_score_centering_applied()
            # Train
            if self.args.use_routing_replay:
```

### slime/backends/megatron_utils/actor.py： train_actor 挂载检测只在物理 PP last stage

**Before**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/actor.py","kind":"before","id":"slime/backends/megatron_utils/actor.py#2","sha256":"bde660986aaa5d483be48d91410edee584bd20231a23ed9b17fa50bd2dc283b6"} -->
```python
            if getattr(self.args, "enable_score_centering", False):
                from slime.backends.megatron_utils.sc_terms import assert_score_centering_was_applied

                assert_score_centering_was_applied(self.args)
```

**After**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/actor.py","kind":"after","id":"slime/backends/megatron_utils/actor.py#2","sha256":"bde660986aaa5d483be48d91410edee584bd20231a23ed9b17fa50bd2dc283b6"} -->
```python
            if getattr(self.args, "enable_score_centering", False) and mpu.is_pipeline_last_stage(ignore_virtual=True):
                from slime.backends.megatron_utils.sc_terms import assert_score_centering_was_applied
                assert_score_centering_was_applied(self.args)
```

### slime/backends/megatron_utils/actor.py： MegatronTrainRayActor.init critic 防继承 SC 的最终守卫

**Before**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/actor.py","kind":"before","id":"slime/backends/megatron_utils/actor.py#3","sha256":"bde660986aaa5d483be48d91410edee584bd20231a23ed9b17fa50bd2dc283b6"} -->
```python
        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return start_rollout_id
```

**After**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/actor.py","kind":"after","id":"slime/backends/megatron_utils/actor.py#3","sha256":"bde660986aaa5d483be48d91410edee584bd20231a23ed9b17fa50bd2dc283b6"} -->
```python
        if role == "critic":
            if getattr(self.args, "enable_score_centering", False):
                self.args.enable_score_centering = False
                if hasattr(self.args, "use_score_centering"):
                    self.args.use_score_centering = False
            if self.args.offload_train:
                self.sleep()
            return start_rollout_id
```

### slime/backends/megatron_utils/model.py： forward_only.forward_step old-policy 取 head keys

**Before**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/model.py","kind":"before","id":"slime/backends/megatron_utils/model.py#0","sha256":"3b907f32d727e2068d684772c1ca30d6087c9d455dda9eda6ff1bd780288ea98"} -->
```python
                "max_seq_lens",
            ],
            args.data_pad_size_multiplier,
```

**After**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/model.py","kind":"after","id":"slime/backends/megatron_utils/model.py#0","sha256":"3b907f32d727e2068d684772c1ca30d6087c9d455dda9eda6ff1bd780288ea98"} -->
```python
                "max_seq_lens",
            ] + (
                ["rollout_topk_token_ids", "sc_response_token_ids"]
                if getattr(args, "enable_score_centering", False)
                and args.sc_formula == "ppo_weighted"
                and args.sc_weight_denominator == "old_actor"
                and store_prefix == ""
                and mpu.is_pipeline_last_stage(ignore_virtual=True)
                else []
            ),
            args.data_pad_size_multiplier,
```

### slime/backends/megatron_utils/model.py： forward_only.forward_step 复用实际 old-actor 前向回调

**Before**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/model.py","kind":"before","id":"slime/backends/megatron_utils/model.py#1","sha256":"3b907f32d727e2068d684772c1ca30d6087c9d455dda9eda6ff1bd780288ea98"} -->
```python
        return output_tensor, partial(
            f,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
        )
```

**After**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/model.py","kind":"after","id":"slime/backends/megatron_utils/model.py#1","sha256":"3b907f32d727e2068d684772c1ca30d6087c9d455dda9eda6ff1bd780288ea98"} -->
```python
        callback = partial(
            f,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
        )
        if (
            getattr(args, "enable_score_centering", False)
            and args.sc_formula == "ppo_weighted"
            and args.sc_weight_denominator == "old_actor"
            and store_prefix == ""
            and mpu.is_pipeline_last_stage(ignore_virtual=True)
        ):
            from slime.backends.megatron_utils.sc_terms import old_policy_head_callback
            callback = old_policy_head_callback(callback, args, batch)
        return output_tensor, callback
```

### slime/backends/megatron_utils/model.py： train_one_step.forward_step SC-only batch keys

**Before**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/model.py","kind":"before","id":"slime/backends/megatron_utils/model.py#2","sha256":"3b907f32d727e2068d684772c1ca30d6087c9d455dda9eda6ff1bd780288ea98"} -->
```python
                "group_mask_sums",
            ],
            args.data_pad_size_multiplier,
```

**After**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/model.py","kind":"after","id":"slime/backends/megatron_utils/model.py#2","sha256":"3b907f32d727e2068d684772c1ca30d6087c9d455dda9eda6ff1bd780288ea98"} -->
```python
                "group_mask_sums",
            ] + (
                ["rollout_topk_token_ids", "rollout_topk_log_probs", "sc_response_token_ids",
                 "sc_old_head_log_probs"]
                if getattr(args, "enable_score_centering", False)
                and args.loss_type in {"policy_loss", "custom_loss"}
                and mpu.is_pipeline_last_stage(ignore_virtual=True)
                else []
            ),
            args.data_pad_size_multiplier,
```

### slime/backends/megatron_utils/data.py： log_rollout_data 大矩阵不得进入标量指标归约

**Before**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/data.py","kind":"before","id":"slime/backends/megatron_utils/data.py#0","sha256":"8ef582c708fe28aae972e8465a7c4e717b67337a02341e4b53d651fa7702ed35"} -->
```python
        for key, val in rollout_data.items():
            if key in [
```

**After**

<!-- SC-SPEC {"path":"slime/backends/megatron_utils/data.py","kind":"after","id":"slime/backends/megatron_utils/data.py#0","sha256":"8ef582c708fe28aae972e8465a7c4e717b67337a02341e4b53d651fa7702ed35"} -->
```python
        for key, val in rollout_data.items():
            if getattr(args, "enable_score_centering", False) and key in {
                "rollout_topk_token_ids", "rollout_topk_log_probs",
                "sc_response_token_ids", "sc_old_head_log_probs",
            }:
                continue
            if key in [
```


# 5. 方案自检与完备性审计（Self-Verification Audit）

## 5.1 全方案完备性与正确性自检报告

审计层级：**代码级推导/内存重建 Pass 不等于线上验收 Pass**。本轮测试执行的是本文拟议代码字符串，使用 compile/exec 装载到独立进程，没有创建或修改业务 .py/.sh。现有 test_harbor_atif_adapter.py 通过内存替换模块执行；pytest 关闭 cacheprovider 且 PYTHONDONTWRITEBYTECODE=1。没有运行训练、启动/停止 Ray、修改 SGLang 或应用补丁。

| 审核项 | 结论 | 证据/限制 |
|---|---|---|
| score 的反向等于 A(q−αp)∇logp | **Pass** | CPU FP64 解析系数比对；q/adv 梯度为 None；系数显式 detach |
| weighted 与 PPO clip 梯度权重对应 | **Pass（所支持算子）** | 原 compute_policy_loss 未改；r 在平坦区变0；不支持 dual-clip；与score非等价已明确 |
| 所有模块可直接重建 | **Pass（静态）** | 第6章提取器按 Before/After 顺序重建；hash/唯一锚点/AST/compile 检查，结果见6.5 |
| FP64 无降精度 | **Pass** | 算法与 head gather 保留 float64；生产 float32 |
| Tail epsilon=1e-8 | **Pass** | 对差值 clamp_min，而非对质量 clamp(1−eps)；q零尾部保持0 |
| masked 工具 token / padding | **Pass（合成）** | gather 前 mask 选行；标量层 where；masked NaN/Inf 测试梯度有限且为0 |
| Token ID 严格对齐 | **Pass（合成）** | prompt/token/head digest、完整 response row IDs、slice前后断言、采样logp cross-check |
| prefix/suffix投影处理 | **Pass** | normalized builder 保留真实输出；5 fixture覆盖prompt suffix、response prefix/suffix投影、split、retokenized |
| 真正response hard trim | **Pass** | 新发现sample_postprocess六轨统一切片；晚期rollout_data裁切报错测试 |
| Sample字段与往返 | **Pass（CPU）** | 动态字段经原to_dict/from_dict恢复；off/on去掉三项后原数据相等 |
| 既有Harbor adapter测试 | **Pass** | **18 passed**；真实torch、独立进程；未修改原测试文件 |
| CP映射镜像当前get_responses | **Pass** | **21组** THD/BSHD、CP1/2/4、zigzag/allgather逐元素对照 |
| CP2两种公式、old head、反向 | **Pass（gloo）** | 2进程，zigzag/allgather×score/weighted，old head匹配单机参考；反向有限 |
| TP2 + chunk | **Pass（gloo/FP64）** | 2进程，chunk=1/3/256；forward和局部梯度匹配全词表，无TP倍增 |
| PPO原生损失代码保留 | **Pass（源码级）** | 不修改compute_policy_loss、奖励/GAE、原loss归约、entropy/KL、R3算子 |
| SC-off不读/分配Top-k | **Pass（分支推导）** | 大张量读取、HTTP fields、Ray keys、取数与collective均由enable守卫 |
| SC-off真实训练逐bit | **未执行 / 发布门禁未通过** | 未占用生产GPU跑step，必须Stage3比对全部梯度与参数更新 |
| Ray多节点内存与网络 | **未执行 / 发布门禁未通过** | 预算不是实测峰值；无业务disk-spill须按最大并发测量 |
| SGLang binary canary | **未执行 / 发布门禁未通过** | live未改；最小patch只写在本文 |
| “所有Python执行指令逐字节相同” | **Fail：命题不可能成立** | gate本身新增指令；本文承诺受控数值/请求/有效payload不变 |
| “任意PPO-SC无条件全词表零漂移” | **Fail：前提不成立** | 公式差异、tail近似、action-dependent A/权重及重分词条件失配有限制 |
| 直接全量启用27B/122B | **Fail：不可作为已验收上线批准** | 须完成Stage1–4；本文是完整实施规格，不是已验证的生产部署 |

## 5.2 逐模块逐路径闭环审阅

**配置。** 每个新参数在add_algo_arguments定义；canonical/legacy CLI共用destination，canonical YAML在合并后映射。不增加pg-loss-type，不切REINFORCE。SC-only validation拒绝NaN、非法尺寸、未支持TIS/OPSM/constraint/dual-clip和自定义数据回调。保留基线cap hook白名单，不能笼统禁止所有hook导致两脚本无法启用。角色SC参数不得独立覆盖；critic init是第二道隔离。

**算法与autograd。** _fp保留double；masked行在exp前归零；coefficient显式detach；adv组合前detach且masked归零。normalizer为完整词表TP MAX/SUM，候选采用global ID映射；index_select形成独立workspace，in-place不触碰原logits。backward一次分配完整梯度，index_copy写唯一row，无逐sample大slice-backward累积。没有backward all-reduce；empty本地行保持可微零路径。once_differentiable明确仅支持一阶梯度。

**通信与codec。** 请求只传SGLang既有top_logprobs_num和return_text_in_logprobs；score_centering_top_k从args透传到Harbor构造器再映射，不向SGLang塞未知schema key。k、行数、base64字节长度、dtype、唯一ID、概率质量、vocab bounds均检查。JSON/binary显式选择，不静默fallback。extra必须JSON-safe：Harbor Trial会model_dump_json，不能直接塞np.ndarray；进入Sample前才解码紧凑数组。

**Token流水。** 每轮envelope绑定prompt/generated IDs；生产chunk逐段校验context/action。投影drop不改真实response；实际hard cap使用_slice_sc_tracks。未来任何token mutation必须同步所有轨道，否则pack/actor身份断言失败。Top-k不命中采样token是合法事件，不无条件要求hit rate>0.95；命中时核对logp。digest防错配但不证明SGLang内部概率计算正确。

**Ray/DP/actor。** SC才挂动态字段；打包逐样本检查，不能只检查samples[0]。同一partition切分三个list，不新增remote参数。actor复制writable CPU缓冲，避免写Ray只读数组。非PP-last删除head；critic删除head并关闭SC。actor不预切zigzag，allgather可读取全CPU行；old_head仅训练侧使用。

**Old-actor生产者。** 现有old_actor切换后的store_prefix="" forward_only回调复用同一次前向；ref_/teacher_不触发。有效mask行一次packed gather，scatter回完整response，CP SUM后CPU保存；已有动态microbatch逆排序适用于新key。分母必须与use_rollout_logprobs吻合。最终输出回调由Megatron在产生logits的stage执行。

**CP。** response_row_layout同时产生packed logits行与full response行，prediction offset=-1。allgather中adv先从zigzag拼full再取continuous行；weighted用该A符号；先乘A再调用本地redistribute，显式args/max_seq_lens；最终addend shape等于PPO advantages。BSHD+allgather被拒绝，因为当前get_responses不支持其padding语义。

**损失与指标。** Harbor及core保留实际PPO调用；gate在原算子后、原reducer前。scale0使用no_grad且不执行pg_loss+0×correction。group_mask_sums、per-token/per-group、entropy与KL不变；矩阵不进入标量日志。_APPLIED逐rollout重置、仅物理PP-last检查。SC训练下trial异常上抛，避免坏head被当作可丢弃数据继续训练；因此SC-on容错行为更严格，SC-off原样。

## 5.3 主机内存、Ray与GPU峰值

raw resident heads成本 B=ΣR_i×(8k+4)，包括两个矩阵与独立token IDs。k32为260B/token，k128为1028B/token；718×40k条token在k32约6.95GiB，在k128约27.5GiB。718仅说明规模，不是当前动态partial rollout实测样本数；真实预算必须遍历全部response_length。

DP partition互不相交，一轮ray.put总raw head是B，不是DP×B。真实副本还包括：TrialWorker→manager Sample、manager数组、object store partition、各节点拉取、TP×CP物理PP-last进程私有CPU数组、old-head、转换临时数组、Harbor保留的base64 extra。不能只乘DP或假设跨进程所有副本均零拷贝。

base64约二进制4/3，Harbor result.json持久化也增加磁盘占用。这不是tensor_store spill，但必须计容量。单请求预算256MiB，单LLM实例累计预留512MiB（失败/取消保留预留，保守失败），batch打包8GiB。**这些不是整个job统一RAM硬上限**：多LLM实例/子代理、并发trial、在途rollout须乘实际数量；batch门禁发生于收集后，不能防止收集前OOM。生产先低concurrency、k32，再依据峰值扩量。

SC GPU softmax workspace约C×Vp×4，C256、Vp62080约60.6MiB。但还有int64[N,k] IDs、float32[N,k] head/系数、索引及**完整梯度T×Vp×4**。T32768、Vp62080时仅该梯度约7.58GiB；不能沿用“只增加0.5GiB”的结论。T须取本rank packed/padded shape。old-head no_grad不留新反向图，仍有CP全response临时通信与CPU副本。

上线前测host RSS/PSS、object-store used/spilled、网络流量、GPU peak allocated/reserved；本方案发布门槛为最大观测峰值保留至少20%余量、无持续spill增长。容量不足则降低k/并发/在途rollout/microbatch token限额；不得无验证将logp改fp16或ID改int16。

# 6. 后续实施阶段验收步骤（待批准后执行的 Stage 1 ~ Stage 4）

本轮只生成文档。以下工具与测试也只交付代码块，源码修改、测试文件创建、shadow构建或集群执行均留待后续批准。

## 6.1 Stage 1：冻结基线与配置快照

1. 保存源码hash、git diff --binary HEAD（包括已暂存）、untracked清单、两脚本与父脚本hash；不通过reset获得“干净基线”。SGLang备份含完整包。
2. 重新只读核对upstream origin HEAD；与8ee9c1e1不同须逐提交审阅，不能直接pull覆盖规格。
3. 固定run tag和同一scratch root采before/after。现有DRY_RUN=1会先mkdir、写resolved YAML，故仅在scratch路径执行；它不会产生完整TRAIN_CMD，须补采内部argv。
4. 比较resolved parent环境和inner array构造的argv/runtime_env。节点IP discovery是动态数据，工具明确排除；真实提交前另行核验runtime_env节点IP与shadow provenance。密钥不明文写入公共工件。
5. vars(args)快照在目标运行时独立进程中解析真实argv并JSON化；仅白名单本文SC keys，其余参数逐值相同。--dry-run不是原训练CLI参数，不能直接追加给train_async.py。

未来路径 tools/audit_ppo_sc_plan.py，完整只读源码重建器，无apply模式：

```python
"""Read the specification; validate anchors and compile proposed code in memory.

Usage: python -B tools/audit_ppo_sc_plan.py --repo REPO --plan PLAN
This utility contains no source write/apply mode.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def load_proposed(repo, plan):
    document = Path(plan).read_text()
    sources = {}
    fence = chr(96) * 3
    pattern = re.compile(r"<!-- SC-SPEC (\{[^\n]+\}) -->\n" + fence + r"python\n(.*?)\n" + fence, re.S)
    records = list(pattern.finditer(document))
    if not records:
        raise ValueError("No executable specification records found")
    for index, record in enumerate(records):
        descriptor = json.loads(record.group(1))
        path = descriptor["path"]
        if path.startswith("/") or ".." in Path(path).parts:
            raise ValueError("Invalid relative specification path")
        if path not in sources:
            source = Path(repo, path)
            text = source.read_text() if source.exists() else ""
            if hashlib.sha256(text.encode()).hexdigest() != descriptor["sha256"]:
                raise ValueError(f"Source drift detected: {path}")
            sources[path] = text
        content = record.group(2)
        kind = descriptor["kind"]
        if kind == "full":
            sources[path] = content + "\n"
        elif kind == "before":
            continue
        elif kind == "after":
            if index == 0:
                raise ValueError("After record has no Before")
            previous = records[index-1]
            before_descriptor = json.loads(previous.group(1))
            if before_descriptor["kind"] != "before" or before_descriptor["id"] != descriptor["id"]:
                raise ValueError("Unpaired Before/After record")
            before = previous.group(2)
            if sources[path].count(before) != 1:
                raise ValueError(f"Ambiguous/missing patch anchor: {path} {descriptor['id']}")
            sources[path] = sources[path].replace(before, content, 1)
        elif kind == "append":
            sources[path] += "\n" + content + "\n"
        else:
            raise ValueError("Unknown specification record")
    for path, text in sources.items():
        tree = ast.parse(text, filename=path)
        if any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
               and node.value.value is Ellipsis for node in ast.walk(tree)):
            raise ValueError(f"Ellipsis placeholder statement in proposed source: {path}")
        compile(tree, path, "exec")
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    sources = load_proposed(args.repo, args.plan)
    print(json.dumps({
        path: {"lines": len(text.splitlines()), "sha256": hashlib.sha256(text.encode()).hexdigest()}
        for path, text in sources.items()
    }, indent=2))


if __name__ == "__main__":
    main()
```


未来路径 tools/snapshot_ppo_sc_launch.py，实现父级环境获取与真实inner argv构造：

```python
"""Stage 1 only: capture parent defaults and reconstruct the real inner argv."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    repo = Path(options.repo).resolve()
    script = repo / options.script
    scratch = Path(options.scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        DRY_RUN="1", BACKGROUND="0", CLEAN_START="0", USE_EXISTING_RAY_HEAD="1",
        BASE_FOLDER=str(scratch / "base"), RUN_TRIALS_DIR=str(scratch / "trials"),
        SLIME_DIR=str(repo), PYTHONDONTWRITEBYTECODE="1",
    )
    with tempfile.TemporaryDirectory(prefix="ppo-sc-snapshot-") as temporary:
        temporary = Path(temporary)
        capture = temporary / "environment.bin"
        hook = temporary / "bash_env"
        hook.write_text(
            "set -T\n"
            "_sc_capture() {\n"
            "  if [[ \"$BASH_COMMAND\" == 'echo \"[dry-run] all pre-flight checks passed; exiting without launching.\"' ]]; then\n"
            "    (umask 077; env -0 > \"$SC_CAPTURE_ENV\")\n"
            "  fi\n"
            "}\n"
            "trap _sc_capture DEBUG\n"
        )
        env.update(BASH_ENV=str(hook), SC_CAPTURE_ENV=str(capture))
        completed = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
        if completed.returncode or not capture.exists():
            raise SystemExit("Parent dry-run failed or changed its escape anchor; inspect preflight locally")
        resolved = {}
        for item in capture.read_bytes().split(b"\0"):
            if item:
                key, value = item.split(b"=", 1)
                resolved[key.decode()] = value.decode()
        resolved.pop("BASH_ENV", None)
        resolved.pop("SC_CAPTURE_ENV", None)
        resolved["CLEAN_START"] = "0"
        resolved["RUN_ROLE"] = "launcher"
        launcher = (repo / "examples/harbor_terminal_rl/run_qwen35_9b_terminal_bench2_64h20.sh").read_text()
        def part(begin, end):
            if launcher.count(begin) != 1 or launcher.count(end) != 1:
                raise ValueError("Launcher snapshot anchors changed; re-review the harness")
            lo = launcher.index(begin)
            hi = launcher.index(end, lo)
            return launcher[lo:hi]
        prefix = part("RUN_ROLE=${RUN_ROLE:-launcher}", 'if [ "${CLEAN_START}" = "1" ]; then')
        arrays = part('source "${MODEL_SCRIPT}"', 'mkdir -p "${RAY_TEMP_DIR}"\nmkdir -p "$(dirname "${WORKER_STOP_MARKER}")"')
        runtime = part('export no_proxy="${JOB_NO_PROXY}"', "_XTRACE_WAS_ON=0")
        argv_path = temporary / "argv.bin"
        runtime_path = temporary / "runtime.json"
        program = (
            "set -e\n"
            + "SCRIPT_DIR=" + shlex.quote(str(script.parent)) + "\n"
            + prefix + "\ncd " + shlex.quote(str(repo)) + "\n"
            + 'SAVE_DIR="$' + '{BASE_FOLDER%/}/$' + '{SAVE_SUBDIR}"\n'
            + arrays
            + '\nJOB_NO_PROXY="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"\n'
            + 'if [ -n "${no_proxy:-}" ]; then JOB_NO_PROXY="${JOB_NO_PROXY},${no_proxy}"; fi\n'
            + runtime
            + "\nprintf '%s\\0' \"${TRAIN_CMD[@]}\" > " + shlex.quote(str(argv_path)) + "\n"
            + "printf '%s' \"${RUNTIME_ENV_JSON}\" > " + shlex.quote(str(runtime_path)) + "\n"
        )
        subprocess.run(["bash"], input=program, env=resolved, text=True, check=True, capture_output=True)
        argv = [value.decode() for value in argv_path.read_bytes().split(b"\0") if value]
        runtime_env = json.loads(runtime_path.read_text())
        for key, value in runtime_env["env_vars"].items():
            if any(word in key.upper() for word in ("TOKEN", "API_KEY", "PASSWORD", "SECRET")):
                runtime_env["env_vars"][key] = {
                    "present": bool(value), "sha256": hashlib.sha256(value.encode()).hexdigest()
                }
        result = {
            "script": options.script, "argv": argv, "runtime_env_without_dynamic_ray_ips": runtime_env,
            "scope": "parent dry-run plus exact inner argument assembly; Ray node-IP discovery excluded",
        }
        Path(options.output).write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```


该工具依赖精确源码锚点；失败须重审，不得跳过。真实路径/环境preflight失败记录为未完成，不伪造占位路径。工具尚未运行真实父脚本。

## 6.2 Stage 2：Synthetic Tensor、Token与CPU分布式

未来路径 tests/sc_spec_synthetic.py，完整核心验算程序：

```python
"""Read-only spec verification runner. Does not write source or initialize a training job."""
import argparse
import ast
import json
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from tools.audit_ppo_sc_plan import load_proposed
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sources = load_proposed(args.repo, args.plan)
    module = types.ModuleType("slime.utils.score_centering")
    exec(compile(sources["slime/utils/score_centering.py"], "<SC-spec>", "exec"), module.__dict__)
    torch.manual_seed(117)
    logits = torch.randn(7, 11, dtype=torch.float64, requires_grad=True)
    rows = torch.tensor([0, 2, 4, 6])
    ids = torch.tensor([[0, 2, 5], [1, 3, 7], [3, 5, 10], [2, 6, 9]])
    weights = torch.randn(4, 3, dtype=torch.float64)
    for chunk in (1, 2, 256):
        actual = module.calculate_head_log_probs(logits, rows, ids, chunk_size=chunk)
        expected = logits.log_softmax(-1)[rows].gather(-1, ids)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        ga = torch.autograd.grad((actual * weights).sum(), logits, retain_graph=True)[0]
        ge = torch.autograd.grad((expected * weights).sum(), logits, retain_graph=True)[0]
        torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-12)
    p = torch.tensor([[.25, .3], [.4, .2]], dtype=torch.float64).log().requires_grad_()
    q = torch.tensor([[.2, .2], [.2, .3]], dtype=torch.float64).log().requires_grad_()
    adv = torch.tensor([2., -3.], dtype=torch.float64, requires_grad=True)
    correction, _ = module.score_centering_correction(
        train_head_logp=p, sampler_head_logp=q, advantages=adv)
    gp, gq, ga = torch.autograd.grad(
        (adv.detach() * correction).sum(), (p, q, adv), allow_unused=True)
    alpha = (1-q.detach().exp().sum(-1)) / (1-p.detach().exp().sum(-1)).clamp_min(1e-8)
    expected = adv.detach()[:, None] * (q.detach().exp()-alpha[:, None]*p.detach().exp())
    torch.testing.assert_close(gp, expected, atol=1e-12, rtol=1e-12)
    assert gq is None and ga is None
    for sign in (-1., 1.):
        ratio = torch.tensor([.5, .79, .8, .81, 1., 1.19, 1.2, 1.21, 3.],
                             dtype=torch.float64, requires_grad=True)
        surrogate = torch.maximum(-sign*ratio, -sign*ratio.clamp(.8, 1.2))
        observed = torch.autograd.grad(surrogate.sum(), ratio)[0] * ratio.detach() / (-sign)
        expected = module.ppo_effective_weight(ratio.detach(), torch.tensor(sign), .2, .2)
        torch.testing.assert_close(observed, expected)
    lp = torch.tensor([[float("nan"), float("inf")], [-1., -2.]], requires_grad=True)
    out, _ = module.score_centering_correction(
        train_head_logp=lp, sampler_head_logp=lp.detach(), advantages=torch.ones(2),
        loss_mask=torch.tensor([False, True]))
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(lp.grad).all() and lp.grad[0].eq(0).all()
    tokens = np.array([3, 4], dtype="<i4")
    head_ids = np.array([[3, 0], [4, 0]], dtype="<i4")
    head_lp = np.array([[-.5, -10.], [-.5, -10.]], dtype="<f4")
    envelope = module.encode_turn_head(head_ids, head_lp, tokens, [1, 2])
    decoded = module.decode_turn_head(envelope, [1, 2], tokens.tolist(), [-.5, -.5], k=2)
    assert all(np.array_equal(x, y) for x, y in zip(decoded, (head_ids, head_lp, tokens)))
    print("Synthetic tensor, detach, clip-boundary, masked NaN and codec audit: PASS")


if __name__ == "__main__":
    main()
```


未来路径 tests/sc_tp_smoke.py；落地后执行 torchrun --standalone --nproc-per-node=2 tests/sc_tp_smoke.py：

```python
"""Run after implementation: torchrun --standalone --nproc-per-node=2 tests/sc_tp_smoke.py"""
import os
import torch
import torch.distributed as dist

from slime.utils.score_centering import calculate_head_log_probs

dist.init_process_group("gloo")
torch.set_num_threads(1)
rank, world = dist.get_rank(), dist.get_world_size()
generator = torch.Generator().manual_seed(20260926)
full = torch.randn(7, 8 * world, dtype=torch.float64, generator=generator)
rows = torch.tensor([0, 2, 4, 6])
ids = torch.tensor([[0, 8 * world - 1], [2, 7], [4, 9], [6, 10]])
coefficients = torch.randn(4, 2, dtype=torch.float64, generator=generator)
for chunk in (1, 3, 256):
    local = full[:, rank * 8:(rank + 1) * 8].clone().requires_grad_()
    actual = calculate_head_log_probs(local, rows, ids, dist.group.WORLD, chunk)
    reference_input = full.clone().requires_grad_()
    expected = reference_input.log_softmax(-1)[rows].gather(-1, ids)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    torch.testing.assert_close(
        local.grad, reference_input.grad[:, rank * 8:(rank + 1) * 8],
        atol=1e-12, rtol=1e-12,
    )
if rank == 0:
    print("TP forward/backward and chunk invariance: PASS")
dist.destroy_process_group()
```


以下为本轮实际内存测试的完整程序：从stdin读取JSON字典“相对路径→拟议完整源码”，可由load_proposed生成。每项在独立python -B进程中执行，不写源码，不把mpu测试桩带入训练。

### Token / Harbor对照程序

```python
import sys,json,types,copy,ast
from pathlib import Path
sources=json.load(sys.stdin)
import torch,numpy as np
torch.set_num_threads(1)
root=Path.cwd()
sc=types.ModuleType("slime.utils.score_centering")
exec(compile(sources["slime/utils/score_centering.py"],"<spec-sc>","exec"),sc.__dict__)
sys.modules[sc.__name__]=sc
import slime_plugins.harbor.atif_adapter as original
new=types.ModuleType("slime_plugins.harbor.atif_adapter")
exec(compile(sources["slime_plugins/harbor/atif_adapter.py"],"<spec-atif>","exec"),new.__dict__)
from slime.utils.types import Sample
fixtures=[
({"prompt_token_ids":[[10,11,900,901],[10,11,20,30,40,900,901]],"completion_token_ids":[[20,30],[50]],"logprobs":[[-.2,-.3],[-.5]]},"trajectory_normalized"),
({"prompt_token_ids":[[10,900,901],[10,20,30,40,900,901]],"completion_token_ids":[[777,20,30],[50]],"logprobs":[[-.7,-.2,-.3],[-.5]]},"trajectory_normalized"),
({"prompt_token_ids":[[10,900,901],[10,20,30,40,900,901]],"completion_token_ids":[[20,30,777],[50]],"logprobs":[[-.2,-.3,-.7],[-.5]]},"trajectory_normalized"),
({"prompt_token_ids":[[10,11,900,901],[70,71,900,901]],"completion_token_ids":[[20],[80]],"logprobs":[[-.2],[-.8]]},"trajectory_normalized"),
({"prompt_token_ids":[[10,11,900,901],[10,11,200,40,900,901]],"completion_token_ids":[list(range(100,140)),[50]],"logprobs":[[-i/100 for i in range(100,140)],[-.5]],"extra":{"completion_text":["same","next"],"last_assistant_message_content":[None,"same"],"last_assistant_message_span":[None,[2,3]]}},"trajectory_retokenized"),
]
for detail,mode in fixtures:
  kw=dict(reward=1.,status=Sample.Status.COMPLETED,mode=mode)
  baseline=original.samples_from_rollout_details([copy.deepcopy(detail)],Sample(index=1),**kw)
  off=new.samples_from_rollout_details([copy.deepcopy(detail)],Sample(index=1),**kw)
  assert [s.to_dict() for s in baseline]==[s.to_dict() for s in off]
  enriched=copy.deepcopy(detail);envelopes=[]
  for prompt,tokens,lps in zip(detail["prompt_token_ids"],detail["completion_token_ids"],detail["logprobs"],strict=True):
    t=np.array(tokens,dtype="<i4")
    ids=np.stack([t,np.zeros_like(t)],axis=1)
    logps=np.stack([np.array(lps,dtype="<f4"),np.full(len(t),-10,dtype="<f4")],axis=1)
    envelopes.append(sc.encode_turn_head(ids,logps,t,prompt))
  enriched.setdefault("extra",{})["score_centering_head"]=envelopes
  on=new.samples_from_rollout_details([enriched],Sample(index=1),score_centering_top_k=2,**kw)
  for old,updated in zip(off,on,strict=True):
    d=updated.to_dict()
    for k in sc.HEAD_KEYS: d.pop(k)
    assert d==old.to_dict()
    assert np.array_equal(updated.sc_response_token_ids,updated.tokens[-updated.response_length:])
    recovered=Sample.from_dict(updated.to_dict())
    assert np.array_equal(recovered.rollout_topk_token_ids,updated.rollout_topk_token_ids)
  corrupt=copy.deepcopy(enriched)
  corrupt["extra"]["score_centering_head"][0]["digest"]="0"*64
  try: new.samples_from_rollout_details([corrupt],Sample(),score_centering_top_k=2,**kw)
  except ValueError: pass
  else: raise AssertionError("corrupt digest accepted")
print("ADAPTER_5_FIXTURES_OFF_ON_DIGEST_ROUNDTRIP_OK")
sys.modules["slime_plugins.harbor.atif_adapter"]=new
import pytest
status=pytest.main(["-q","-p","no:cacheprovider","tests/test_harbor_atif_adapter.py"])
raise SystemExit(status)
```


### 真正hard cap六轨程序

```python
import sys,json,types,copy
sources=json.load(sys.stdin)
import torch,numpy as np
torch.set_num_threads(1)
for name,path in (
 ("slime.utils.score_centering","slime/utils/score_centering.py"),
 ("slime_plugins.harbor.atif_adapter","slime_plugins/harbor/atif_adapter.py"),
 ("slime_plugins.harbor.sample_postprocess","slime_plugins/harbor/sample_postprocess.py"),
):
 m=types.ModuleType(name);exec(compile(sources[path],path,"exec"),m.__dict__);sys.modules[name]=m
from slime.utils.types import Sample
from slime_plugins.harbor.sample_postprocess import cap_total_tokens_allow_response_trim_in_samples,cap_total_tokens_allow_response_trim_in_rollout_data
s=Sample(tokens=[8,9,10,11,12,13,14],response_length=5,loss_mask=[1]*5,rollout_log_probs=[-.5]*5)
s.rollout_topk_token_ids=np.array([[t,0] for t in range(10,15)],dtype="<i4")
s.rollout_topk_log_probs=np.array([[-.5,-10]]*5,dtype="<f4")
s.sc_response_token_ids=np.array(range(10,15),dtype="<i4")
args=types.SimpleNamespace(enable_score_centering=True,harbor_max_train_total_tokens=4)
cap_total_tokens_allow_response_trim_in_samples(args,[[s]])
assert s.tokens==[11,12,13,14] and s.response_length==3
assert s.sc_response_token_ids.tolist()==[12,13,14] and s.rollout_topk_token_ids[:,0].tolist()==[12,13,14]
cap_total_tokens_allow_response_trim_in_rollout_data(args,0,dict(total_lengths=[4]))
try: cap_total_tokens_allow_response_trim_in_rollout_data(args,0,dict(total_lengths=[5]))
except ValueError: pass
else: raise AssertionError("Late trim accepted")
print("TRUE_RESPONSE_TRIM_AND_LATE_TRIM_GUARD_OK")
```


### CP映射与原get_responses对照程序

```python
import sys,json,types,ast
sources=json.load(sys.stdin)
import torch
torch.set_num_threads(1)
from megatron.core import mpu
sc=types.ModuleType("slime.utils.score_centering")
exec(compile(sources["slime/utils/score_centering.py"],"<spec-sc>","exec"),sc.__dict__)
sys.modules[sc.__name__]=sc
terms=types.ModuleType("slime.backends.megatron_utils.sc_terms")
exec(compile(sources["slime/backends/megatron_utils/sc_terms.py"],"<spec-terms>","exec"),terms.__dict__)
# Execute the exact existing response iterator with real torch and cp_utils.
ns={"torch":torch,"mpu":mpu}
import slime.backends.megatron_utils.cp_utils as cp
ns["get_logits_and_tokens_offset_with_cp"]=cp.get_logits_and_tokens_offset_with_cp
tree=ast.parse(open("slime/backends/megatron_utils/loss.py").read())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="get_responses")
mod=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),node],type_ignores=[])
exec(compile(ast.fix_missing_locations(mod),"<original-get-responses>","exec"),ns)
count=0
for fmt in ("thd","bshd"):
 for size in (1,2,4):
  for gather in (False,True):
   if gather and fmt=="bshd":continue
   for rank in range(size):
    mpu.get_context_parallel_world_size=lambda:size
    mpu.get_context_parallel_rank=lambda:rank
    totals=[7,13,8];responses=[2,9,1];maximum=16
    if fmt=="bshd":shape=(3,maximum//size,5)
    elif gather:shape=(1,((sum(totals)+size-1)//size),5)
    elif size==1:shape=(1,sum(totals),5)
    else:shape=(1,sum(2*((t+2*size-1)//(2*size)) for t in totals),5)
    logits=torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
    tokens=[torch.arange(t) for t in totals]
    args=types.SimpleNamespace(qkv_format=fmt,allgather_cp=gather,rollout_temperature=1.)
    batch=dict(total_lengths=totals,response_lengths=responses,unconcat_tokens=tokens,max_seq_lens=[maximum]*3 if fmt=="bshd" else None)
    flat,layouts=terms.response_row_layout(args,batch,logits)
    baseline=ns["get_responses"](logits,args=args,unconcat_tokens=tokens,total_lengths=totals,response_lengths=responses,max_seq_lens=batch["max_seq_lens"])
    for i,((rows,idx),(values,tok)) in enumerate(zip(layouts,baseline,strict=True)):
      assert torch.equal(flat[rows],values),(fmt,size,gather,rank,i)
      assert torch.equal(tokens[i][-responses[i]:][idx],tok)
    count+=1
print("CP_LAYOUT_CASES_OK",count)
```


### CP2 old-policy producer、两种公式与两种布局程序

```python
import sys,json,types,ast,socket
sources=json.load(sys.stdin)
import torch,torch.distributed as dist,torch.multiprocessing as mp
torch.set_num_threads(1)
from megatron.core import mpu
import slime.backends.megatron_utils.cp_utils as cp
sc=types.ModuleType("slime.utils.score_centering")
exec(compile(sources["slime/utils/score_centering.py"],"<spec-sc>","exec"),sc.__dict__)
sys.modules[sc.__name__]=sc
terms=types.ModuleType("slime.backends.megatron_utils.sc_terms")
exec(compile(sources["slime/backends/megatron_utils/sc_terms.py"],"<spec-terms>","exec"),terms.__dict__)
# Isolate only the existing redistribution implementation; no CUDA model import.
fake_loss=types.ModuleType("slime.backends.megatron_utils.loss")
ns=fake_loss.__dict__;ns.update(torch=torch,dist=dist,mpu=mpu,F=torch.nn.functional,slice_log_prob_with_cp=cp.slice_log_prob_with_cp)
tree=ast.parse(open("slime/backends/megatron_utils/loss.py").read())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_allgather_cp_redistribute")
module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),node],type_ignores=[])
exec(compile(ast.fix_missing_locations(module),"<original-redistribute>","exec"),ns)
sys.modules[fake_loss.__name__]=fake_loss
sock=socket.socket();sock.bind(("127.0.0.1",0));port=sock.getsockname()[1];sock.close()
def worker(rank):
 dist.init_process_group("gloo",init_method=f"tcp://127.0.0.1:{port}",rank=rank,world_size=2)
 mpu.get_context_parallel_world_size=lambda:2
 mpu.get_context_parallel_rank=lambda:rank
 mpu.get_context_parallel_group=lambda:dist.group.WORLD
 mpu.get_tensor_model_parallel_group=lambda:None
 gen=torch.Generator().manual_seed(93)
 totals=[7,11];responses=[3,6];vocab=13;k=3
 tokens=[torch.arange(t)%vocab for t in totals]
 full=[torch.randn(t,vocab,generator=gen) for t in totals]
 old=[torch.randn(t,vocab,generator=gen) for t in totals]
 qfull=[torch.randn(r,vocab,generator=gen).log_softmax(-1) for r in responses]
 ids=[torch.stack([tok[-r:],(tok[-r:]+1)%vocab,(tok[-r:]+2)%vocab],-1) for tok,r in zip(tokens,responses)]
 q=[lp.gather(-1,ix) for lp,ix in zip(qfull,ids)]
 masks=[torch.tensor([1,0,1]),torch.tensor([1,1,0,1,1,1])]
 adv=[torch.tensor([1.,-2.,3.]),torch.tensor([-1.,2.,-3.,1.,-2.,3.])]
 def layout(values,gather):
  if gather:
   flat=torch.cat(values);pad=(-len(flat))%2
   return torch.cat([flat,flat.new_zeros((pad,vocab))]).chunk(2)[rank][None].clone()
  parts=[]
  for value,t in zip(values,totals):
   c=(t+3)//4
   padded=torch.cat([value,value.new_zeros((4*c-t,vocab))])
   parts.extend([padded[rank*c:(rank+1)*c],padded[(3-rank)*c:(4-rank)*c]])
  return torch.cat(parts)[None].clone()
 for gather in (False,True):
  args=types.SimpleNamespace(qkv_format="thd",allgather_cp=gather,sc_top_k=k,sc_chunk_size=2,
   sc_formula="ppo_weighted",sc_weight_denominator="old_actor",eps_clip=.2,eps_clip_high=.2,sc_tail_eps=1e-8)
  batch=dict(total_lengths=totals,response_lengths=responses,unconcat_tokens=tokens,loss_masks=masks,
   rollout_topk_token_ids=ids,rollout_topk_log_probs=q,sc_response_token_ids=[t[-r:].int() for t,r in zip(tokens,responses)],
   advantages=[cp.slice_log_prob_with_cp(x,t,r) for x,t,r in zip(adv,totals,responses)],
   rollout_log_probs=[cp.slice_log_prob_with_cp(lp.gather(-1,tok[-r:,None]).squeeze(-1),t,r) for lp,tok,t,r in zip(qfull,tokens,totals,responses)])
  callback=terms.old_policy_head_callback(lambda logits:(torch.empty(0),{}),args,batch)
  _,old_result=callback(layout(old,gather))
  batch.update(old_result)
  for o,actual,ix,t,r,mask in zip(old,old_result["sc_old_head_log_probs"],ids,totals,responses,masks):
   expected=o[t-r-1:t-1].log_softmax(-1).gather(-1,ix)
   torch.testing.assert_close(actual[mask.bool()],expected[mask.bool()],atol=1e-6,rtol=1e-6)
  for formula in ("score","ppo_weighted"):
   args.sc_formula=formula
   logits=layout(full,gather).requires_grad_()
   addend,metrics=terms.get_score_centering_terms(args,batch,logits,advantages=torch.cat(batch["advantages"]))
   local_loss=addend.sum()
   local_loss.backward()
   refs=[]
   for f,o,ql,ix,av,mask,t,r in zip(full,old,q,ids,adv,masks,totals,responses):
    lp=f[t-r-1:t-1].log_softmax(-1).gather(-1,ix)
    ol=o[t-r-1:t-1].log_softmax(-1).gather(-1,ix)
    c,_=sc.score_centering_correction(train_head_logp=lp,sampler_head_logp=ql,advantages=av,
     formula=formula,old_head_logp=ol,loss_mask=mask)
    refs.append(av*c)
   expected=torch.cat([cp.slice_log_prob_with_cp(x,t,r) for x,t,r in zip(refs,totals,responses)])
   torch.testing.assert_close(addend,expected,atol=1e-5,rtol=1e-5)
   assert torch.isfinite(logits.grad).all()
 dist.destroy_process_group()
mp.start_processes(worker,nprocs=2,start_method="fork",join=True)
print("CP2_BOTH_LAYOUTS_BOTH_FORMULAS_OLD_HEAD_AND_BACKWARD_OK")
```


实施后还须覆盖：binary/json往返、坏base64/字节数、重复/超词表ID、missing keys、错误digest、response0、k1、全mask、质量≈1、错误old分母、YAML-only enable、role覆盖、off Sample pickle/schema、R3行、DP uneven partitions、cap后group_mask_sums、每轮reset、非PP-last正常跳过。该补充清单不能标成已运行。

## 6.3 Stage 3：单Step分布式小批梯度验收

使用固定checkpoint、固定已采rollout、RNG、TP/CP/PP/DP与microbatch schedule。先小模型完整机制，再目标拓扑单step；不能比较两次重新采样的在线轨迹来证明bitwise。

A为本轮冻结原工作区；B为实施后SC关闭；C为真实head+scale0；D为scale1。各组使用相同sample IDs、tokens、mask、advantages、old/ref/critic logprobs。

A/B要求loss、全部parameter gradient、optimizer step后参数及state逐bit一致；SC GPU allocation/collective计数为0。若环境A/A重跑也不bitwise，必须明确缺少bitwise证据，不改allclose后继续声称100%。

B/C要求相同输入下loss/gradient/step参数逐bit一致；允许新增forward诊断与通信，禁止SC可微图和loss加法。C必须真实q，trainer合成q无法验证HTTP/Harbor链路。

D比较grad(D)−grad(B)与冻结SC系数的独立解析梯度；小词表全分布参考分别验证score与weighted/old_actor。CP按全局token身份重建比较，不比较局部rank位置。gloo测试不能替代NCCL、PP/VPP、R3和真实FP32 kernel。

未来路径 tests/compare_ppo_sc_step.py，完整参数比较器：

```python
"""Compare trusted locally produced single-step artifacts."""
import argparse
import torch


def compare(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
            raise AssertionError(f"Tensor contract mismatch at {path}")
        if not torch.equal(a, b):
            raise AssertionError(f"Bitwise tensor mismatch at {path}")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(f"Key mismatch at {path}")
        for key in a:
            compare(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError(f"Sequence mismatch at {path}")
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            compare(left, right, f"{path}/{index}")
    elif a != b:
        raise AssertionError(f"Value mismatch at {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    required = {"loss", "gradients", "parameters", "optimizer", "sample_ids", "rng"}
    if not required <= baseline.keys() or not required <= candidate.keys():
        raise ValueError("Incomplete step artifacts")
    for key in sorted(required):
        compare(baseline[key], candidate[key], key)
    print("Single-step bitwise comparison: PASS")


if __name__ == "__main__":
    main()
```


可信本地工件包含loss、gradients(name→tensor)、parameters、optimizer(平坦name→tensor/scalar)、sample_ids、rng。gradient在optimizer.step前clone，参数/state在step后采；A/B使用同一采集harness，不重新forward生成工件。完整训练工件采集依赖批准后的目标运行时，本文不伪造结果。

## 6.4 Stage 4：SGLang canary与扩量

1. 未改live，用wire=json、低并发和小max_new_tokens验证真实q；坏响应不能转合成q。JSON不是长context生产默认。
2. 构建完整shadow，校验所有节点文件hash与driver/actor/child的__file__。最小块只改top-logprobs输出编码，不改采样/scheduler/R3。不带top_logprobs_num的基准请求sample token/logp/R3须配对一致；先验证A/A可重复。
3. 同response binary与JSON头逐元素一致；return_text_in_logprobs=True/stream旧分支验证。SC客户端仍非stream。
4. 两个基准模型各从小批起；不能凭122B上下文短推断总内存安全。覆盖CP4、TP、PP、R3、partial completion、cap和weighted old_actor。
5. scale0完整链路后scale1配对实验。监控reward/eval、gradient norm、NaN、head质量分位数、alpha、conditioning_match比例、alignment错误、host/object-store/GPU峰值、耗时；不硬编码“k32必覆盖99%”。
6. 回退到SC-off配置与原PYTHONPATH；不回滚用户其他修改、不重启集群、不清理live SGLang。已SC更新的checkpoint不能作为原PPO基线checkpoint作零回归证明。

**只有Stage1–4真实工件通过后才能改上线结论为Pass。方案交付完成不意味着已经部署或生产验收完成。**

## 6.5 本轮最终文档审计与零源码修改证明

本轮最终从已写入的Markdown重新提取83条SC-SPEC记录，按真实源码SHA256和精确Before/After重建**15个拟议模块、12336行源码**，全部AST解析/compile通过。另有**18个可独立编译的Python代码块**通过语法检查；局部块在完整文件重建后检查，不把缩进片段的单独编译错误当作实现错误。保留源码中的既有类型注解Ellipsis（例如tuple[Any, ...]），拒绝作为语句的未实现Ellipsis。

实际测试结果：核心Synthetic Tensor/冻结系数解析梯度/clip边界/NaN-mask/codec通过；Harbor五组新增轨迹、动态字段往返、错误digest拦截通过；现有adapter 18/18通过；真实response hard trim与晚期trim守卫通过；CP21组映射通过；CP2两种布局、两种公式、old-head生产者与反向通过；TP2三个chunk配置的前向与梯度通过。部分环境导入出现既有torchao/modelopt/docstring提示，测试仍正常退出；原arguments.py的invalid escape sequence警告为既有源码内容，本文没有借机修改。

零源码修改最终核验：对git ls-files --cached --others --exclude-standard列出的.py/.sh/.patch逐文件计算SHA256，再对“绝对路径→hash”排序JSON求hash。以下**本轮读取前/写文档后完全一致**：

| 范围 | 文件数 | 聚合SHA256（前=后） |
|---|---:|---|
| slime-terminal-RL源码与patch | 645 | 814213f6208f7e372b72684e040cdd9d89f1dbe890b1880a4047bebb271132c7 |
| /sgl-workspace/sglang源码与patch | 3426 | d5ff135b941240df92c3b040830d34d5f84b79486a7a294dc03c4cad3f78cde8 |

工作区原有dirty/untracked状态保留；没有git add/commit/push、没有patch apply、没有改写训练脚本。**本轮唯一写入成果为本文件PPO_SC_INTEGRATION_PLAN.md。** 本阶段已停止于文档交付，等待人工审查；Stage1–4均未在生产集群执行。

远端核验记录：已确认本地上游含#2405/#2406/#2407；GitHub页面搜索未返回可用内容，API curl在10秒超时，git ls-remote连接失败，因此远端2026-09-26最新HEAD仍未获证实。本文所有实现判断均来自已读本地源码与本輪内存测试，没有把历史论文“开销<1%”“k32零精度代价”等未复核数字当作生产保证。

补充终检：从文档重建的LLM客户端经httpx.MockTransport验证，SC-off请求JSON字节与原版一致、完整LLMResponse相等；SC-on请求含正确top_logprobs_num且extra envelope可完整解码。影子SGLang的After块已替换到原文件字符串并compile通过，运行时文件未变；shadow构建工具与launch snapshot工具语法检查通过，尚未实际执行其环境写入流程。

