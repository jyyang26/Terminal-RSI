# TB-GAE 首期落地与离线验收报告

日期：2026-09-27。依据 `TB4_REWARD_TBGAE_PATCH.md` 第 9–10 节实施。

> **最新状态：Value Bootstrap 与跨 chunk continuation 已落地（2026-09-29）。**
> max_turns/可取得后继状态的 timeout 不再因为 `truncated` 被一律丢弃或拒绝：
> 保存真实后继输入、更新前 critic 前向、跨 DP chunk 边界递推已接通。
> 本轮 **338 项离线测试通过**，包括 DP=2×CP=4 的 Gloo 验证；详见第 7 节。
> 下文第 1–6 节“尚未实现 bootstrap / 生产全量不能启用”的描述是此前阶段记录。
> **代码级截断/多 chunk 阻断已解除；未执行 27B/256k/PP2 的 GPU 集群 smoke，不能把离线通过等同上线验收。**

> **二次复核后的追加修复（2026-09-29）：**此前上层 trial 容错确实会吞掉 TB 转换异常并丢样本，
> “没有静默丢弃”的旧表述不准确，现已纠正。新增门控让 TB 开启时的错误/空样本直接失败，
> 不再靠筛掉失败轨迹凑 batch；本轮相关测试 **103/103 通过**。详见第 6 节。
> **这只是防止错误训练继续的修复，不是截断 bootstrap；mixed3000 生产全量仍暂不可启用。**

> **2026-09-29 修复更新：`termination_kind` 无生产者的阻塞已修复。**
> 实际使用的 Harbor Terminus-2 已在真实退出事件处记录原因，Slime 在 TB 开启时自动启用生产者，
> 并在创建 trial 前检测旧版 Harbor。新增真实 agent/Chat/parser/Factory/Trial smoke 已通过，
> 本轮共 **92 项通过**；具体代码、部署步骤和测试范围见第 5 节。
> **自然终止的单 chunk 通路可用；含 max_turns/timeout 截断或多 chunk 的生产全量仍不能直接启用。**
> 本轮没有实现这些非终止尾部的 bootstrap，没有启动集群训练。

**结果：首期数据通路和算法已实现，201/201 离线测试通过，0 失败、0 跳过。**
其中 TB 专项 117 项，既有 Harbor/DP/角色/SC 回归 84 项；最终整套耗时 285.91 秒。
仅使用 CPU/Gloo，没有启动集群训练、GPU/NCCL 测试或修改生产启动配置。

**首期历史启用边界（现行规则见第 7 节）：**本次实现文档定义的首期能力，即 THD、普通 zigzag CP=1/4、单 detail、
单 chunk、可信真实终止、KL reward 系数 0、SC/OPD 关闭。
9 月 27 日的版本缺少 `agent_result.metadata.termination_kind` 生产者；9 月 29 日已补齐，
但只有真实自然结束才输出 `terminated`。跨 chunk、外部截断 bootstrap、BSHD 尚未开放，
不能仅凭本报告宣称当前 fresh_base 全量训练已可直接启用。加载旧 Harbor 时会在 trial 创建前报错。
没有用 EOS/stop、成功 reward 或 COMPLETED 状态猜测终止，也没有静默过滤困难轨迹。

## 1. 文件清单

| 文件 | 状态与核心改动 |
|---|---|
| `slime/utils/tb_gae_config.py` | 新增：严格 bool；显式 CLI > 任务 YAML > 角色顶层 > 默认 False；两项轮次参数默认 1.0；角色局部 TB 参数禁用；校验所有有效角色配置 |
| `slime/utils/arguments.py` | 注册参数，保留显式 CLI 来源，统一 YAML 合并，创建 worker 前验证，开启时记录最终参数 |
| `slime_plugins/harbor/tb_gae_data.py` | 新增：CPU bool 紧凑 masks、span/终止验证、全样本收集、同步切片；未知终止/多 chunk 显式失败 |
| `slime_plugins/harbor/atif_adapter.py` | 三个入口增加 keyword-only `enable_tb_gae=False`；仅 True 记录实际输出 spans、生成两 masks |
| `slime_plugins/harbor/rollout.py` | `_run_one_trial` 仅训练且 True 时将开关传到 adapter；eval 不生成 TB masks |
| `slime_plugins/harbor/sample_postprocess.py` | True 时按同一 response slice 裁两 masks；GAE 后的 cap 只校验、不重裁 |
| `slime/ray/rollout.py` | True 时 convert 验证并提升 metadata；DP 沿原 partition 传两字段；原白名单和调度不变 |
| `slime_plugins/harbor/tb_gae.py` | 重构：标量 outcome 落最后有效 action；full masks + ragged local values；真实调用 CP wrapper；原生 slice；no_grad/detach；并行反向 affine scan |
| `slime/backends/megatron_utils/loss.py` | 原 custom/PPO 前增加 True 分支，两角色共享；原 PPO reward 构造与 GAE 调用未重构 |
| `slime/backends/megatron_utils/data.py` | True 时日志跳过两 masks |
| `tests/test_tb_gae_config.py` | 新增参数、严格 schema、实际 slime YAML 合并和角色测试 |
| `tests/test_tb_gae_zero_regression.py` | 新增冻结源码对照、Sample/cap/DP/payload、真实 loss/所有 logits 梯度/AdamW 更新，以及完整开启通路 |
| `tests/test_tb_gae_math.py` | 新增 S3/S4、顺序参考、终止/掩码/cap、奖励和双角色测试 |
| `tests/test_tb_gae_cp_smoke.py` | 新增实际 CPU Gloo CP=1/4 多进程验收 |
| `tests/tb_gae_test_utils.py`、`tests/tb_gae_cp_worker.py` | 测试辅助、冻结源码加载和分布式 worker |
| `tests/fixtures/tb_gae_pre_patch/` | 修改前工作区的 12 份源码快照及 SHA-256，不是从修改后代码生成的期望结果 |
| `tests/artifacts/tb_gae_20260927/` | 最终 JUnit、JSON 统计、源文件哈希和 309 个受保护脚本/配置的哈希 |

未改动 `Sample` dataclass、原生 `ppo_utils.py`、`cp_utils.py`、`dp_schedule.py`、
actor/model、现有 shell/YAML 配置、`/sgl-workspace/sglang` 或既有训练运行目录。
Python 测试均设置 `PYTHONDONTWRITEBYTECODE=1`，避免 import 时写入包目录的 pyc。

## 2. 测试结果与复现

下表时间来自最终 JUnit 各 test case 的累计时间，不包含全部 collection/进程启动开销。

| 套件 | 用例 | 累计秒数 | 通过率 |
|---|---:|---:|---:|
| `test_tb_gae_config.py` | 59 | 5.089 | 100% |
| `test_tb_gae_zero_regression.py` | 23 | 16.454 | 100% |
| `test_tb_gae_math.py` | 27 | 0.224 | 100% |
| `test_tb_gae_cp_smoke.py` | 8 | 256.096 | 100% |
| 既有 Harbor adapter/postprocess、DP、角色、SC 测试 | 84 | 1.009 | 100% |
| 合计 | **201** | **整套墙钟 285.91** | **100%** |

原始结果：`tests/artifacts/tb_gae_20260927/junit.xml`；结构化统计与实现哈希：`results.json`。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/tb_gae_inductor \
python -B -m pytest \
  tests/test_tb_gae_config.py \
  tests/test_tb_gae_zero_regression.py \
  tests/test_tb_gae_math.py \
  tests/test_tb_gae_cp_smoke.py \
  tests/test_harbor_atif_adapter.py \
  tests/test_harbor_sample_postprocess.py \
  tests/utils/test_megatron_role_config.py \
  tests/test_dp_schedule.py \
  tests/test_ppo_sc_args.py \
  tests/test_ppo_sc_pipeline.py \
  -o addopts='' -p no:cacheprovider -q \
  --junitxml=/tmp/tb_gae_final.xml --tb=short
```

环境：Python 3.12，PyTorch 2.11.0+cu129，pytest 9.0.3；测试设备 CPU，通信 Gloo。
Black/isort 检查通过；仅检查本次修改业务文件的 `git diff --check` 通过。
全仓库有原先存在的 docker patch 空白错误，未为此修改无关文件。

额外 CPU probe：262,144 个 action token、S3、终局 reward=0.7，
返回值最大解析误差 `3.331899642944336e-5 < 1e-4`，约 0.028 秒。
这是单条 CPU 算子探测，不是 GPU 性能承诺或生产吞吐测量。

## 3. Zero Regression 证据与范围

1. **真实旧源码。** 快照在本次改代码前捕获，包括工作区已有改动。
   测试先校验 SHA-256，再执行冻结 adapter/postprocessor/GAE dispatcher/loss；
   原 Ray 方法逐节点 AST 抽取执行，避开构造额外 Ray actor。
2. **缺省与显式 False。** 两种情况都对照旧源码。测试拦截 TB runtime 模块 import，
   确保关闭通路不加载算法或 mask helper，不新增 Sample/metadata/train_data/DP masks 键。
   Sample 序列化往返、hard cap 后内容、DP payload 键值以及 pickle 字节相同。
3. **原生调用与计算。** spy 确认原 `get_advantages_and_returns_batch` 仍被调用。
   对固定输入验证 RNG 状态不变，advantages/returns、core policy loss、生产 Harbor policy loss、
   critic value loss、指标、所有 logits 梯度、AdamW 参数与优化器状态逐字节一致。
   包含原生 gamma/lambda=1.0 与 0.97、归一化开/关。
4. **实际 CP 通信。** localhost Gloo CP=1/4 测试对比完整结果和原生切片长度/token 身份，
   覆盖非整除长度、极短 response、空 local response、S3/S4、all-action/interleaved observation。
   同一 CP 组还执行冻结旧源码与新关闭路径的 targets、真实 policy/critic loss 和所有 logits 梯度逐字节对比。
5. **未动原生代码。** `types.py/ppo_utils.py/cp_utils.py/dp_schedule.py` 与快照哈希一致；
   309 个既有 shell/YAML 文件编辑前后哈希一致。

这些证据说明**被测离线输入和 CPU/Gloo 环境**下的数值/梯度为逐 bit 一致，
不是未运行过的 GPU/NCCL、多节点在线训练和所有可能输入的数学证明。
关闭分支只新增 O(1) 布尔判断；没有新增 token 工作、tensor、数据通信或 mask 显存。
控制面参数解析和分支本身不能被描述为物理意义的零 CPU 指令。

### CP 原生既有边界

增强测试发现：某些极短/有 padding 的样本在 CP rank 0 的 response slice 为空，
旧 PPO dispatcher 仍做 `k[-1] += reward`，会 IndexError。为保持零回归，本次未修改这段旧行为。
TB gather/scan/slice 对这些极端样本全部验收；旧/新关闭路径的 loss/梯度对照选择
原生能够运行的输入（total/response 为 64/1、4096/2048）。这项范围没有被静默掩盖。

## 4. S3/S4 标准配置

> 此节保留最初的参数格式；**现行部署要求、协议 v2 和分阶段启动命令见第 7.6 节**。

**以下是参数格式示例，不是“当前生产全量可直接运行”的承诺。**
必须先部署第 5 节的 Harbor + Slime 修复并通过实际加载路径检查；当前仅支持自然终止的单 chunk。
生产课程会遇到 `max_turns`、timeout 或 split 时，仍需要 bootstrap/continuation 工程，
不能靠本节命令直接覆盖整个生产数据分布。
不要手工填写一个恒为 terminated 的任务参数来绕过终止验证；该值必须来自实际 agent 环境终止事件。

在独立的角色配置副本顶层写入，不能放在任一角色 `overrides` 下：

```yaml
# S3
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

S4 使用相同文件结构，仅将公共参数改成：

```yaml
enable_tb_gae: true
tb_gae_gamma_turn: 0.999
tb_gae_lambda_turn: 0.999
```

也可把这些键放入现有完整任务配置的副本顶层，任务 YAML 优先于角色公共项。
通过现有 `MEGATRON_CONFIG_PATH` / `CONFIG_TEMPLATE` 指向副本即可，不需改生产脚本。
未新建或修改实际生产 YAML；示例仅写在本报告中。

CLI 路径（没有执行；仅在上述支持条件成立后使用）：

```bash
# S3
EXTRA_MISC_ARGS="--no-load-optim --no-load-rng --enable-tb-gae --tb-gae-gamma-turn 1.0 --tb-gae-lambda-turn 1.0" \
  bash examples/harbor_terminal_rl/launch_27b_fresh_base_critic_warmup2.sh

# S4
EXTRA_MISC_ARGS="--no-load-optim --no-load-rng --enable-tb-gae --tb-gae-gamma-turn 0.999 --tb-gae-lambda-turn 0.999" \
  bash examples/harbor_terminal_rl/launch_27b_fresh_base_critic_warmup2.sh

# 显式回到原生；覆盖 YAML 中的 true
EXTRA_MISC_ARGS="--no-load-optim --no-load-rng --no-enable-tb-gae" \
  bash examples/harbor_terminal_rl/launch_27b_fresh_base_critic_warmup2.sh
```

不需要 `--custom-advantage-function-path`。保留 `--advantage-estimator ppo`；
critic 的通用 custom hook 清空规则不变，TB 由其之前的公共开关分支执行。
S3→S4 仅改两项轮次参数，不同时改 reward/warmup/数据。当前 fresh_base reward 不是本文历史 S2 reward，
比较时应明确所用基线。

---

## 附：修复前可运行性复核（2026-09-29，审阅者补充，保留历史）

> 以下准确描述本轮修复前的代码。该“生产者为 0”问题现已按第 5 节修复；
> 截断、跨 chunk、旧运行进程加载旧模块等条件仍需按第 5 节处理。

> 结论：**当前 Harbor 生产链路无法直接启用 TB-GAE。** 按第 4 节的 S3/S4 启动命令拉起，会通过全部
> 配置校验、正常进入训练，但在**第一次 rollout 转换**时对**每一条 trial** 抛 `ValueError` 而失败。
> 本报告第 9–13 行的"生产启用边界"已诚实地预告了这一点；下面把它定位到具体代码，并给出修复项。

### 阻塞点：`termination_kind` 无生产者（硬失败，非降级）

1. **消费端会硬失败。** `enable_tb_gae=True` 时，`slime_plugins/harbor/atif_adapter.py:91-95` 调用
   `trial_termination_kind(result)`，后者在 `slime_plugins/harbor/tb_gae_data.py:16-30` 读取
   `agent_result.metadata["termination_kind"]`，若不严格等于 `"terminated"` 直接抛
   `ValueError("TB-GAE requires explicit agent_result.metadata.termination_kind='terminated'…")`。
   `atif_adapter.py:826-827` 再次强制同一条件。

2. **生产端根本不写这个字段。** 在整个 Harbor 源码树中 `termination_kind` 的**生产者数量为 0**
   （`grep -rn termination_kind harbor/` 无命中）。Terminus-2 只在
   `harbor/src/harbor/agents/terminus_2/terminus_2.py:1662` 写入
   `context.metadata = {"n_episodes", "api_request_times_msec", "summarization_count"}`（外加可选
   `all_messages`），**没有** `termination_kind`。因此每条 trial 的 metadata 都缺该键。

3. **后果是整轮 rollout 失败。** 生产配置 `harbor_fail_on_conversion_error: true`，转换异常不会被
   吞掉，会中止该 rollout；不存在"退回原 GAE"的静默降级路径。

4. **离线测试没有覆盖这条真实通路。** `tests/test_tb_gae_math.py:121/139/159` 是**手工注入**
   `termination_kind="terminated"` 的 fixture，甚至专门断言"缺失时应抛错"（:139-140）。所以
   "201/201 通过"证明的是"代码在给定契约时正确、在缺契约时正确报错"，**不**证明"接上真实
   Harbor 数据即可端到端跑通"。报告第 3 节（"被测离线输入…不是…所有可能输入的数学证明"）已限定了
   这一范围，但第 4 节的 S3/S4 启动命令看起来可直接运行，容易造成"配置校验通过=可训练"的误判。

### 与 SC 的本质差别（为何 SC 可跑而 TB-GAE 不可）

SC 的上游契约是 **top-k logprobs**，即向 SGLang 请求 `top_logprobs_num=k`（`slime_plugins/harbor/llm.py:135-150`），
这是引擎**原生能力**且已在 2026-09-26 真实 GPU 冒烟中验证。TB-GAE 的上游契约是
`metadata.termination_kind`，是一个**目前无人写入**的字段。差别不在算法实现，而在数据契约能否被现有栈满足。

### 使其可运行所需的最小修复（三选一，按推荐排序）

- **（推荐）在真实终止事件处补生产者。** 由 Terminus-2/adapter 依据 agent 实际结束原因写入
  `context.metadata["termination_kind"]`：仅当轨迹是**自然完成/环境判定 done**（非达到 `max_turns`
  截断、非异常/取消）时才置 `"terminated"`；截断或未完成置其它值（如 `"truncated"`）。注意本报告
  第 116 行的告诫——该值**必须来自真实终止事件**，不能填一个"恒为 terminated"的常量绕过校验，否则
  会把被截断的困难轨迹当作真实终止来 bootstrap，污染优势估计。
- **实现 bootstrap 协议。** 第 12 行提到的"外部截断 bootstrap"尚未开放；实现后即可正确处理
  非终止尾部，而不必要求每条轨迹都自然终止。
- **暂缓 TB-GAE，先跑 SC。** SC 已具备可运行条件（见同目录 `SC_IMPLEMENTATION_REPORT_20260929.md`
  与新增的 `examples/harbor_terminal_rl/launch_27b_run2_warmcritic_sc.sh`）。

### 其它次要观察

- **单 chunk 限制**：`tb_gae_data.py:61-63` 要求 `is_last_chunk=True` 且单块；`trajectory_retokenized`
  下若一条 trial 因超长被切成多 chunk（本模型 256k + 229376 压缩阈值下可能发生），同样会抛
  "requires a complete terminated single chunk"。启用前需确认目标任务在 max_turns 课程较大时不会触发多 chunk。
- **子 agent**：`atif_adapter.py` 在 `include_subagents=True` 时对 TB-GAE 直接报错；需确保
  `harbor_include_subagent_rollouts=false`（当前默认满足）。
- **配置校验本身无误**：`validate_tb_gae_config`（`slime/utils/tb_gae_config.py:109-144`）对生产 argv
  （S3：ppo/thd/CP4/kl0/retokenized…）实测**通过**。问题不在校验，而在校验之后的运行期数据契约。

## 5. 2026-09-29：真实终止契约修复与 smoke 闭环

### 5.1 根因确认与结论

审阅者指出的“只测了手工字段 fixture”问题成立。本轮先用冻结的旧 Terminus-2
实际运行两次 task_complete，得到真实 AgentContext；开启 TB 的 adapter 仍因
`kind=None` 报错，**不依赖手工造缺字段样本复现**。

生产启动链默认 `HARBOR_SRC=/path/to/workspace/harbor/src`，
任务 agent 为 `slime_plugins.harbor.agent:SlimeTerminus2`，
其基类实际加载文件是：

```text
/path/to/workspace/harbor/src/harbor/agents/terminus_2/terminus_2.py
```

本轮在共享源码与本地 Harbor 环境中验证了该继承/导入关系；没有进入正在运行的 Ray worker
替换模块，也没有声称所有节点已热更新。**已启动进程可能仍持有旧类定义，须重建 worker 后使用。**

另一个精确性修正：adapter 中的此异常是普通 `ValueError`，
而 `_run_one_trial` 的对应 catch 只捕获 `RolloutConversionError`。
因此不是把 `harbor_fail_on_conversion_error` 改为 false 就能降级。
外层 trial-error 策略还可能决定失败统计/重试，但没有把 TB 静默改成原生 GAE 的通路。

**本轮完成的是缺失生产者和远程评测隔离，不是完整截断 bootstrap。**
自然结束的主 agent、单 chunk 可以生成正确 masks 并运行 S3/S4；
真实截断仍明确报错，不能保证当前生产课程的每个 rollout 都可完成。

### 5.2 实际代码修改

| 文件 | 修改 |
|---|---|
| `../harbor/src/harbor/agents/terminus_2/terminus_2.py` | 新增版本标记 `TERMINATION_METADATA_VERSION=1`、keyword-only `record_termination_metadata=False`、真实退出事件记录；外层 run 在 finally 中输出原因 |
| `slime_plugins/harbor/tb_gae_termination.py`（新） | 校验实际 agent 类型与实际加载的 Harbor 协议/构造函数；只在 TB 训练时注入生产者开关，旧包在 Trial.create 前失败并显示源码路径 |
| `slime_plugins/harbor/rollout.py` | `_set_default_agent_config` True 且非 eval 才启用生产者；eval args 显式关 TB；远程 eval 传递 False 覆盖 |
| `slime_plugins/harbor/trial_worker.py` | 允许并严格校验 eval 的 `enable_tb_gae=False` 覆盖，使用 args 副本，不污染训练配置 |
| `slime_plugins/harbor/tb_gae_data.py` | 拒绝非终止的语义不变；错误信息增加实际 kind/reason 和生产者缺失的修复提示 |
| `tests/test_tb_gae_termination_smoke.py`（新） | 真实 run/Chat/parser、Factory/Trial 执行、截断/取消、旧版失败、off 对照、远程 eval 全链分派测试 |
| `tests/fixtures/tb_gae_termination_pre_patch/` | 本轮修改前源码与哈希；用于真实旧 agent 对照 |
| `tests/artifacts/tb_gae_termination_20260929/` | 三份 JUnit、JSON 结果、源码哈希、可审阅的跨仓库 `implementation.patch` |

`../harbor` 不在 slime 的 Git 工作树中，因此**只交付 slime diff 会漏掉最重要的生产者补丁**。
本轮实际修改的是上表共享 Harbor 文件；跨仓库 patch 同时收录两个目录的改动，
用于审阅/迁移到其它部署路径，当前工作区无需重复应用。

#### 事件语义

| 真实执行分支 | kind | reason | 是否可进入当前 TB 训练 |
|---|---|---|---|
| Terminus-2 第二次确认 task_complete，进入原有自然 return | terminated | task_complete_confirmed | 是，仍需单 chunk |
| 用完循环次数，未进入确认完成 return | truncated | max_turns | 否，缺后继 bootstrap |
| 会话提前消失 | truncated | session_ended | 否 |
| asyncio 取消 | truncated | cancelled | 否 |
| TimeoutError | truncated | timeout | 否 |
| 未恢复的 context/output 长度异常 | truncated | context_length / output_length | 否 |
| 其它异常 | error | agent_exception | 否，原异常继续抛出 |

Terminus-2 的完成语义来自已有“两次确认”的控制流程，不是 verifier 成功判断：
**agent 自然结束而 verifier reward=0 仍是 terminated**，不会只让成功轨迹通过。
恰好在最后允许轮次完成第二次确认时，terminated 优先；
仅第一次宣称完成即达到 max_turns 时仍为 truncated。
普通工具调用的 stop/EOS 不生成 terminated。

关键实现与实际代码一致：

```python
# Terminus2._run_agent_loop：原有自然完成 return 分支
if is_task_complete:
    if was_pending_completion:
        self._record_termination("terminated", "task_complete_confirmed")
        return

# for episode in range(self._max_episodes) 正常耗尽后
self._record_termination("truncated", "max_turns")
```

只有 `record_termination_metadata is True` 时 `_record_termination` 才创建状态字段。
旧 run 主体保存在 `_run_without_termination_tracking`，关闭时直接执行原主体，
不输出 termination metadata。开启时每次 run 开始先重置状态，包住原主体的异常/取消，
在原 metrics finally **之后**写 metadata，避免被原 `context.metadata = {...}` 覆盖。
异常包括初始 terminal I/O 和 skills 获取阶段，不仅覆盖 agent loop。
测试验证重复使用 agent 实例不会把上次 terminated 泄漏到本次截断。

实际 Harbor `Trial._execute_agent` 用 `asyncio.wait_for`：
超时先向 agent 注入 CancelledError，因此 agent 记录 `truncated/cancelled`；
Trial 随后抛 AgentTimeoutError。二者不冲突，也不会误记为真实终止。

#### 门控与版本校验

无需用户另行加 agent kwargs。正常 TB 开关解析后：

```python
# rollout.py::_set_default_agent_config
if getattr(args, "enable_tb_gae", False) is True and not _is_eval_args(args):
    from slime_plugins.harbor.tb_gae_termination import configure_termination_producer
    configure_termination_producer(agent)
```

helper 要求实际 `SlimeTerminus2` 的基类声明 v1，
并且构造函数显式接受 `record_termination_metadata`；
验证成功才设置 `agent["kwargs"]["record_termination_metadata"] = True`。
禁止 `agent.name` 覆盖 import_path 或冲突的 false；不会让旧 Harbor 的 `**kwargs`
悄悄吞掉一个它不认识的参数。版本检查发生在 trial config 构造期间，
**早于 Trial.create 和环境分配**，但不是模型训练 driver argparse 阶段。

TB 关闭/缺省不调用该 helper。远程 eval 额外发送 `enable_tb_gae=False`，
避免已有 TrialWorker 持有训练 args 导致 eval 误走 TB 转换；
SC-only 评测覆盖格式保持兼容。

### 5.3 本轮测试结果（不混用 9 月 27 日结果）

| 套件 | 结果 | 本次命令墙钟 |
|---|---:|---:|
| 新增真实终止 smoke | 36/36 | 与下两项一起 4.40 秒 |
| 既有 Harbor adapter | 18/18 | 同上 |
| 既有 sample postprocess | 2/2 | 同上 |
| TB 数学测试的无 Megatron 部分 | 16/16，另 11 项未选择 | 0.20 秒 |
| Harbor 原有 Terminus-2 / tmux 单测 | 20/20 | 3.94 秒 |
| **本轮合计执行** | **92 通过，0 失败，0 跳过** | 三条命令如上 |

新增 smoke 的重要覆盖：

- 真实 `SlimeTerminus2.run → _run_agent_loop → Chat.chat → JSON parser` 生成 AgentContext，
  经过实际 JSON 序列化后进入真实 adapter、hard cap、`compute_tb_gae`；
  S3/S4 与 reward=0/0.7 均通过，未向 metadata 手工塞 terminated。
- 真实 AgentFactory 根据生成的 AgentConfig 创建 agent，
  执行真实 `Trial._execute_agent`，覆盖正常完成和 wait_for 超时。
  Trial 的环境对象、外部模型/终端 I/O 是替身，不创建 Daytona 容器。
- max_turns 未完成、只确认一次、撤销完成、会话丢失、
  setup/loop 阶段异常与取消、真实 wait_for 取消、实例复用。
- 缺省/False 与冻结旧 Terminus-2 的实际 AgentContext 和 Sample 字典相同，
  覆盖自然结束和轮次耗尽；没有新增 termination 实例状态/metadata。
- 真实配置函数、eval coordinator、远程分派和 TrialWorker 方法的 AST 提取执行：
  验证生产者开关/旧版失败/评测覆盖；仅替换 Ray 传输和日志。
  这是本地函数执行，**不等价于真实 Ray 集群验收**。

本轮环境是 Harbor 的 Python **3.13.13**、PyTorch **2.10.0+cu128**、pytest **9.0.2**；
默认系统 Python 缺少 Harbor 依赖，该 Harbor venv 缺 Ray/Megatron。
因此本轮**未重新运行** 9 月 27 日完整 201 项、CPU CP=4、GPU/NCCL 或生产在线训练；
算法/CP/GAE loss 文件未在本轮修改。不能把本轮 92 项称为完整训练已通过。

Ruff 对修改文件的 check/format 通过；Harbor 全项目
`uv run --no-sync ty check --extra-search-path packages/rewardkit/src` 通过。
不带 extra-search-path 的默认 ty 命令有 rewardkit 导入路径诊断，补路径后通过；
未为此更改业务包或环境。为保护既有代码，仅格式化本次文件，未运行会重写全仓库的格式化命令。
311 个既有 shell/YAML 哈希不变；未修改 SGLang 或既有训练目录。

### 5.4 如何验证部署并启用

1. **同时部署 Harbor 与 Slime 侧补丁。** 当前共享路径已修改；
   若另一个 job 用其它 `HARBOR_SRC`、容器内安装包或已加载旧模块的 worker，
   必须把同一生产者补丁部署到实际路径，并重建该 job 的 worker。
   不需要重启/修改 SGLang 引擎源码，本次也没有操作运行中的 job。

2. 在**将用于训练的同一解释器与 PYTHONPATH**下运行无模型、无容器的预检，
   不要只 grep 某个未被使用的 Harbor 目录：

```bash
PYTHONDONTWRITEBYTECODE=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
python -B - <<'PY'
import inspect
from slime_plugins.harbor.agent import SlimeTerminus2
from slime_plugins.harbor.tb_gae_termination import configure_termination_producer

agent = {
    "import_path": "slime_plugins.harbor.agent:SlimeTerminus2",
    "kwargs": {"collect_rollout_details": True},
}
configure_termination_producer(agent)
print("loaded loop:", inspect.getfile(SlimeTerminus2._run_agent_loop))
print("protocol:", SlimeTerminus2.TERMINATION_METADATA_VERSION)
print("tracking:", agent["kwargs"]["record_termination_metadata"])
PY
```

预期输出协议 1、tracking True，并显示实际 patched Harbor 文件。
**此预检只证明生产者支持，不证明所有样本都是自然终止。**

3. 可复现本次 smoke（绝对路径对应本工作区；仅外部 I/O 替身）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
LITELLM_LOCAL_MODEL_COST_MAP=True \
PYTHONPATH=/path/to/workspace/harbor/src:/path/to/workspace/slime-terminal-RL \
/path/to/workspace/harbor/.venv/bin/python -B -m pytest \
  tests/test_tb_gae_termination_smoke.py \
  tests/test_harbor_atif_adapter.py tests/test_harbor_sample_postprocess.py \
  -o addopts='' -p no:cacheprovider -q --tb=short
```

4. 第 4 节 S3/S4 参数保持不变，`record_termination_metadata` 自动联动，
   **不要手填 `termination_kind: terminated`**。
   若原始 agent 自然终止、单 chunk，已消除本次报告定位的缺字段阻断；
   若实际数据出现 `kind='truncated', reason='max_turns'`，那是新的正确诊断，
   不是生产者又失效。eval 已单独关闭 TB，不受该训练限制。

### 5.5 尚未解决的生产全量条件及正确后续方案

当前 fresh_base 课程包含 max_turns、时间墙、上下文压缩；
**仅补生产者不会自动让这种混合数据整轮可训练**。此前 adapter 虽然抛错，
但外层通用容错会把它变成丢弃样本；第 6 节已补上异常传播防护。
没有把截断视为真实 done，也没有静默回退原生算法。

要支持这类数据，后续应另行完成并验收：

- **外部截断后继状态 value**：保存真实后继 observation/state 及对应 tokenizer 上下文，
  用同一份 old critic 前向得到 `bootstrap_values`，贯穿 adapter/DP/CP 到 TB 入口。
  当前数学核心接受显式 bootstrap/continuation，但公共入口与数据协议尚未实现这些值的生产。
  不能拿最后 action 的旧 value 冒充它之后状态的 value，不能默认 0。
- **跨 chunk continuation**：保留 episode/chunk 顺序，跨 DP 交换边界信息，
  从后向前传递下一 chunk 的 value/advantage；terminal outcome 在整 episode 只放一次。
- 同时制定 timeout、任务预算与有限时域目标的清晰语义；
  如果未来把 max_turns 定义成任务本身的 terminal horizon，属于目标函数变更，
  不能作为这次 bugfix 悄悄带入。
- 用真实截断/多 chunk replay、CP=4 和 actor/critic 一致性做独立验收后，
  才把“支持自然终止子集”升级为“支持当前生产全量”。

本轮交付的直接结论：**报告中的无生产者死结已修复并有真实 agent smoke 证据；
全量截断/多 chunk 训练仍有明确、未被掩盖的工程边界。**

---

## 附：二次复核结论（2026-09-29，审阅者，针对"现在能否启动生产训练"）

> 下面的异常吞掉问题准确描述第 6 节修复前的行为。第 6 节现已改成显式失败，
> 生产全量不支持截断/多 chunk 的结论不变。

对照实际源码逐条复核本轮修复，结论：**生产者死结确已修复，但当前 mixed3000 生产训练仍不能启用。**

**已验证为真（生产者修复是实的，不是纸面）：**
- `harbor/.../terminus_2.py:73` 有 `TERMINATION_METADATA_VERSION=1`；`:177` 构造函数接受
  `record_termination_metadata`；`:1582-1646` 的 `_record_termination` + `run()` 外层 finally
  在原 `context.metadata = {...}`（`:1724`，位于 `_run_without_termination_tracking` 的 finally）
  **之后**合并写入 `termination_kind`，顺序正确，不会被覆盖。
- `slime_plugins/harbor/tb_gae_termination.py` 存在；`rollout.py:1538-1540` 在 TB 开启且非 eval 时
  调用 `configure_termination_producer(agent)` 注入开关。链路完整。

**仍然致命（针对本工作负载）——截断轨迹会被成批丢弃：**
- 事件语义（本报告 5.2 表）：只有 `task_complete` 二次确认的自然结束才记 `terminated`；
  `max_turns`/`timeout`/`cancelled`/`context_length` 一律记 `truncated`。
- `tb_gae_data.py::trial_termination_kind` 对 `!= "terminated"` 抛**普通 `ValueError`**；
  `rollout.py:1293-1308` 的 `except` 只接 `RolloutConversionError`，**接不住这个 ValueError**
  （故 `harbor_fail_on_conversion_error=false` 无效）。它上抛到 `rollout.py:816-827` 的
  通用 `except Exception`：因生产 `harbor_fail_on_trial_error: false`，该样本被 **`stats["failed"]++`
  后丢弃**（`harbor_retry_missing_samples: 0`，不重试）。
- 历史量级：yaml 注释记载旧 run #2 的 dump 20 中，**196/256（76.6%）在 max_turns=60 停止**。
  这是风险信号，不是当前 fresh_base **100→500** 课程的实测截断率；本次没有重扫该课程数据。
  修复前若丢弃此类轨迹，batch 会偏向能在预算内自然结束的轨迹，产生选择偏差；
  不能将自然结束等同于“简单任务”，也不能宣称当前必然丢掉 76.6%。

**因此：**
- 生产全量（mixed3000 / 256k / 课程 max_turns）：**暂不启动**。需先实现截断/多 chunk 的
  bootstrap（具体见第 5.5 节；本文没有第 7 节）。迁移数据域不能取代未支持尾部的语义处理，
  “绝大多数自然结束”也不保证剩余截断不会使当前 TB 运行失败。
- 自然终止子集的真实在线冒烟可作为未来验证；**本轮只跑了 CPU 离线 smoke**，
  使用真实 agent/Factory/Trial 控制流，模型回复与终端 I/O 是替身，未运行真实线上任务。
- 因上述原因，审阅者**未**提供 TB-GAE 的生产启动脚本；SC 的生产启动脚本
  `examples/harbor_terminal_rl/launch_27b_run2_warmcritic_sc.sh` 不受 TB 的 terminated 条件限制。
  SC 的完整生产可用性不属于本轮复验范围，不由这条比较单独证明。

## 6. 二次复核追加修复：TB 异常不再被丢弃样本策略吞掉（2026-09-29）

### 6.1 确认的问题与本次范围

审阅者补充的上层异常链成立：fresh_base YAML 为
`harbor_fail_on_trial_error: false`、`harbor_retry_missing_samples: 0`。
旧 `_run_one_group` 把 `trial_termination_kind` 抛出的普通 ValueError 计为 failed，
再返回空样本；coordinator 可能补采其它 group，导致只用存活轨迹构成 batch。
之前仅修 adapter 生产者而未处理这个异常层，是一次遗漏。

**本次只补异常传播和报告准确性，不假装实现了 bootstrap。**
受支持轨迹的算法、两掩码、reward、GAE、CP 均未改；TB=False 的错误容忍和补样行为保持不变。
生产课程出现截断/多 chunk 时，现在明确失败，不能带着选择偏差继续更新参数。

### 6.2 实际代码：`slime_plugins/harbor/rollout.py`

1. `_run_one_trial`：True 且非 eval 时，RolloutConversionError 直接重新抛出，
   不受 `harbor_fail_on_conversion_error=false` 影响。
2. `_run_one_group`：True 时所有 trial Exception 直接传播；空 samples 也显式报错，
   不计作可忽略的 missing、不进入重试/补采。该规则同样覆盖远程异常的包装，
   不依赖字符串匹配或远程是否保留原异常类。
3. `_generate_rollout_async`：已完成的额外 group 出错，即使 batch 已凑齐，也不能忽略；
   尾任务取消/收集时若出现实际 Exception，仍传播。
4. 部分 group 提前结束以及 coordinator 清理子任务时，检查取消期间的实际错误。
   正常 asyncio 取消保留既有取消语义，不把取消本身当转换失败；
   已有主要异常时保留主要异常，避免被清理过程的第二个错误覆盖。

核心门控（实际代码在相应 catch 内）：

```python
except Exception as exc:
    if getattr(args, "enable_tb_gae", False) is True:
        raise
    # 原有 SC 及普通 trial 容错逻辑继续保留
```

之所以不仅在 config 中强制 `harbor_fail_on_trial_error=true`，是因为后者仍不足以覆盖
converter 返回空列表、额外采样错误和取消清理期间的异常。
没有改生产 YAML，没有自动启用 TB，也没有把失败轨迹冒充 terminated。

注意：本修复不宣称消除了所有采样偏差。已有 oversampling/partial cutoff 的主动取消策略
仍按原设计执行；这里保证已观测到的 TB 错误不会被当成普通失败丢弃。
若 bootstrap 落地后对吞吐/采样偏差有进一步要求，需要单独评估这些采样策略。

### 6.3 测试与证据

新增 `tests/test_tb_gae_error_propagation.py`，共 **14 项**：

- 读取实际 fresh_base YAML 的两项容错设置，用真实 agent 的 max_turns 输出复现旧代码
  `failed=1, missing=1, samples=[]`，确认修复后原 ValueError 直接传播、不重试。
- 缺省/False 下与本轮冻结的 `_run_one_group` 对照，失败/空样本处理一致。
- converter 容错、空样本、远程异常包装、batch 已满时的完成额外 group、
  取消时的异常、原异常优先级和部分 group 清理均覆盖。

最终与真实终止 smoke、Harbor adapter/cap、异步 rollout 和 eval 计数测试一起运行：
**103/103 通过，0 失败、0 跳过，5.16 秒**。
环境仍为 Harbor Python 3.13.13 / PyTorch 2.10.0 的 CPU 环境；
使用实际 rollout/group/coordinator 函数与异步任务调度，外部服务/远程传输由测试替身替代。
本次未启动 Ray 集群、真实任务容器、GPU 训练，也未重跑历史完整 201 项。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
LITELLM_LOCAL_MODEL_COST_MAP=True \
PYTHONPATH=/path/to/workspace/harbor/src:/path/to/workspace/slime-terminal-RL \
/path/to/workspace/harbor/.venv/bin/python -B -m pytest \
  tests/test_tb_gae_error_propagation.py tests/test_tb_gae_termination_smoke.py \
  tests/test_harbor_atif_adapter.py tests/test_harbor_sample_postprocess.py \
  tests/test_harbor_rollout_async.py tests/test_harbor_eval_count_metrics.py \
  -o addopts='' -p no:cacheprovider -q --tb=short
```

结果文件：`tests/artifacts/tb_gae_error_propagation_20260929/`。
本轮冻结源码：`tests/fixtures/tb_gae_error_propagation_pre_patch/`。
311 个既有 shell/YAML 哈希仍相同；本轮未修改 Harbor 基类、SGLang 或任何既有运行目录。

### 6.4 当前决策

**mixed3000 / 256k / max_turns 课程仍不建议启用生产 TB-GAE。**
本次把“可能丢弃截断轨迹后继续”修成“显式失败”，防止错误训练继续。
真正扩大支持范围仍须完成第 5.5 节的后继状态 value 和跨 chunk continuation；
把 max_turns 改为任务 terminal horizon 是独立的目标函数选择，不作为本次 bugfix 带入。

## 7. 2026-09-29：截断 Value Bootstrap 与多 chunk 全链路落地

### 7.1 现在是否能拉起 mixed3000 TB-GAE

**原先两个代码级阻断已解除：**

- `max_turns`、可恢复后继状态的 `timeout/cancelled/context_length` 轨迹不再被
  `termination_kind != terminated` 一律拒绝。它们保留 `is_real_done=False`，
  并使用真实后继输入的 old critic value。
- 原始 `trajectory_retokenized` 的 split，以及训练 context cap 产生的多个 chunk，
  可以跨 DP 放置并连续递推 GAE，不再要求所有数据单 chunk。

真实混合 rollout 的离线测试同时输入自然完成、max_turns、timeout 三条轨迹，
**三条全部进入 batch，failed=0，没有补采替换**。
与“把异常传播变响亮”不同，本轮新增了处理这些轨迹的实际能力。

**但本轮没有启动生产集群，也没有运行 27B / 256k / PP=2 的 GPU smoke。**
现在可以进入受控 S3 GPU smoke 阶段；是否能够按生产吞吐和显存限制持续训练，
仍须该阶段验证，不能由 CPU 测试保证。不得把历史 76.6% 当作当前课程的实测比例。

支持域仍限定为主 agent 单 detail、THD ordinary zigzag CP=1/4、PPO、KL reward=0，
SC/OPD/路由 replay 关闭。真实多 step/subagent、BSHD 及另一类 CP 布局尚未开放。
有效后继状态无法取得、数据损坏或 episode chunks 不完整时仍报错；
本轮解决“正常截断不被误拒绝”，不把所有基础设施故障当成有效数据。

### 7.2 实现文件和协议

| 文件 | 本轮修改 |
|---|---|
| `../harbor/src/harbor/agents/terminus_2/terminus_2.py` | 终止协议升级 v2；只在开启记录时保存下一轮 prompt/消息历史；截断后采集后继输入及已完成 action-token 数 |
| `slime_plugins/harbor/llm.py` | `encode_bootstrap_state(prompt, message_history)`：复用实际 `_encode_messages`，不请求 SGLang、不生成 token |
| `slime_plugins/harbor/tb_gae_termination.py` | 预检要求实际加载 Harbor v2，旧 worker/旧包显式失败 |
| `slime_plugins/harbor/atif_adapter.py` | 接受带有效后继状态的 truncated；按原始 split 逐 chunk 标 masks；附加 episode/chunk 边界；截断 Sample 状态为 TRUNCATED |
| `slime_plugins/harbor/tb_gae_data.py` | 验证终止/后继状态、完整 chunk 集；True 时收集两 masks 与 `tb_boundary` |
| `slime_plugins/harbor/tb_gae_chunks.py`（新） | 模型长度 cap 下保留全部 action，分段训练；不再在 TB 路径丢掉 response 前缀 |
| `slime_plugins/harbor/sample_postprocess.py` | TB 开启时调用分段逻辑；关闭保持原 hard trim |
| `slime/ray/rollout.py` | True 时沿相同 DP partition 透传 `tb_boundary`；原调度不变 |
| `slime/backends/megatron_utils/tb_gae_bootstrap.py`（新） | 后继状态 probe 构造、DP/PP/CP 对齐的额外 old-critic forward、CP gather 提取标量 |
| `slime/backends/megatron_utils/actor.py` | critic 在 optimizer 更新前算 bootstrap；随 old values 回传 actor；actor 使用同一份 bootstrap |
| `slime_plugins/harbor/tb_gae_boundary.py`（新） | 跨 DP gather 小型边界摘要，按 episode/chunk 顺序反向求 continuation |
| `slime_plugins/harbor/tb_gae.py` | 完整 episode 分派；outcome 只落最后 chunk；带 bootstrap/continuation 递推，原生 CP slice 返回 detached targets |
| `slime/backends/megatron_utils/data.py` | 日志跳过 boundary 与 bootstrap 非标量字段 |
| `slime/utils/tb_gae_config.py` | 启用期明确拒绝 routing replay，避免未实现的分段 routing 误用 |
| `tests/test_tb_gae_bootstrap*.py`、`tests/tb_gae_bootstrap_distributed_worker.py` | 新增 producer/critic/数学/跨 DP×CP 验证；更新旧“截断必须拒绝”测试为新协议验收 |

默认 `enable_tb_gae=False` 不创建 masks、boundary、bootstrap 字段，不执行上述额外前向或通信。
`Sample` dataclass、原生 `ppo_utils.py/cp_utils.py/dp_schedule.py` 未修改。

新增数据形态（只在 True 时）：

```python
# Harbor AgentContext.metadata：只在 truncated 时需要后继状态
{
    "termination_kind": "truncated",
    "termination_reason": "max_turns",  # 或 cancelled/timeout/context_length 等
    "termination_metadata_version": 2,
    "bootstrap_token_ids": [...],      # 下一次 action 前的真实 chat-template 输入
    "bootstrap_action_count": 1234,    # 必须等于已收集主 agent action token 数
}

# 每个 Sample.metadata / train_data["tb_boundary"][i]
{
    "episode": "...",                  # 与原 group 身份一致
    "chunk_index": 0,
    "chunk_count": 3,
    "terminal": False,                 # 仅最终真实终止 chunk 为 True
    "bootstrap_token_ids": None,       # 只在最终截断 chunk 携带完整后继输入
}
```

chunk 集在 DP 切分前要求完整、不重复。训练侧再次校验 chunk count、
顺序、terminal 位置和同 episode 的 reward 一致性。
如果旧 dump 缺少新字段，不能把缺失数据猜成零 bootstrap；需要用新版 producer 重新采集
或从可验证的原始状态恢复。不要在线重用不满足 v2 的旧缓存。

### 7.3 后继状态与 old critic 的精确含义

#### 真实状态采集

- 正常 max_turns：保存最后一轮 action 执行后的 observation，以及下一次模型调用前的消息历史。
- 等待下一次模型回复时 timeout：该 action 尚未生成，保存请求使用的后继输入，
  不把未完成生成算作训练 action。
- 工具执行期间 timeout/cancel：在异常清理中读取终端增量输出（最多等待 2 秒），
  作为真实的部分执行 observation；不再执行新工具命令。
- context unwind/summary 改变历史时，保存实际请求使用的消息历史副本，
  不把旧 prompt 与后来的历史改写错误组合。
- 若环境/终端已不可读且没有可靠后继状态，写 `bootstrap_error`，
  adapter 明确拒绝这条无有效状态的数据；不会退回 V(last action) 或常量零。

`encode_bootstrap_state` 使用与 rollout `call()` 一致的 tokenizer/chat template，
含 generation prompt，且**不发生网络推理请求**。
`bootstrap_action_count` 与 adapter 的完整 action 数对齐，用于拒绝错接尾部。

#### Value 前向与 shift

critic 在同一次 rollout 的 `train_critic` 中按以下顺序执行：

```text
原有 old values forward
    -> 后继状态 bootstrap forward
    -> TB-GAE / returns
    -> critic optimizer update
    -> 返回 old values + tb_bootstrap_values 给 actor
```

把后继状态前缀 `P` 构造成 `P + [dummy]`，
`response_length=1`，复用现有 `get_values` 的 shifted-logit 约定：
它取的是 **P 最后一个 token 的 hidden/value**，即下一 action 开始前的状态值。
dummy 不产生训练 loss、不更新模型，也不是额外 rollout action。
测试中的因果 toy critic 明确区分这个值和最后一个 action 的 value。

所有 value 前向和 target 计算均在 no_grad 下；actor 接收到的是**更新前** critic 的标量列表，
不会用更新后权重重新估计。测试记录调用顺序并在 optimizer 后改变 toy 权重，
确认 actor 收到的仍是旧值。

#### 上下文和性能约束

当前固定有限上下文策略：

- 正常完整 sample 若未超 cap，原 tokens 不变。
- 超 cap 的 response 以约 cap/2 的跨度分段，每段保留最多 cap 个总 token 的左侧上下文；
  所有 action token 保留、顺序不变、不会重复训练历史 action。
  纯 observation 段不独立成为训练样本，它们仍可以作为后续段的 context。
- 后继输入超过 `seq_length-1` 时，bootstrap probe 取最近 `seq_length-1` 个 prefix token，
  末尾加 dummy。**这是有限窗口状态价值，不是无限历史的精确 V。**
  若任务依赖窗口外信息，需要保证 rollout 的压缩/摘要策略保留它；本实现不能证明信息无损。

额外 forward 使用每 probe 一个 microbatch，避免以原 response 长度打包造成长后继状态 OOM。
probe 数通过全训练组 Gloo MAX 对齐并补至 VPP microbatch group 的倍数；
没有真实 bootstrap 的样本用短 dummy probe 保持调度一致，避免各 DP rank 独立跳过造成 collective 错位。
代价是 **开启时增加 critic forward 与一轮小型控制通信**，
可能显著增加 256k 训练时间；本轮没有给出 GPU 吞吐/峰值显存保证。
False 路径没有这些开销。

### 7.4 跨 chunk / 跨 DP continuation 数学与奖励

每个 chunk 按 action-token 顺序递推。它与下一 chunk 的连接写成：

```text
A_first(chunk) = b + c_v * V_first(next_chunk) + c_a * A_first(next_chunk)
```

其中 `b` 是以边界 value/advantage=0 计算的首优势；
`c_v`、`c_a` 来自同一递推的 γ/λ 乘积。
同 turn 内切分的 chunk 边界不额外折扣；真实 turn_end 才使用轮次折扣。

每个 CP rank 先用原生 helper 恢复自己的完整 chunk values，
仅把 episode ID、chunk index/count、首 value、b/c_v/c_a、最终 bootstrap 等**小型摘要**
在对应 DP Gloo group 上 all-gather。
排序后从最后 chunk 反推，得到每个 chunk 的 next value / next advantage，
最后本地重新递推、用原生 `slice_log_prob_with_cp` 返回局部 targets。
不要求同一 episode 的所有 chunk 在同 DP rank，也不改变原 DP microbatch 调度。

尾边界：

- 真实 terminated：`V_next=0, A_next=0`。
- truncated：`V_next=V_old(successor), A_next=0`；
  后者是有限 rollout 的 GAE 尾端，不等于把前者也设成零。

标量 outcome 在 **整个 episode 最后一个有效 action** 放置一次。
仍沿用现有 reward postprocess 输出作为截断点已观察 reward；不会把每个 chunk 的同一 trial reward 重复加总。
使用这一协议后，truncated S3 不应再与旧“每个 chunk 独立、尾部 V=0”的原生结果相等；
等价测试必须使用相同边界和奖励语义的整 episode 参考。
对旧 complete-terminal S3 的 action 位负对照仍然保留。

### 7.5 离线验收结果

本轮最终执行 **338 项，全部通过，0 失败、0 跳过**：

| 测试组 | 用例 | 墙钟 |
|---|---:|---:|
| 训练侧：bootstrap/数学/默认关闭零回归/配置/DP×CP/旧 CP/DP 调度/角色/SC 回归 | 207 | 321.38 秒 |
| Harbor 侧：真实后继状态、真实终止、异常传播、adapter/cap/异步 rollout/eval | 111 | 5.42 秒 |
| Harbor 原有 Terminus-2 / tmux 单元测试 | 20 | 4.05 秒 |

重点证据：

- 同一真实控制流程的 mixed rollout：自然完成/max_turns/timeout 全部保留，failed=0，无替代补采。
- 真实 agent + Chat/parser 产生后继字段，正常 max_turns、等待模型的 timeout、
  工具运行中 timeout、真实 `Trial._execute_agent` 的 AgentTimeoutError 均覆盖。
- 实际 `make_bootstrap_batch → forward_only → get_batch → get_values → CP gather` 路径：
  CPU toy critic、替代 pipeline scheduler；只在 CPU 测试将生产 metadata 的 `.cuda()` 迁移变为 no-op。
- critic 更新前 bootstrap、actor 接收同一份旧值；S3/S4 截断/多 chunk 对整 episode 参考的比较。
- 真正 **Gloo DP=2×CP=1/4** 多进程：chunk 跨 DP、局部顺序反转、各 DP 样本数不同、
  空 CP response、非整除长度；bootstrap probe count 对齐、原生 CP slice、continuation 均通过。
- 原有 False/缺省路径仍对冻结旧源码比较 Sample、DP payload、targets、loss、所有 logits 梯度、
  AdamW 更新；相关测试通过。
- 311 个正式 shell/YAML 哈希保持不变；`Sample/ppo_utils/cp_utils/dp_schedule` 未修改；
  SGLang 源码、既有运行目录未写入。
- 新增文件 Ruff/Black 检查通过；Harbor 修改文件及带 rewardkit 路径的 ty 检查通过。
  原 `llm.py` 的 B004、`sample_postprocess.py` 的 UP035 是已有 lint 问题，未顺便改无关逻辑。

原始结果和代码哈希：`tests/artifacts/tb_gae_bootstrap_20260929/results.json`，
同目录含三份 JUnit 和本轮跨仓库 implementation.patch。
该 patch 与前轮 patch 分层；部署完整功能需包含此前开关/终止生产者补丁及本轮改动，
或直接使用当前工作区对应文件。

**没有执行真实 PP=2 pipeline、27B GPU forward、NCCL、多节点在线 rollout 或生产 optimizer step。**
CPU mock scheduler 验证了实际调用/packing/callback，不是实际 Megatron pipeline 调度器的全规模验收。
不要把 338 项离线通过写成“生产已跑通”。

复现训练侧测试（当前系统 Python 有 Megatron/PyTorch）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/tb_bootstrap_inductor \
python -B -m pytest \
  tests/test_tb_gae_bootstrap.py tests/test_tb_gae_math.py \
  tests/test_tb_gae_zero_regression.py tests/test_tb_gae_config.py \
  tests/test_tb_gae_bootstrap_distributed.py tests/test_tb_gae_cp_smoke.py \
  tests/test_dp_schedule.py tests/utils/test_megatron_role_config.py \
  tests/test_ppo_sc_args.py tests/test_ppo_sc_pipeline.py \
  -o addopts='' -p no:cacheprovider -q --tb=short
```

Harbor 测试需使用 Harbor venv 与两份源码路径（与第 5 节相同），测试列表：

```text
tests/test_tb_gae_bootstrap_agent.py
tests/test_tb_gae_termination_smoke.py
tests/test_tb_gae_error_propagation.py
tests/test_harbor_atif_adapter.py
tests/test_harbor_sample_postprocess.py
tests/test_harbor_rollout_async.py
tests/test_harbor_eval_count_metrics.py
```

### 7.6 如何真正拉起：部署与启动参数

**无需修改正式启动脚本。** 先让训练 driver、所有 Harbor workers 和 actor/critic workers
加载同一版本的两套源码；现有运行进程不会因磁盘改动自动替换已 import 的类。
旧 v1 / 缺 encoder 的数据 producer 不具备新能力。

在同一训练解释器/PYTHONPATH 下做静态路径预检：

```python
import inspect
from slime_plugins.harbor.agent import SlimeTerminus2
from slime_plugins.harbor.llm import SlimeSGLangLLM
from slime_plugins.harbor.tb_gae_termination import configure_termination_producer

config = {
    "import_path": "slime_plugins.harbor.agent:SlimeTerminus2",
    "kwargs": {"collect_rollout_details": True},
}
configure_termination_producer(config)
assert SlimeTerminus2.TERMINATION_METADATA_VERSION == 2
assert hasattr(SlimeSGLangLLM, "encode_bootstrap_state")
print(inspect.getfile(SlimeTerminus2))
print(config["kwargs"]["record_termination_metadata"])  # True
```

S3 的公共 YAML（写在独立任务/角色配置副本顶层，不能写单角色 overrides）：

```yaml
enable_tb_gae: true
tb_gae_gamma_turn: 1.0
tb_gae_lambda_turn: 1.0
```

S4 只改两个轮次参数：

```yaml
enable_tb_gae: true
tb_gae_gamma_turn: 0.999
tb_gae_lambda_turn: 0.999
```

保持 PPO、THD、ordinary CP=4、SC=False、OPD=False、routing replay=False；
新 protocol 与 bootstrap 由公共开关自动接通，不手工写 `termination_kind` 或 `bootstrap_values`。

建议先在独立运行目录做 S3 GPU smoke。以下仅是**未执行的命令示例**：

```bash
SMOKE=1 NUM_ROLLOUT=1 NUM_CRITIC_ONLY_STEPS=0 \
RUN_TAG=TBGAE_S3_GPU_SMOKE_20260929 \
EXTRA_MISC_ARGS="--no-load-optim --no-load-rng --enable-tb-gae --tb-gae-gamma-turn 1.0 --tb-gae-lambda-turn 1.0" \
bash examples/harbor_terminal_rl/launch_27b_fresh_base_critic_warmup2.sh
```

必须显式 `NUM_ROLLOUT=1`，因为顶层 fresh_base 脚本已给 100，单设 SMOKE 不一定覆盖；
smoke 用 `NUM_CRITIC_ONLY_STEPS=0` 让 actor 同步覆盖，生产 warmup 仍沿用原设置。
此生产启动链会分配真实集群、创建模型/环境并可能触发保存，它不是 CPU 单测命令。
本轮没有执行它；在独立资源与新目录准备好后再运行。
Smoke 数据中应包含 max_turns/timeout 和 split，不能只选自然完成轨迹证明 bootstrap。

实际生产 S3 参数仍可通过原入口：

```bash
EXTRA_MISC_ARGS="--no-load-optim --no-load-rng --enable-tb-gae --tb-gae-gamma-turn 1.0 --tb-gae-lambda-turn 1.0" \
bash examples/harbor_terminal_rl/launch_27b_fresh_base_critic_warmup2.sh
```

通过 GPU smoke 后再扩大 batch；S4 将两个 1.0 改为 0.999，其余条件固定。
回滚用 `--no-enable-tb-gae`，保留原生 GAE，不删除 bootstrap 模块或热切在途数据。

上线观察应至少包括：truncated 保留数量、episode/chunk 完整性、
实际后继 value 分布、critic/actor targets 一致性、probe forward 时间与显存峰值。
若出现 `bootstrap_error`，应排查实际状态是否可恢复，而不是填零或将 truncated 标成 terminated。
