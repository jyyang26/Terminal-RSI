# PPO-SC 实现说明与开启指南

**整理日期：2026-09-29。** 适用仓库：`/path/to/workspace/slime-terminal-RL`。

本文回答两个问题：**SC 是怎样接进现有 PPO/Harbor 训练链路的，以及怎样正确开启、关闭和确认它生效。** 本次只写本文档，没有修改业务源码、配置、依赖或启动训练。

当前代码已经实现 SC 的采样、数据、损失和分布式计算链路。2026-09-26 曾完成 302 项离线测试及单节点 8 GPU、1-step 端到端冒烟。**这些是 9 月 26 日的验收结果；9 月 29 日工作区已有后续 TB-GAE 改动，不能把历史结果当作当前整个工作区又完成了一次验收。**

最重要的使用结论：

1. 当前已通过真实 GPU 冒烟的是 `sc_formula: score`、`sc_wire_format: json`、`sc_top_k: 32`、`sc_correction_scale: 1.0`。
2. **SC 默认关闭。开启时必须显式选 `json`，除非已经单独验证 SGLang 的 binary head 输出能力。** 代码的 wire 默认值是 `binary`，只加总开关并不适合未打补丁的引擎。
3. 角色 YAML 的 SC 参数放在**根层 `score_centering:` 块**，与 `megatron:` 同级，不能只在 Actor 的 `overrides` 中开启。
4. 现成 `megatron_roles_27b_criticlr30x_warmcritic_sc_shadow.yaml` 的 **scale=0**，只是诊断模式，**不会给 PPO loss 加上 SC 梯度**。
5. **当前 TB-GAE 与 SC 不能同时启用。** `slime/utils/tb_gae_config.py::validate_tb_gae_config` 明确要求启用 TB-GAE 时 `enable_score_centering=False`。只测试 SC 时应确保 `enable_tb_gae: false`，或通过 `--no-enable-tb-gae` 显式关闭 TB-GAE。

## 1. 快速开启与关闭

### 1.1 推荐方式：复制 warmcritic 角色配置，追加公共 SC 块

保留原文件：

[`examples/harbor_terminal_rl/megatron_roles_27b_criticlr30x_warmcritic.yaml`](examples/harbor_terminal_rl/megatron_roles_27b_criticlr30x_warmcritic.yaml)

在派生文件中使用下列**完整角色配置**。它与原文件的有效 Actor/Critic 参数一致，仅添加 SC 和显式关闭 TB-GAE：

```yaml
# 完整派生角色配置：真实 SC 加项
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

# 与 megatron 同级；TB-GAE 当前不支持与 SC 同开。
enable_tb_gae: false

# 此块先作用于 rollout/actor 的公共参数，再生成角色副本。
score_centering:
  enable_score_centering: true
  sc_top_k: 32
  sc_formula: score
  sc_wire_format: json
  sc_correction_scale: 1.0
  sc_tail_eps: 1.0e-8
  sc_weight_denominator: old_actor
  sc_assert_alignment: full
  sc_chunk_size: 256
  sc_max_response_bytes: 268435456
  sc_max_trial_bytes: 536870912
  sc_max_batch_bytes: 8589934592
```

`sc_weight_denominator` 只在 `ppo_weighted` 模式参与 SC 系数计算；这里保留 `old_actor` 是为了与现有 PPO 分母约定一致，**不会使 score 模式额外收集 old-policy head**。

下面命令可生成派生配置，**只创建新文件，不覆盖原角色文件**。命令放在文档中供之后执行；本次没有执行它：

```bash
cd /path/to/workspace/slime-terminal-RL

export SC_ROLE_CONFIG="$PWD/examples/harbor_terminal_rl/megatron_roles_27b_criticlr30x_warmcritic_sc_active.yaml"

python3 - <<'PY'
import os
from pathlib import Path
import yaml

source = Path("examples/harbor_terminal_rl/megatron_roles_27b_criticlr30x_warmcritic.yaml")
target = Path(os.environ["SC_ROLE_CONFIG"])
config = yaml.safe_load(source.read_text())
config["enable_tb_gae"] = False
config["score_centering"] = {
    "enable_score_centering": True,
    "sc_top_k": 32,
    "sc_formula": "score",
    "sc_wire_format": "json",
    "sc_correction_scale": 1.0,
    "sc_tail_eps": 1e-8,
    "sc_weight_denominator": "old_actor",
    "sc_assert_alignment": "full",
    "sc_chunk_size": 256,
    "sc_max_response_bytes": 268435456,
    "sc_max_trial_bytes": 536870912,
    "sc_max_batch_bytes": 8589934592,
}
with target.open("x") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
print(target)
PY
```

然后通过已有环境变量把角色配置交给现有启动链：

```bash
# 这是现有多节点 run2 启动入口，不是单节点 8 卡启动器。
# 执行前需具备该入口要求的集群、数据池、checkpoint 和鉴权环境。
MEGATRON_CONFIG_PATH="$SC_ROLE_CONFIG" \
EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---no-load-optim --no-load-rng} --no-enable-tb-gae" \
  bash examples/harbor_terminal_rl/launch_27b_run2_warmcritic_ctrf82.sh
```

`launch_27b_run2_warmcritic_ctrf82.sh` 会继续调用原 27B wrapper 和父启动器；没有必要为了 SC 修改这些 `.sh`。这个命令**不会自动把原多节点配置缩成单节点**。

还要检查 `--custom-config-path` 指向的 Harbor YAML：如果其中也存在 SC 参数，它的值会覆盖角色文件公共块；如果其中开启了 TB-GAE，仅在角色文件设置 `enable_tb_gae: false` 未必足够，所以示例通过 CLI `--no-enable-tb-gae` 显式关闭。

### 1.2 关闭 SC

最直接的方法是继续使用原角色文件，且不传 SC enable CLI、不在 Harbor YAML 中开启 SC。需要显式关闭时，在**派生角色文件的根层**写：

```yaml
score_centering:
  enable_score_centering: false
```

关闭是否最终生效，要看所有配置来源。高优先级的任务 YAML 仍可开启 SC；如果 CLI 显式开启而 YAML 显式关闭，当前代码会报冲突，而不是悄悄忽略其中一方。

SC 总开关是 `store_true`：

- 开启：`--use-score-centering` 或 `--enable-score-centering`。
- 关闭：不传这两个 CLI 开关，并确保 YAML 最终值为 `false`。
- **不要写** `--use-score-centering=False`、`--enable-score-centering false`，也不要假设存在 `--no-enable-score-centering`。

### 1.3 诊断模式与真实训练的区别

| 模式 | enable_score_centering | sc_correction_scale | 请求/搬运 Head | SC 加入反向 |
|---|---|---:|---|---|
| 原 PPO | false/缺省 | 不参与计算 | 否 | 否 |
| SC 诊断 | true | 0.0 | 是 | 否 |
| PPO + SC | true | 1.0 | 是 | 是 |

现有诊断配置：

[`examples/harbor_terminal_rl/megatron_roles_27b_criticlr30x_warmcritic_sc_shadow.yaml`](examples/harbor_terminal_rl/megatron_roles_27b_criticlr30x_warmcritic_sc_shadow.yaml)

文件名中的 `shadow` 表示 **scale=0 的算法诊断模式**，不代表已经创建或切换了 SGLang Shadow 环境。诊断模式仍然有采样、网络、CPU/GPU前向和指标开销，不能当作关闭 SC。

### 1.4 另外两种配置入口

**直接 CLI：** 在原训练 argv 后追加以下参数即可，前提是任务/角色 YAML 没有覆盖它们，且原训练配置满足第 6 节的要求：

```text
--use-score-centering
--score-centering-top-k 32
--sc-formula score
--sc-wire-format json
--sc-correction-scale 1.0
--no-enable-tb-gae
```

`--score-centering-top-k` 的别名是 `--sc-top-k`。`sc_top_k` 表示**每个生成 token 返回多少项概率信息**，不是限制模型采样支持集的 `--rollout-top-k`；后者在本实现要求保持 `-1`。

**Harbor 任务 YAML：** 对 `--custom-config-path` 指向的 YAML，在根层写扁平字段：

```yaml
# 这是 Harbor/custom-config 文件的根层片段，不是角色文件的嵌套公共块。
enable_score_centering: true
sc_top_k: 32
sc_formula: score
sc_wire_format: json
sc_correction_scale: 1.0
enable_tb_gae: false
```

当前 `configure_score_centering` 只从**角色 YAML**识别嵌套的 `score_centering:`，对 custom YAML 识别扁平 SC keys。**不要把角色文件的嵌套写法原样粘到 custom YAML 中并假设会生效。**

SC 配置优先级为：**parser/CLI → 角色 YAML 根层 score_centering 块 → custom YAML 扁平 SC keys**。这与当前 TB-GAE“显式 CLI 优先”的规则不同。建议每次只选择一个主要 SC 参数来源。

## 2. 我是怎样实现 SC 的

### 2.1 实施原则和完成过程

先完整阅读历史冲突/B9/B10文档，对照用户提供的上游 Slime 快照；随后把方案中的 SC-SPEC Before/After 按原工作区 SHA256 重建，逐模块落地。保护的是用户当时已定制的工作区，而不是直接 reset 到 Git HEAD 后再合并。

上游参考为 `slime_latest/slime` 的 `8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b`，包含 #2405 及 #2406；这是当时核实的本地参考提交，不代表已确认 2026-09-29 远端最新版本。

实际做法是保留本地 PPO、Harbor、Ray 和 Megatron 主流程，在明确开关后挂接 SC 数据与损失项。没有把整个上游 `loss.py`、`tensor_store.py` 或 rollout 系统直接覆盖到本地；也没有把 PPO 目标替换成上游默认的 REINFORCE-SC。

阶段一/二完成配置、算法、采样、适配器、分发、训练/损失、old-policy producer及离线测试。之后用独立源码快照和 venv，在单节点真实运行 Daytona→SGLang→Harbor→Ray→Megatron，依据运行结果调整了**冒烟作业自己的拓扑和环境变量**。

### 2.2 实际数据链路

```text
公共 args：enable_score_centering / sc_top_k / sc_formula / sc_wire_format
  → Harbor _set_default_agent_config
  → SlimeSGLangLLM.call
  → SGLang /generate：top_logprobs_num=k
  → meta_info.output_top_logprobs（json 模式）
  → 解码、合法性验证、绑定 prompt/token/head digest
  → LLMResponse.extra.score_centering_head
  → Harbor Chat 汇总到每轮 rollout_details
  → atif_adapter 构造完整 response 对齐 Head
  → sample_postprocess 同步真实 Token 裁切
  → Sample 的三个 SC-only 动态字段
  → RolloutManager 校验、按同一 DP partition 放入 Ray
  → Actor PP-last 读取完整 CPU Head
  → model.get_batch 按 microbatch 提取
  → sc_terms 映射 TP/CP/logits 行
  → Harbor/core PPO loss 加入 SC addend
  → 原 loss_mask/group_mask_sums 归约
  → backward / optimizer.step
```

### 2.3 模块与职责

| 相对路径 | 关键入口 | 具体作用 |
|---|---|---|
| `slime/utils/arguments.py` | `get_slime_extra_args_provider`、`slime_validate_args`、`parse_megatron_role_args` | 定义 CLI；YAML合并后解析SC；角色覆盖校验 |
| `slime/utils/score_centering_config.py` | `SC_DEFAULTS`、`configure_score_centering`、`isolate_score_centering_role` | 单一enable状态；别名归一化；公共块解析；Critic关闭 |
| `slime/utils/score_centering_validate.py` | `validate_score_centering_args` | 仅SC-on校验支持范围和数值参数 |
| `slime/utils/score_centering.py` | `score_centering_correction`、`calculate_head_log_probs`、codec helpers | 冻结系数标量化；完整词表TP概率；Head编码/检查 |
| `slime/utils/pg_objectives.py` | `apply_score_centering` | Harbor/core共享挂载；scale0不执行loss加法 |
| `slime_plugins/harbor/llm.py` | `SlimeSGLangLLM.__init__ / call` | 条件请求Top-k、每轮解码、预算、JSON-safe extra |
| `slime_plugins/harbor/rollout.py` | `_set_default_agent_config`、`_run_one_trial` | 从公共args注入LLM/转换器，禁止SC数据错误静默丢弃 |
| `slime_plugins/harbor/trial_worker.py` | `run_trial` | SC作业远端eval覆盖为关闭，避免持久worker误用训练args |
| `slime_plugins/harbor/atif_adapter.py` | `_normalized_trajectory_samples_from_detail`、`_attach_normalized_sc_head`、`_slice_sc_tracks` | 多轮、工具上下文、split、Token身份和同步切片 |
| `slime_plugins/harbor/sample_postprocess.py` | 两个 `cap_total_tokens_allow_response_trim_*` hook | Sample阶段真实裁切同步；SC-on拒绝训练晚期再次裁切 |
| `slime/ray/rollout.py` | `_convert_samples_to_train_data`、`_split_train_data_by_dp` | 逐样本检查；三字段使用同一分区；batch字节预算 |
| `slime/backends/megatron_utils/actor.py` | `init`、`_get_rollout_data`、`train_actor` | Critic隔离、CPU Head归属、每rollout挂载检查 |
| `slime/backends/megatron_utils/model.py` | `forward_only`、`train_one_step` 内部 `forward_step` | gated batch keys；复用old-actor forward收集head |
| `slime/backends/megatron_utils/sc_terms.py` | `response_row_layout`、`old_policy_head_callback`、`get_score_centering_terms` | full response→局部logits映射；优势与CP布局；返回已乘A的加项 |
| `slime/backends/megatron_utils/data.py` | `log_rollout_data` | 防止大矩阵被错误当作标量日志归约 |
| `slime_plugins/harbor/train_loss.py` | `policy_loss_no_entropy_when_disabled` | 两条生产基线实际使用的custom_loss挂载点 |
| `slime/backends/megatron_utils/loss.py` | `policy_loss_function` | core路径对称挂载 |

当时 SC 实施没有改写 `slime/utils/ppo_utils.py::compute_policy_loss`、`Sample` dataclass schema 或 checkpoint目录隔离函数。后续 TB-GAE 对共同优势入口的改动必须另行看待，见第 9 节。

## 3. 两种公式与梯度边界

记 p=当前训练策略，q=SGLang采样概率，o=冻结old-actor策略，A=优势。基线最小化的PPO仍是：

\[
\ell_{\mathrm{PPO},t}=\max\{-A_t r_{t,y},-A_t\operatorname{clip}(r_{t,y},1-\epsilon_l,1+\epsilon_h)\},\qquad r_{t,y}=p_t(y)/o_t(y).
\]

原基线 `keep_old_actor=True`、`use_rollout_logprobs=False` 时，o是训练侧old-actor重算，不能直接认为o=q。

### 3.1 已通过GPU冒烟的 score 模式

K为采样器返回的Top-k集合，head概率保持**全词表归一化**，不在K内重新归一化：

\[
P_K=\sum_{v\in K}p_v,\quad Q_K=\sum_{v\in K}q_v,\quad
\alpha=\frac{\max(1-Q_K,0)}{\max(1-P_K,10^{-8})}.
\]

构造冻结系数和SC加项：

\[
c_v=\operatorname{sg}(q_v-\alpha p_v),\qquad
\ell_{\mathrm{SC},t}=\operatorname{sg}(A_t)\sum_{v\in K}c_v\log p_t(v).
\]

\[
L=L_{\mathrm{PPO}}+\lambda\,\operatorname{Reduce}_{mask,group}(\ell_{\mathrm{SC}}).
\]

因此SC部分梯度为：

\[
\nabla_\theta\ell_{\mathrm{SC},t}=A_t\sum_{v\in K}(q_v-\alpha p_v)\nabla_\theta\log p_t(v).
\]

实现中，系数计算位于 `torch.no_grad()`，最后使用 `coefficient.detach()`；A也detach。梯度只通过训练侧的live logp回传，不通过q、alpha、系数中的p、old-policy head或A回传。

系数计算位于 `score_centering.py`；`sc_terms.py` 先乘A并处理CP布局，返回**已经乘优势的addend**；`pg_objectives.py` 再执行 `pg_loss + scale * addend`。不要在外层重复乘一次A。

epsilon保护应用于 `(1-P_K).clamp_min(1e-8)`；不能用FP32的`clamp(max=1-1e-8)`替代，因为`1-1e-8`可能舍入为1。q的真实零尾质量保持0。

### 3.2 ppo_weighted 模式

历史 PPO-SC 方案希望中心化PPO实际梯度权重。该模式采用：

\[
\bar w(v)=r_v\,\mathbf 1\{\neg[(A>0\land r_v>1+\epsilon_h)\lor(A<0\land r_v<1-\epsilon_l)]\}.
\]

注意：原生PPO clip的平坦区权重为0；未截断区域是原始r，**不是clamp(r)**。

令 \(p_{tail}=\max(1-P_K,\epsilon)\)、\(q_{tail}=\max(1-Q_K,0)\)、\(o_{tail}=\max(1-O_K,\epsilon)\)：

\[
\rho=q_{tail}/p_{tail},\quad r_{tail}=p_{tail}/o_{tail},\quad
c_v=\operatorname{sg}\{q_v\bar w(v)-\rho\bar w(r_{tail})p_v\}.
\]

选择方法：

```yaml
# 加在角色文件的 score_centering 块中；其余参数与前面的配置相同。
score_centering:
  enable_score_centering: true
  sc_formula: ppo_weighted
  sc_weight_denominator: old_actor
  sc_wire_format: json
  sc_top_k: 32
  sc_correction_scale: 1.0
```

当前基线必须保留 `--keep-old-actor`。old-policy head在现有old-actor前向中追加gather产生，不额外再做一次完整模型前向；按完整response收集后放CPU。若PPO显式改为`use_rollout_logprobs=True`，该模式才应配`sc_weight_denominator=sampler`；不要为省事改动原PPO分母。

**score与ppo_weighted不是同一个算法。** 前者严格对应任务指定的q−alpha·p加项；后者对应PPO有效权重中心化。weighted已在离线/CPU分布式测试中验证，本次真实GPU成功作业采用score，不能把该GPU记录当作weighted的实测结果。

### 3.3 数学与验证限制

Top-k尾部折叠假设尾部q与p成比例；weighted模式还要考虑old-policy尾部比例。极小尾质量处有epsilon正则化。Harbor retokenized历史与原采样prompt也可能不同。这些都限制了全词表等价性。

不能据此宣称任意PPO、任意优势分布或任意重分词情况下“严格消除所有漂移”。同样，代数上的零系数不等于浮点逐bit为零；正常的SC标量也可能非零而梯度为零。

## 4. Head提取、Token对齐和分布式计算

### 4.1 请求与传输

Harbor使用自己的httpx客户端，不走通用`slime/utils/http_utils.post`。SC开启后请求额外包含：

```json
{
  "return_logprob": true,
  "top_logprobs_num": 32,
  "return_text_in_logprobs": false
}
```

`return_logprob`本来就存在；SC-off不添加后两项，也不解码Top-k。构造器接收`score_centering_top_k`，随后映射为SGLang支持的`top_logprobs_num`，不向服务端硬塞未知同名字段。

json模式读取`meta_info.output_top_logprobs`；binary模式读取`output_topk_token_ids/log_probs/shape`，缺失时直接报错，不自动降级。

Harbor最终会JSON序列化Trial结果，所以`LLMResponse.extra.score_centering_head`使用base64字符串、shape、version和digest，而不是直接塞np.ndarray。进入Sample前再解码成紧凑CPU数组。digest绑定原prompt、生成token及head数据；解码同时检查行数、k、dtype、重复/负数/越界ID、非有限logp、head质量及采样token概率。

### 4.2 三个SC-only字段

| 字段 | dtype / shape | 生命周期 |
|---|---|---|
| `rollout_topk_token_ids` | NumPy int32 `[R,k]` | Sample→Ray→Actor CPU tensor |
| `rollout_topk_log_probs` | NumPy float32 `[R,k]` | 同上；全词表归一化的log q |
| `sc_response_token_ids` | NumPy int32 `[R]` | 每行真实response token的独立身份 |
| `sc_old_head_log_probs` | torch.float32 CPU `[R,k]` | 仅weighted/old_actor训练侧生产和消费 |

R包括response内部工具/环境上下文。三项Sample字段只在SC开启时动态附加，原`to_dict/from_dict`已经支持恢复附加字段；SC-off不因新增默认None字段而改变Sample字典。

### 4.3 真实裁切与历史“丢弃”计数不能混用

`trajectory_retokenized`生产构建器里的`prompt_suffix_dropped_tokens`、`response_prefix_dropped_tokens`等，有些只表示**匹配下一轮历史投影时的丢弃**；实际生成token仍保留。按这些统计值直接砍head会造成错位。

本实现依据真正落入`sample.tokens`的context/action片段组装head，逐段验证Token ID。response里的工具token填有限dummy head，mask=0；SC在softmax/gather前排除这些行，并在标量计算前再次where清理，防止`0 * NaN/Inf`。

真正改变response的是`sample_postprocess.cap_total_tokens_allow_response_trim_in_samples`。它调用`_slice_sc_tracks`，使用相同切片同步：tokens、mask、sampled logps、head IDs、head logps、row token IDs；返回独立数组，避免短切片仍持有完整长轨迹底层buffer。

在DP调度、old-actor/critic前向、GAE或CP切分之后再裁切会破坏全部对应关系。因此SC-on的rollout_data阶段hook只允许“已不超过上限”的数据通过，发现晚期超长会报错，要求在Sample阶段处理。

### 4.4 TP、CP、PP和归约

- **TP：** 自定义`_HeadLogProbs`对完整分片词表计算归一化，再按global token ID取head。前向使用TP小张量MAX/SUM；反向不再对复制的loss梯度做TP all-reduce，避免倍增。
- **显存：** 以`sc_chunk_size`限制softmax工作区，直接从packed logits选行，不为每个sample复制大词表矩阵。反向仍需要完整logits梯度buffer，不能只按head大小估算峰值。
- **CP：** Actor侧保留完整CPU head。`response_row_layout`明确映射每个CP rank的预测logit行与response行，注意token预测的−1偏移。allgather路径需要恢复优势、选择连续行，再将addend重分布回PPO使用的zigzag布局。
- **PP：** 只有产生最终logits的PP-last处理SC。非PP-last Actor不保留SC head，Critic关闭SC。每rollout重置挂载标记，防止“前一轮算过”掩盖本轮未触达。
- **归约：** 使用原`loss_masks`、`group_mask_sums`和原per-token/per-group规则，不在SC外层另造batch均值；矩阵字段从常规标量日志中跳过。

## 5. warm Critic、关闭态和角色隔离

| 项目 | Actor | Critic |
|---|---|---|
| lr/min_lr | 5e-7 | 1.5e-5 |
| 比例 | 1 | 30 |
| clip_grad | 原顶层继承，基线为1.0 | 100.0 |
| value_clip | 无value head | 4.0 |
| SC | 公共开关 | 强制关闭 |

角色加载继续使用原`_apply_megatron_role_overrides`更新LR等普通字段；SC字段单独校验，避免两端不同步。可以在Critic overrides显式写`enable_score_centering: false`，但不必写；true会报错。Actor只允许重复声明与公共配置相同的SC值。

warm Critic仍由原`_isolate_critic_checkpoint_dirs`发现`<load>_critic` sibling，沿用weights-only加载及`critic_load_optim`例外。SC不改写load/save/no_load_optim/no_load_rng/finetune。

原run2入口默认`NUM_CRITIC_ONLY_STEPS=2`：前两轮只训练Critic，**即使SC已启用也不会出现Actor SC loss**。要复现仅1-step的SC冒烟，必须像9月26日作业那样设为0；正式训练是否保留warmup是原训练方案的决定，不能为了看指标擅自去掉。

SC-off路径不请求、不传输、不计算Head；PPO原算子和张量反向路径不因SC改变。9月26日用冻结的实施前工作区对照，Harbor/core的loss、logits梯度、AdamW更新及指标均逐bit一致。这里不承诺新增Python判断、配置读取或整个程序字节码相同，也不把受控离线比较扩大为当前整个生产训练作业的永久零回归保证。

## 6. 参数、前置条件和常见错误

### 6.1 当前默认值

| 参数 / 内部字段 | 默认值 | 说明 |
|---|---|---|
| `--use-score-centering` / `enable_score_centering` | false | 总开关；旧CLI别名`--enable-score-centering` |
| `--score-centering-top-k` / `sc_top_k` | 32 | 旧CLI别名`--sc-top-k`；整数1–1024且不超词表 |
| `--sc-formula` | score | score与ppo_weighted二选一 |
| `--sc-wire-format` | binary | **当前已验证原版引擎要显式设json** |
| `--sc-correction-scale` | 1.0 | 有限非负；0为诊断、1为完整加项 |
| `--sc-tail-eps` | 1e-8 | 启动校验要求固定值 |
| `--sc-weight-denominator` | old_actor | 仅weighted消费，必须等于实际PPO分母 |
| `--sc-assert-alignment` | full | 当前只允许full |
| `--sc-chunk-size` | 256 | SC工作区分块，独立于原log_probs_chunk_size |
| `--sc-max-response-bytes` | 268435456 | 单请求预检/响应raw head预算，256MiB |
| `--sc-max-trial-bytes` | 536870912 | 单LLM实例累计请求预留预算，512MiB |
| `--sc-max-batch-bytes` | 8589934592 | Ray打包前raw head总预算，8GiB |

单LLM实例的trial预算按每次请求`max_new_tokens`预留，失败/取消不返还；是保守阈值，不是实际生成字节精确计费。raw head每token约`8*k+4`字节，k32约260字节，不包括base64、JSON、进程副本及临时矩阵。预算不等于整个Ray job的主机内存硬上限。

### 6.2 必须满足的配置

- Megatron后端、`advantage_estimator=ppo`。
- `loss_type=policy_loss`，或者Harbor指定custom_loss：`slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled`。
- rollout必须为`slime_plugins.harbor.rollout.generate_rollout`；当前SC校验不接受`fully_async_rollout.generate_rollout_fully_async`。
- Harbor mode为`trajectory_normalized`或`trajectory_retokenized`，LLM backend为`slime_sglang`，`collect_rollout_details=true`。
- 实际每次请求满足temperature=1、top_p=1、top_k=−1、min_p=0；无presence/frequency/repetition penalty改动或结构化约束/logit_bias。
- 不组合TIS、OPSM、OPD、mismatch metrics、unbiased KL、MTP或dual-clip；自定义数据回调仅允许已审计的cap hook等明确支持范围。
- 不组合当前TB-GAE；不要遗留TB-GAE专用custom_advantage hook。
- BSHD+allgather-CP被SC拒绝；THD allgather还要通过原Megatron针对模型架构的额外校验。不能因为SC有布局代码就认为27B可任意开启allgather-CP。

### 6.3 常见现象与排查

| 现象 | 首先检查 |
|---|---|
| binary response missing | 默认wire=binary但SGLang没输出binary字段；改用json或另行准备并验收隔离引擎 |
| sc_applied=0 | 是否使用了现成shadow配置，scale仍为0 |
| 前两轮没有SC指标 | run2的critic-only warmup仍为2，Actor未参与 |
| 开启后role override报错 | 是否只在Actor overrides设置enable/k，而公共rollout仍使用别的值 |
| TB-GAE requires enable_score_centering=False | 关闭TB-GAE；当前不支持二者组合 |
| SC sampled-token logprob differs | Head/token搬运错位或采样分布不一致；不要关闭对齐断言掩盖 |
| Head行数与response不符 | 检查真实cap是否同步三项Head字段，是否误按历史投影drop计数裁切 |
| SC trial head budget exceeded | 累计预留的max_new_tokens超过单LLM预算；评估turn数/请求长度和容量 |
| scale=0仍有额外内存/网络 | 正常：诊断模式仍采集和计算head；总开关false才关闭链路 |
| TorchMemorySaver报expandable_segments不支持 | 清除本Job的PYTORCH_ALLOC_CONF/PYTORCH_CUDA_ALLOC_CONF相关设置，不改全局库 |
| TP4/PP1单机初始化OOM | 9月26日改训练TP2/PP2后通过；不能只看27B权重/TP估算GDN与梯度buffer |

## 7. 如何确认SC真的生效

必须同时看到**配置、数据、损失、梯度**四层证据：

1. 最终公共args和Actor args中enable=true、formula正确、scale=1；Critic=false。不能只看CLI文本或一个没被读取的YAML字段。
2. SGLang响应和Harbor结果包含每轮head；训练batch中三个SC字段存在，shape与response匹配。
3. Actor Step日志有`train/sc_applied=1.0`，并有head质量、hit、logp deviation及correction指标。
4. backward完成且梯度有限，optimizer.step成功；仅有`sc_applied`说明前向挂载已执行，本身不足以证明反向完成。

`sc_head_hit`按loss mask归约，Top-k之外的采样token原则上可以合法存在；不要把任何小于1的hit率一概判为错位。每当采样token命中head，独立sampled logp应与head中该token概率对账。原始诊断均值如果包含工具mask=0行，会比最终masked指标低。

`sc_correction`是SC函数产生、随后供乘A的标量项诊断，不应直接解释为“整个训练loss增加了这个数值”。

## 8. 已完成的验收及单节点复用说明

### 8.1 2026-09-26 离线验收

结果为**302 passed、0 failed、0 skipped**，另有`TP2×CP2`和`TP1×CP4`两个四进程gloo任务通过。覆盖参数、数学、codec、Token/cap、角色/暖启动、Ray分区、old-policy producer、真实旧源码回归等。

结果文件：[`PPO_SC_STAGE12_TEST_RESULTS.json`](PPO_SC_STAGE12_TEST_RESULTS.json)。统一runner：[`tools/test_ppo_sc_offline.py`](tools/test_ppo_sc_offline.py)。历史基线目录在本次整理时仍存在：

```bash
cd /path/to/workspace/slime-terminal-RL
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  python3 -B tools/test_ppo_sc_offline.py \
  --baseline /tmp/ppo-sc-stage12-46ux60iv/before \
  --output-dir /tmp/ppo-sc-validation-20260929
```

这是**供之后复测的命令**，9月29日本轮没有重跑该套件。当前多了TB-GAE等代码，旧基线对照出现差异时必须区分新功能增量与SC回归，不能替换成实施后baseline以制造通过结果。

### 8.2 2026-09-26 真实GPU冒烟

| 项目 | 实测 |
|---|---|
| 成功Job | `sc_singlenode_27b_20260926_095713-r3` |
| 时间UTC | 2026-09-26 10:12:55.867–10:27:51.158 |
| 节点 | <node-ip>，单机8×H20 |
| 训练/采样布局 | 训练TP2×PP2×CP1×DP1=4卡；SGLang TP4=4卡 |
| 规模 | 1 rollout；Critic和Actor各Step0；2条Harbor轨迹 |
| Daytona并发 | 2，不是20并发压力测试 |
| SC | score、json、k32、scale1 |
| 结果 | SUCCEEDED，exit0；两条任务reward均1.0 |
| loss / pg_loss | 0.9357990026473999 |
| sc_applied | 1.0 |
| sc_head_hit | 1.0 |
| sc_head_logp_deviation | 0.0 |
| sc_correction | 0.03517688065767288 |
| Actor grad_norm | 18.64021570694831 |
| Critic value_loss | 1.4477572441101074 |
| Critic grad_norm | 210.30279509554788 |

两条sample的Head为`[526,32]`、`[643,32]`；训练token数分别271、372。PP-last的rank2/rank3均记录两次SC前向和反向钩子，反向有限、每microbatch gradient_abs_sum=0.5。step后完成权重同步并正常退出。

在线样本总长1309/1430，小于4096，故在线未触发hard cap。之后用同一真实轨迹回放cap256，分别丢弃271/388个response行、保留255行，六轨同步通过。这是**真实数据离线裁切回放**，不能称为GPU在线裁切被实际触发。

### 8.3 单节点脚本在哪里

成功冒烟的独立运行目录：

`/path/to/workspace/ppo_sc_smoke_runs/sc_singlenode_27b_20260926_095713`

可查阅：

- [独立 launch.sh](../ppo_sc_smoke_runs/sc_singlenode_27b_20260926_095713/launch.sh)：实际单机TP2/PP2与SGLang TP4参数。
- [roles_smoke.yaml](../ppo_sc_smoke_runs/sc_singlenode_27b_20260926_095713/roles_smoke.yaml)：原warmcritic角色参数和真实SC开关。
- [harbor_smoke.yaml](../ppo_sc_smoke_runs/sc_singlenode_27b_20260926_095713/harbor_smoke.yaml)：2并发、3轮上限、Daytona delete=true。
- [smoke_driver.py](../ppo_sc_smoke_runs/sc_singlenode_27b_20260926_095713/smoke_driver.py)：限时、单节点断言、模型/PG清理。
- [smoke_hooks.py](../ppo_sc_smoke_runs/sc_singlenode_27b_20260926_095713/smoke_hooks.py)：SC前向/反向观测钩子，运行结束后不在基线中加载。
- [submit.py](../ppo_sc_smoke_runs/sc_singlenode_27b_20260926_095713/submit.py)：仅临时单机Ray Job的环境注入。
- [完整报告](../ppo_sc_smoke_runs/sc_singlenode_27b_20260926_095713/REPORT_zh.md)和[SUMMARY.json](../ppo_sc_smoke_runs/sc_singlenode_27b_20260926_095713/SUMMARY.json)。

该目录的**venv、鉴权文件和临时Ray均已清理**，不能今天直接`bash launch.sh`就重跑。复用时要创建新的run目录、恢复隔离依赖、检查目标节点空闲、启动仅本次使用的Ray，再提交带期限的job。不要重新使用旧trial目录或把历史Job状态当成今天节点状态；本次整理没有访问远端节点。

### 8.4 冒烟中发现的问题与隔离要求

- Ray零GPU driver隐藏CUDA，而Megatron参数校验读取GPU架构；只在专用driver中恢复可见性，模型worker仍由PG约束。
- TorchMemorySaver不兼容expandable_segments，清除了该Job的allocator设置。
- TP4/PP1的Actor梯度buffer曾OOM；TP2/PP2通过。27B只有4个KV groups，不能简单改TP8解决。
- SGLang的`max_total_tokens=16384`、`max_mamba_cache_size=4`适用于此次小任务；不是大并发配置。
- 不直接patch `/sgl-workspace/sglang`；json已完成真实验证。binary若将来需要，另建完整隔离引擎并验收，不复用损坏的历史top_p.patch。
- 每个Job的Triton、Torch扩展、Inductor、CUDA、FlashInfer、TVM-FFI和W&B缓存都应显式放在run-local目录，不修改HOME或全局shell配置。

实际GPU作业期间，TVM-FFI曾忽略XDG_CACHE_HOME，在默认`/root/.cache/tvm-ffi`新建3个目录；确认仅含本次文件后已删除，后续提交器补了`TVM_FFI_CACHE_DIR`。所以历史结论是**全局包/源码未改、结束后这些临时目录无残留**，不是“运行期间所有默认缓存都绝对未写过”。

2026-09-26清理记录确认：2个任务沙箱与1个canary均删除，8卡0MiB/0%，临时端口关闭，380项全局包版本与5474个SGLang文件及427个原仓库文件核验一致。文档中不保存任何Daytona/W&B Key；凭证应仅注入作业环境，重试时使用新的受控凭证输入。

## 9. 2026-09-29 复核：当前工作区与历史快照的区别

本次只读比对了9月26日GPU作业保存的427个文件指纹，发现以下8个原路径内容已有变化：

```text
slime/backends/megatron_utils/data.py
slime/backends/megatron_utils/loss.py
slime/ray/rollout.py
slime/utils/arguments.py
slime_plugins/harbor/atif_adapter.py
slime_plugins/harbor/rollout.py
slime_plugins/harbor/sample_postprocess.py
slime_plugins/harbor/tb_gae.py
```

已展开检查的共用调用链增量主要为TB-GAE参数解析、turn mask、优势分派和裁切支持；SC核心数学、配置模块与pg_objectives仍是当前说明的实现。新加入的`slime/utils/tb_gae_config.py`明确禁止TB-GAE+SC。

因此：

- 本文的“如何开启”依据9月29日实际配置代码，明确要求当前仅开SC时关闭TB-GAE。
- 302项离线通过和GPU step成功属于9月26日快照，不代表9月29日全部改动已复测。
- 单步成功不等于256K、多节点、20并发、长期训练、weighted GPU路径或算法收益已验收。
- 当前Git HEAD仍是`2cf4f3066ff661fb376dde8dc9f157fff8f017fe`，但dirty/untracked文件包含实际实现，单靠HEAD无法完整复现实验。

## 10. 源码与证据入口

| 目的 | 入口 |
|---|---|
| SC数学实现 | [score_centering.py](slime/utils/score_centering.py) |
| 开关默认值/公共角色块 | [score_centering_config.py](slime/utils/score_centering_config.py) |
| 开启条件/不支持组合 | [score_centering_validate.py](slime/utils/score_centering_validate.py) |
| 当前TB-GAE互斥限制 | [tb_gae_config.py](slime/utils/tb_gae_config.py) |
| 实际Harbor损失挂载 | [train_loss.py](slime_plugins/harbor/train_loss.py) |
| TP/CP/old head训练拼接 | [sc_terms.py](slime/backends/megatron_utils/sc_terms.py) |
| 历史完整设计 | [PPO_SC_INTEGRATION_PLAN.md](PPO_SC_INTEGRATION_PLAN.md) |
| Stage1/2实施记录 | [PPO_SC_STAGE12_IMPLEMENTATION_REPORT_zh.md](PPO_SC_STAGE12_IMPLEMENTATION_REPORT_zh.md) |
| 离线机器结果 | [PPO_SC_STAGE12_TEST_RESULTS.json](PPO_SC_STAGE12_TEST_RESULTS.json) |
| 真实GPU证据 | [成功冒烟目录](../ppo_sc_smoke_runs/sc_singlenode_27b_20260926_095713/) |

**本次文档整理没有运行配置生成命令、测试套件、Ray作业或训练，只新增本文件。**
