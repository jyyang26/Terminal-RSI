# TB4 · Qwen3.8-27B Dense Terminal PPO 256k —— 启动与运维手册

本文档说明如何拉起当前这一轮训练（run #2：warm critic + CTRF 82 + per-family 分母 + shuffle）。
超参数的完整取值与 GAE / Reward 的数学定义见 **`TB4_27B_hyperparameter.html`**。

当前在跑的实例：

| 项 | 值 |
|---|---|
| Run tag | `PROD3EP_RUN2_20260921_083327` |
| Ray job | `raysubmit_GjTqnv8c5aYxy4kB` |
| 日志 | `/path/to/workspace/PROD_run2_PROD3EP_RUN2_20260921_083327.log` |
| 拓扑 | 64 train + 56 infer = 120 GPU（15 节点） |
| 步数 | 36（3 epoch），`save_interval=5`，前 4 步 critic-only |

---

## 0. TL;DR —— 一条命令

```bash
cd /path/to/workspace/slime-terminal-RL

RUN_TAG="PROD3EP_RUN2_$(date +%Y%m%d_%H%M%S)"
LOG=/path/to/workspace/PROD_run2_${RUN_TAG}.log

NODE_IP_LIST="$(cat /tmp/node_ip_list_15.txt)" RUN_TAG="$RUN_TAG" \
NUM_ROLLOUT=36 SAVE_INTERVAL=5 NUM_CRITIC_ONLY_STEPS=4 \
  setsid nohup bash examples/harbor_terminal_rl/launch_27b_run2_warmcritic_ctrf82.sh \
  > "$LOG" 2>&1 < /dev/null &

echo "$RUN_TAG" > /tmp/prod2_tag.txt
echo "$LOG"     > /tmp/prod2_log.txt
```

**必须用 `setsid nohup ... &`**：训练要跑约 3 天，脱离终端会话才不会因 SSH 断开被杀。

启动约 12 分钟后从日志里抓 job id 并挂上 keepalive 看门狗：

```bash
JOB=$(grep -oE 'raysubmit_[A-Za-z0-9]+' "$(cat /tmp/prod2_log.txt)" | head -1)
echo "$JOB" > /tmp/prod2_job.txt; echo "job=$JOB"

cd /path/to/workspace
NODE_IP_LIST="$(cat /tmp/node_ip_list_15.txt)" \
  setsid nohup bash keepalive_watchdog.sh "$JOB" 300 > /tmp/watchdog_prod2.log 2>&1 < /dev/null &
```

---

## 1. 脚本调用链

```
launch_27b_run2_warmcritic_ctrf82.sh          ← 本轮的薄封装：只设「与 run #1 不同」的 env
  └─ run_27b_dense_128gpu_ppo_256k.sh         ← 27B/256k 的拓扑、显存、代理、keepalive
       └─ run_tmax15k_..._strongest_partial_reward.sh   ← 奖励 hook + preflight + 物化 config
            └─ run_tmax15k_..._strongest.sh            ← ADVANTAGE_ESTIMATOR=ppo
                 └─ run_qwen35_9b_terminal_bench2_64h20.sh  ← 组装 TRAIN_CMD、ray job submit
                      └─ train_async.py                     ← 异步训练主循环
```

最外层脚本**只导出环境变量**，不改任何共享脚本。真正的参数在三个文件里：

| 文件 | 负责 |
|---|---|
| `examples/harbor_terminal_rl/launch_27b_run2_warmcritic_ctrf82.sh` | 本轮的 env 覆盖（checkpoint、预热步数、eps_clip、拓扑） |
| `examples/harbor_terminal_rl/harbor_mixed3000_128gpu_27b_dense_daytona_ppo_cp4_256k_run2_ctrf82.yaml` | 任务池、shuffle、agent 配置、**CTRF 分母 + per-family 表** |
| `examples/harbor_terminal_rl/megatron_roles_27b_criticlr30x_warmcritic.yaml` | actor / critic 的 **per-role** lr、value_clip、clip_grad |

---

## 2. 启动前检查清单

逐条确认，任何一条不过都不要启动。

### 2.1 集群有 120 张空闲卡

```bash
RAY_ADDRESS=<node-ip>:23456 ray status | grep GPU
# 期望：0.0/120.0 GPU
```

### 2.2 GPU keepalive 在 15 个节点上全部在跑

```bash
cd /path/to/workspace
NODE_IP_LIST="$(cat /tmp/node_ip_list_15.txt)" ./keepalive_ctl.sh start
NODE_IP_LIST="$(cat /tmp/node_ip_list_15.txt)" ./keepalive_ctl.sh status
# 期望：15 行全部 procs=1
```

> **这不是可选项。** 平台会回收长时间低利用率的集群，而启动窗口（SGLang 启动 + 351 GB
> checkpoint 加载）有 5–10 分钟处于约 0% 利用率——正是无保护集群被杀的时刻。
> launcher 有硬门：任何节点 `procs=0` 就 `_die`。
>
> **永远不要 `stop` 它。** keepalive 自带退避（`main_dynamics.py` 采样利用率，
> 只在平均利用率低于 `--min_util` 80 时烧算力），真实训练饱和后它什么都不做。

### 2.3 两个 load root 都在，且 tracker 一致

```bash
cd /path/to/workspace/checkpoints
cat q38_27b_ppo256k_run2_loadroot/latest_checkpointed_iteration.txt          # 263
cat q38_27b_ppo256k_run2_loadroot_critic/latest_checkpointed_iteration.txt   # 263
ls -la q38_27b_ppo256k_run2_loadroot q38_27b_ppo256k_run2_loadroot_critic
```

两个目录是一对：

- `q38_27b_ppo256k_run2_loadroot/iter_0000263` → 转换后的 SFT iter263（torch_dist）
- `q38_27b_ppo256k_run2_loadroot_critic/iter_0000263` → run #1 的 `iter_0000009`（训练过的 value head）

`placement_group.py:195` 在 `<load>_critic` 存在时会把 critic 的 `--load` 重定向过去。
两个 tracker 都写 263 是因为 Megatron 按 `<load>/iter_<tracker>` 解析，
critic 侧的 `iter_0000263` 只是指向 run #1 `iter_0000009` 的软链接。

### 2.4 磁盘余量

```bash
df -BG /path/to/shared_storage | tail -1
```

36 步在 `save_interval=5` 下会保存 **8 次**（rollout_id 4/9/14/19/24/29/34 + 最后一步 35），
每次 actor 538 GB + critic 521 GB ≈ **1059 GB**，合计约 **8.27 TB**。

> ⚠️ 本次启动时只有 7.77 TB，**最后一次保存预计差约 0.5 TB**。
> 前 7 次没问题，但需要在中途回收空间。可回收的旧 run：
>
> ```bash
> du -sh /path/to/workspace/slime_harbor_runs/*/ | sort -rh | head
> ```
>
> 删除是**不可逆**操作，确认目录后再动。

### 2.5 CP=4 的依赖

```bash
python3 -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('fla.ops.cp') else 1)" \
  && echo OK || echo "缺 fla-core"
```

GDN 混合线性注意力在 CP>1 时需要 `fla.ops.cp`，它在 **`fla-core`** 包里
（不是 `flash-linear-attention`，尽管报错信息会那样写）。

### 2.6 没有残留进程

```bash
ps -eo pid,cmd= | grep -E 'launch_27b_run2|ray job submit|train_async' | grep -v grep
```

---

## 3. 启动后的验证（按顺序）

### 3.1 preflight（约 60 秒内）

```bash
LOG=$(cat /tmp/prod2_log.txt)
grep -E '^\[run2\]|^\[27b256k\]|^\[preflight\]' "$LOG"
```

必须看到：

```
[run2] critic warmup= 4 critic-only steps
[run2] ctrf scale   = 82 global + per-family p90 table
[run2] eps clip     = low 0.3 / high 5.0
[27b256k] topology   = 64 train (TP4 x CP4 x PP2 => DP2) + 56 infer (4-way TP x 14 engines) = 120 GPU
[27b256k] critic     = WARM (value head from .../q38_27b_ppo256k_run2_loadroot_critic), critic-only steps 4
[preflight] reward   = slime_plugins.harbor.partial_reward.test_count_progress_rewards (SCALE=82)
[preflight] shuffle  = true
[preflight] critic   = WARM <- ... (iter 263, weights-only)
[preflight] all checks passed
```

### 3.2 确认真正下发的训练参数（不要只看脚本回显）

```bash
LOG=$(cat /tmp/prod2_log.txt)
grep -oE '\-\-(num-rollout|save-interval|num-critic-only-steps|eps-clip|eps-clip-high) [^ ]*' "$LOG" | sort -u
```

期望：

```
--eps-clip 0.3
--eps-clip-high 5.0
--num-critic-only-steps 4
--num-rollout 36
--save-interval 5
```

### 3.3 确认 per-role 覆盖真的生效

```bash
grep -oE 'Parsed megatron config for role=[a-z]+ .*overrides = \[[^]]*\]' "$(cat /tmp/prod2_log.txt)" | sort -u
```

期望：

```
role=actor  ... overrides = ['lr', 'min_lr']
role=critic ... overrides = ['lr', 'min_lr', 'value_clip', 'clip_grad']
```

### 3.4 确认 warm critic 没有被重新随机初始化

```bash
LOG=$(cat /tmp/prod2_log.txt)
grep -ci 'Will reinitialize critic' "$LOG"     # 必须是 0
grep -oE 'loading distributed checkpoint from [^ ]* at iteration [0-9]+' "$LOG" \
  | sed 's#.*/checkpoints/##' | sort -u
```

期望看到两条**不同**的路径：

```
q38_27b_ppo256k_run2_loadroot at iteration 263
q38_27b_ppo256k_run2_loadroot_critic at iteration 263
```

> 只要出现 `Will reinitialize critic`，就说明 value head 形状不匹配被重置成随机，
> warm start 失效——此时应停下来检查 critic checkpoint。

### 3.5 确认 CTRF 分母与 family 表进了 args

```bash
LOG=$(cat /tmp/prod2_log.txt)
grep -oE 'harbor_testcount_scale \.+ [^ ]+' "$LOG"
grep -oE 'harbor_family_testcount_scales \.+ .{0,60}' "$LOG"
```

### 3.6 rollout 起来了

```bash
TAG=$(cat /tmp/prod2_tag.txt)
ls /path/to/workspace/slime_harbor_trials/q38_27b_ppo_256k_${TAG} | wc -l
```

启动约 15 分钟后应看到数百个 trial 目录（Daytona 沙箱）。

---

## 4. 日常监控

### 4.1 一行状态

```bash
JOB=$(cat /tmp/prod2_job.txt); LOG=$(cat /tmp/prod2_log.txt); TAG=$(cat /tmp/prod2_tag.txt)
echo "job:    $(RAY_ADDRESS=<node-ip>:23456 ray job status $JOB 2>&1 | grep -oE 'RUNNING|SUCCEEDED|FAILED|STOPPED' | head -1)"
echo "perf:   $(grep -oE 'perf [0-9]+:' "$LOG" | sort -u -t' ' -k2 -n | tail -1)"
echo "steps:  $(grep -c 'Timer train end' "$LOG") actor steps done"
echo "errors: $(grep -cE 'ActorUnavailableError|CUDA error|OutOfMemory|AssertionError|^Traceback' "$LOG")"
echo "ckpt:   $(ls -d /path/to/workspace/slime_harbor_runs/q38_27b_dense_ppo_256k_${TAG}/*/iter_* 2>/dev/null | xargs -n1 basename | sort -u | tr '\n' ' ')"
echo "disk:   $(df -BG /path/to/shared_storage | tail -1 | awk '{print $4}')"
echo "prog:   $(grep -oE '(actor|critic) (train|forward): *[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+' "$LOG" | tail -1)"
```

> **`Timer train end` 只统计 actor 步**，前 4 步 critic-only 不计入。
> 步进度看 `perf N:`（N 是 0-indexed 的 rollout_id）。

### 4.2 关键指标趋势

```bash
python3 - "$(cat /tmp/prod2_log.txt)" <<'PY'
import re,sys
t=open(sys.argv[1],errors='ignore').read()
keys=['rollout/rewards','harbor/reward_mean','rollout/values','rollout/returns',
      'explained_variance','critic-value_loss','critic-grad_norm','critic-value_clipfrac',
      'train/grad_norm','train/pg_clipfrac']
for k in keys:
    v=re.findall(r"'[^']*"+re.escape(k.split('/')[-1])+r"': ([\d.eE+-]+)",t)
    if v: print(f"{k:26s} "+" ".join(f"{float(x):.4g}" for x in v[-12:]))
PY
```

### 4.3 要盯的四个阈值

| 指标 | 健康值 | 触发动作 |
|---|---|---|
| `train/grad_norm`（actor） | < 1.0 | 连续两步 > 5.0 → 把 `EPS_CLIP_HIGH` 从 5.0 降到 1.5，从最近 checkpoint 续训 |
| `pg_clipfrac` | > 0 | 恒为 0 且 `grad_norm` 攀升 → 同上（clip 已完全不起作用，稳定性只靠 clip_grad=1.0） |
| `explained_variance` | 趋向 > 0 | 到 step 10 仍 < −1.0 → warm critic 没对齐新尺度，考虑加长预热或降 critic lr |
| 磁盘 | > 2.5 TB | 低于阈值 → 清理 `slime_harbor_runs/` 里的旧 run |

### 4.4 节点健康

```bash
for ip in $(cat /tmp/node_ip_list_15.txt | sed 's/:[0-9]*//g' | tr ',' ' '); do
  ( r=$(timeout 25 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=6 -o BatchMode=yes "$ip" \
      'nvidia-smi -L 2>&1 | wc -l' 2>/dev/null); echo "$ip gpus=$r" ) &
done; wait
```

每个节点应是 `gpus=8`。少于 8 或报错说明有卡掉了总线——run #1 就是这样死的
（节点 `<node-ip>` 的 GPU 6 掉出 PCI 总线，`ray` 抛 `ActorUnavailableError`）。

---

## 5. 停止训练

```bash
JOB=$(cat /tmp/prod2_job.txt)
RAY_ADDRESS=<node-ip>:23456 ray job stop "$JOB"
```

**这是正确的停止方式**，不要 `ray stop --force`（会杀掉整个 15 节点集群），
也不要 `pkill -f`（见 §7）。

停止后：看门狗会在检测到终止状态时自动恢复 keepalive；最近一次 `iter_*` 保存完整可续训。
确认：

```bash
sleep 60
RAY_ADDRESS=<node-ip>:23456 ray status | grep GPU     # 期望 0.0/120.0
tail -3 /tmp/watchdog_prod2.log                          # 期望 [watchdog] done
```

---

## 6. 常用变体

### 6.1 先跑 3 步做验证

```bash
NUM_ROLLOUT=3 SAVE_INTERVAL=3 NUM_CRITIC_ONLY_STEPS=2 \
NODE_IP_LIST="$(cat /tmp/node_ip_list_15.txt)" \
  setsid nohup bash examples/harbor_terminal_rl/launch_27b_run2_warmcritic_ctrf82.sh \
  > /tmp/smoke3.log 2>&1 < /dev/null &
```

### 6.2 从本轮的某个 checkpoint 续训

需要把 load root 指向该 checkpoint，并**重建 `_critic` 兄弟目录**：

```bash
cd /path/to/workspace
TAG=$(cat /tmp/prod2_tag.txt)
SRC=slime_harbor_runs/q38_27b_dense_ppo_256k_${TAG}
ITER=iter_0000009      # 改成你要续的那个

NEW=checkpoints/q38_27b_resume_${ITER}
mkdir -p "$NEW" "${NEW}_critic"
ln -sfn "$(pwd)/$SRC/Qwen3.8-27B_dense_ppo_256k_mixed3000/$ITER"        "$NEW/$ITER"
ln -sfn "$(pwd)/$SRC/Qwen3.8-27B_dense_ppo_256k_mixed3000_critic/$ITER" "${NEW}_critic/$ITER"
printf '%d\n' "$((10#${ITER#iter_}))" | tee "$NEW/latest_checkpointed_iteration.txt" > "${NEW}_critic/latest_checkpointed_iteration.txt"

PRETRAINED_CKPT="$(pwd)/$NEW" START_ROLLOUT_ID=10 \
NODE_IP_LIST="$(cat /tmp/node_ip_list_15.txt)" \
  setsid nohup bash slime-terminal-RL/examples/harbor_terminal_rl/launch_27b_run2_warmcritic_ctrf82.sh \
  > /tmp/resume.log 2>&1 < /dev/null &
```

> `START_ROLLOUT_ID` 必须显式给：否则 wandb 的 x 轴会从 0 重启并覆盖前一段曲线。
> 续训时 `NUM_CRITIC_ONLY_STEPS=0` 更合理（critic 已经对齐尺度了）。

### 6.3 降低 eps_high（策略发散时）

```bash
EPS_CLIP_HIGH=1.5 ...（其余同续训命令）
```

### 6.4 调整 CTRF 分母

**不要**用 `HARBOR_TESTCOUNT_SCALE` 环境变量 —— 它不在内层 launcher 的
`RUNTIME_ENV_JSON` 白名单里，走 `ray job submit` 时会被静默忽略，奖励仍按默认 20 算。
改 YAML：

```yaml
# harbor_mixed3000_128gpu_27b_dense_daytona_ppo_cp4_256k_run2_ctrf82.yaml
harbor_testcount_scale: 82.0
harbor_family_testcount_scales:
  openfoam_cfd: 795.0
  ...
```

重新标定 per-family 表（用更多 trial）：

```bash
python3 - <<'PY'
import json,pathlib,collections,numpy as np
TR=pathlib.Path("/path/to/workspace/slime_harbor_trials/q38_27b_ppo_256k_<TAG>")
SYM=pathlib.Path("/path/to/workspace/slime-terminal-RL_data/software_reconstruction/mixed3000_27b_ppo_256k_symlink")
fam={}
for t in SYM.iterdir():
    m=t/"tests/private/manifest.json"
    if m.is_file():
        try: fam[t.name]=json.load(open(m)).get("family_id") or "unknown"
        except Exception: fam[t.name]="unknown"
per=collections.defaultdict(list)
for d in TR.iterdir():
    v=d/"verifier"; f=v/"ctrf.json"; c=d/"config.json"
    if not (d.is_dir() and (v/"reward.txt").is_file() and f.is_file() and c.is_file()): continue
    try:
        s=json.load(open(f))["results"]["summary"]
        tid=pathlib.Path(json.load(open(c))["task"]["path"]).name
    except Exception: continue
    n=s['passed']+s['failed']+s['other']+s['pending']
    if n>0: per[fam.get(tid,'unknown')].append(n)
for k,v in sorted(per.items(), key=lambda x:-np.percentile(x[1],90)):
    print(f"  {k}: {np.percentile(v,90):.1f}   # n={len(v)}")
PY
```

---

## 7. 踩过的坑（都是真实事故，不要重犯）

| # | 坑 | 后果 | 正确做法 |
|---|---|---|---|
| 1 | `pkill -f <路径>` | 命令自身的 cmdline 含该路径 → **自杀** | 用 `ps -eo cmd=` 精确前缀匹配选 PID |
| 2 | `ray stop --force` | 杀掉整个 15 节点集群 | 只用 `ray job stop <job>`；单节点故障才在**该节点**上 `ray stop` |
| 3 | 停掉 keepalive | 集群被平台回收 | 永远只 `start`，它自带退避 |
| 4 | `expandable_segments:True` | PPO 强制 `offload_train=True` → TorchMemorySaver 硬报错，加载 checkpoint 前就死 | 固定 `max_split_size_mb:256` |
| 5 | `START_ROLLOUT_ID` 不给 | tracker 是 SFT 的 175 → `range(176, 2) == []`，**循环体一次都不执行**，rollout 跑了 48 分钟全丢、退出码 0 | 新 PPO run 显式 `START_ROLLOUT_ID=0`，续训给实际步数 |
| 6 | `STAGE_HF_CKPT_TO_TMP=1` | 内层脚本把 `STAGED_HF_CKPT` 硬编码成 9B 路径，会用 9B 的 tokenizer 训 27B | 保持 `0` |
| 7 | 用 `HARBOR_TESTCOUNT_SCALE` 改分母 | 不在 runtime env 白名单，静默按 20 算 | 写 YAML |
| 8 | `iter_0000005` 找 step 5 的 checkpoint | `rollout_id` 是 **0-indexed**，step 5 存成 `iter_0000004` | 第 N 步 → `iter_{N-1:07d}` |
| 9 | HF checkpoint 直接给 `--load` | PP>1 时运行时 bridge 对该模型族 `AttributeError on output_layer` | 先离线转 torch_dist（见 §8） |
| 10 | `TP=8` | `num_query_groups=4`，`4 % 8 != 0`，Megatron 与 SGLang 都非法 | TP ≤ 4 |

---

## 8. 附：HF → torch_dist 转换（换新 actor 时需要）

Megatron 的 `--load` 走 torch_dist 读取器（`checkpoint.py:_is_megatron_checkpoint`），
HF 格式的 SFT 导出必须先离线转换。单节点 8 卡、TP1×PP8 约 5 分钟：

```bash
cd /path/to/workspace/slime-terminal-RL
source scripts/models/qwen3.5-27B.sh
export PYTHONPATH="/root/Megatron-LM:$(pwd):${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

HF=/path/to/workspace/checkpoints/q38_full_mix_v3_h20_64_20260916-105043/hf_export/iter_0000263_hf_merged_origin_weights
OUT=/path/to/workspace/checkpoints/q38_27b_iter263_torch_dist

torchrun --nproc_per_node 8 --master_port 12399 \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 8 \
  --distributed-timeout-minutes 60 \
  --hf-checkpoint "$HF" --save "$OUT"
```

> TP 保持 1：mbridge 的 tensor-parallel scatter 在 Qwen3.5 GatedDeltaNet 上可能卡死。
> torch_dist checkpoint 可以在**任意** TP/PP/CP 布局下重新加载，所以 TP1×PP8 转出来的
> 也能被 TP4×CP4×PP2 的训练正常读取。

验证转换结果：

```bash
python3 -c "
from megatron.core.dist_checkpointing.serialization import load_tensors_metadata
md=load_tensors_metadata('$OUT/release')
print('tensors:',len(md))
for k,v in md.items():
    if 'output_layer' in k or 'word_embeddings' in k:
        print(' ',k,tuple(v.global_shape))"
```

期望 `output_layer.weight (248320, 5120)` 与 `embedding.word_embeddings.weight (248320, 5120)`。

然后建 load root（注意转换产物的目录名是 `release`）：

```bash
cd /path/to/workspace
NEW=checkpoints/q38_27b_ppo256k_run2_loadroot
mkdir -p "$NEW"
ln -sfn "$(pwd)/checkpoints/q38_27b_iter263_torch_dist/release" "$NEW/iter_0000263"
printf '263\n' > "$NEW/latest_checkpointed_iteration.txt"
```

---

## 9. 相关文档

| 文档 | 内容 |
|---|---|
| `TB4_27B_hyperparameter.html` | 全部超参数取值、GAE 公式与推导、TB-GAE 对比、Reward 计算链、per-family 分母全表 |
| `paper_list/tb_gae_vs_gae_comparison.html` | TB-GAE vs 标准 GAE 的完整数学对比 |
| `paper_list/reward_and_ppo_optimization_proposal_v2_measured.md` | 基于 run #1 实测日志的优化方案（含 TB-GAE 设计） |
| `slime-terminal-RL_data/software_reconstruction/DENSE_REWARD_CURRENT_IMPLEMENTATION.md` | dense reward 的实际生成链路（judge → CTRF → hook） |
| `CLAUDE.md` | slime/harbor 仓库结构与插件机制 |
