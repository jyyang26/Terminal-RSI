# 集群环境配置手册（Qwen3.5-122B-A10B PPO / 11 节点 88 GPU）

> **最后更新**：2026-08-12
> **适用范围**：`slime-terminal-RL` 上跑 Qwen3.5-122B-A10B PPO，含 **hybrid GDN 上下文并行（CP≥2）**。
> **为什么需要这份文档**：`/usr/local/lib/python3.12/dist-packages` 在 **overlayfs（节点本地）** 上，
> **不是共享文件系统**。所以每加一个节点，Python 依赖都必须**逐节点**重装一遍。历史上因为
> 新节点缺包（`ModuleNotFoundError: tenacity`）和 `fla` 版本不一致，导致过多次训练崩溃。

---

## 0. 一页速查（新节点加入流程）

```bash
NEW_IP=<新节点 IP>
HEAD=head-node

# 1) 补齐依赖（CP>1 必需 fla 0.4.2；Harbor 沙箱必需 daytona/tenacity）
ssh $NEW_IP '
  pip install -U --no-deps "fla-core==0.4.2" "flash-linear-attention==0.4.2"
  # ★ 不要手工列包名 —— 直接把【已验证可跑 rollout 的旧 worker】的整包集合补齐
  #   逐个列名永远会漏（传递依赖 + 惰性 import），见 §6
  mkdir -p /root/.triton_cache_122b_cp2 /root/.triton_cache_122b_cp4
'

# 1b) 用旧 worker 做基准，补齐【所有】缺失包（一次到位，不要打地鼠）
REF=node-03
ssh $REF  'pip list --format=freeze' | sort > /tmp/pip_ref.txt
ssh $NEW_IP 'pip list --format=freeze' | sort > /tmp/pip_new.txt
python3 - <<'EOF' > /tmp/spec.txt
ref={l.split('==')[0].lower().replace('_','-'):l.strip() for l in open('/tmp/pip_ref.txt') if '==' in l}
new={l.split('==')[0].lower().replace('_','-') for l in open('/tmp/pip_new.txt') if '==' in l}
print('\n'.join(ref[k] for k in sorted(set(ref)-new)))
EOF
ssh $NEW_IP "pip install --no-deps $(tr '\n' ' ' < /tmp/spec.txt)"   # --no-deps 防止升级 torch

# 2) 自检（必须全部 IDENTICAL）
bash envs/verify_node_env.sh $NEW_IP

# 3) 加入 Ray（★ 先 ray stop 清理残留，再 join）
ssh $NEW_IP '
  ray stop --force; sleep 3
  mkdir -p /tmp/slime-harbor-ray
  ray start --address=head-node:6379 --num-gpus 8 --temp-dir /tmp/slime-harbor-ray
'

# 4) 复核集群规模
ray status --address=$HEAD:6379 | grep -E "GPU|node_"
```

---

## 1. 集群拓扑

| 项 | 值 |
|---|---|
| 节点数 | **11**（1 head + 10 worker），每节点 **8 × H20 (95 GiB)** = **88 GPU** |
| Ray head | `head-node:6379` |
| Ray temp dir | `/tmp/slime-harbor-ray`（各节点本地） |
| Host RAM | 每节点 **2265 GB** |

### 节点 IP 清单

```
head : head-node
N02  : node-04
N03  : node-05
N04  : node-06
N05  : node-07
N06  : node-08
N07  : node-09
N08  : node-10
N09  : node-03
N10  : node-11      <- 2026-08-12 新增
N11  : node-12      <- 2026-08-12 新增
```

> 节点之间 `ping` 不通是 ICMP 被禁，TCP:22 正常，不影响 Ray/NCCL。

---

## 2. 基准环境版本（全 11 节点必须一致）

以 head 节点为基准，2026-08-12 实测。**加粗项是 CP 训练的硬依赖。**

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.3 | 系统 python，非 conda |
| NVIDIA driver | 535.161.08 | |
| GPU | NVIDIA H20 × 8 | 每卡 95 GiB 可用 |
| CUDA (torch) | 12.9 | |
| nvcc | 12.9 (V12.9.86) | 在 `/usr/local/cuda/bin/`，训练不调用，PATH 里没有也无妨 |
| torch | **2.11.0+cu129** | |
| NCCL | 2.28.9 | torch 自带 |
| triton | **3.6.0** | GDN kernel 靠它 JIT |
| **fla-core** | **0.4.2** | ★ CP 必需：提供 `fla.ops.cp` |
| **flash-linear-attention** | **0.4.2** | ★ 同上，与 fla-core 版本必须一致 |
| transformer_engine | 2.10.0 | full-attention 的 CP（ring/p2p） |
| flash_attn | 2.7.4.post1 | |
| megatron.core | 0.16.0rc0 | 位于 `/root/Megatron-LM`（**已打 patch，见 §4**） |
| sglang | 0.5.12.post1 | 位于 `/sgl-workspace/sglang`（自带 vendored fla，**不要动**） |
| transformers | 5.6.0 | |
| ray | 2.55.1 | |
| numpy | 1.26.4 | |
| wandb | 0.27.2 | |
| torch_memory_saver | present | PPO 的 offload 靠它 |
| **tenacity** | **9.1.4** | ★ Harbor 沙箱重试；缺了直接 `ModuleNotFoundError` |
| **daytona** | **0.168.0** | ★ 沙箱后端 SDK |
| daytona-api-client | 0.168.0 | 随 daytona 装上 |
| **shortuuid** | **1.0.11** | ★ `slime_plugins.harbor.agent` 的传递依赖；缺了 rollout 全废（见 §9） |
| **litellm** | **1.89.3** | ★ 同上。版本必须精确对齐：装成 1.80.5 会 import 失败 |
| pybase64 | 1.4.3 | routing replay 的 payload 解码 |

### 为什么 fla 必须是 0.4.2

CP>1 依赖 `fla.ops.cp`（`build_cp_context` / `FLACPContext`），以及
`chunk_gated_delta_rule` / `causal_conv1d` 上的 `cp_context=` 参数。**0.4.1 完全没有这个模块**。

`0.4.1 → 0.4.2` 的升级安全性已实测验证：

- GDN **前向 bit-identical**（同输入 sha256 一致）→ train/rollout logprob 一致性不受影响
- 反向差异 ≤ 3 bf16 ULP，且**低于 0.4.1 自身的 run-to-run 噪声**（GDN 反向本身非确定性）
- `state_dict` key 与形状完全不变 → 旧 checkpoint 可直接加载

> **回滚**：安装时会把旧版备份到 `/tmp/fla041_backup/fla`，需要时 `cp -r` 覆盖回去即可。

### 一个坑：`FLA_INTRACARD_CP`

fla 0.4.2 新增了一条 `intracard_cp` 快路径，**默认开启**
（`os.environ.get("FLA_INTRACARD_CP","1") != "0"`），在 `inference_mode` + 序列 ≥24576 token 时
自动接管，会改变数值。slime 用的是 `no_grad` 而非 `inference_mode`，实测**不会触发**，
但脚本仍然在 CP>1 时显式设 `FLA_INTRACARD_CP=0` 作为保险。

---

## 3. 逐节点安装步骤

```bash
# ---- 3.1 CP 依赖（每个节点都要做，site-packages 是节点本地的）----
pip install -U --no-deps "fla-core==0.4.2" "flash-linear-attention==0.4.2"
#   --no-deps 是刻意的：避免 pip 顺手把 torch/transformers 升级掉

# ---- 3.2 Harbor rollout 依赖（4 个都要，缺一个 rollout 就全废）----
# (原注释) Harbor 沙箱依赖 ----
pip install "tenacity==9.1.4" "daytona==0.168.0" "shortuuid==1.0.11" "litellm==1.89.3"
#   shortuuid / litellm 是 slime_plugins.harbor.agent 的【传递】依赖，
#   不在任何 requirements 里显式列出 -> 只能靠 §6 的端到端 import 探测发现

# ---- 3.3 Triton 持久化缓存目录（节点本地，勿放共享 fs）----
mkdir -p /root/.triton_cache_122b_cp2 /root/.triton_cache_122b_cp4
```

### 为什么 Triton 缓存必须持久化且节点本地

变长轨迹 + 动态 batching 会产生几十种不同的 padded shape，**每种新 shape 都触发一次
GDN/Triton JIT 编译**（几十到几百秒）。2026-08-11 的崩溃就是这样来的：某个 rank 卡在
>30 分钟的编译里，它的 15 个 expert-parallel 同伴已经进入 MoE all-to-all，NCCL
30 分钟 watchdog 打在**同伴**身上，而卡住的 rank 自己没超时 → 日志表现为
"15 个超时 + 1 个静默"。

三个缓解手段（脚本在 CP>1 时自动设置）：

| 设置 | 作用 |
|---|---|
| `DATA_PAD_SIZE_MULTIPLIER=2048` | shape 收敛：实测 43 种 → 7 种 |
| `TRITON_CACHE_DIR=/root/.triton_cache_122b_cp{CP}` | 编译结果跨 step / 跨进程复用 |
| `SLIME_NCCL_GROUP_TIMEOUT_MINUTES=120`<br>`SLIME_NEW_GROUP_TIMEOUT_MINUTES=120` | 给合法的编译时间留出余量 |

放在 `/root`（overlayfs 本地）而**不是** `/path/to/shared_storage`：共享 fs 上 88 个 rank 抢同一把
文件锁会更慢，且可能死锁。

---

## 4. Megatron 的 patch（routing replay 必需）

`--use-rollout-routing-replay` 依赖一处对 Megatron 的 monkey-patch，它**已经打在**
`/root/Megatron-LM` 里（`docker/patch/qwen35-moe-*/megatron.patch`）：

```python
# megatron/core/transformer/moe/moe_utils.py:714
from slime.utils.routing_replay import get_routing_replay_compute_topk
compute_topk = get_routing_replay_compute_topk(compute_topk)

# megatron/core/transformer/moe/router.py:210
from slime.utils.routing_replay import register_routing_replay
register_routing_replay(self)
```

自检：

```bash
grep -c get_routing_replay_compute_topk /root/Megatron-LM/megatron/core/transformer/moe/moe_utils.py  # 应为 1
grep -c register_routing_replay        /root/Megatron-LM/megatron/core/transformer/moe/router.py      # 应为 1
```

`/root/Megatron-LM` 也是**节点本地**的，新节点若来自同一镜像则自带 patch；
上面两条 grep 返回 0 的话必须补打。

> `ENABLE_ROUTING_REPLAY` 环境变量**不需要**手工加进 `RUNTIME_ENV_JSON` ——
> `slime/ray/actor_group.py:86` 会在 `use_routing_replay=True` 时逐 actor 自动注入
> （且**只给 actor，不给 critic**）。

---

## 5. 共享文件系统路径（所有节点都必须能读到）

| 路径 | 内容 |
|---|---|
| `/path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B` | HF 权重（245 GB），SGLang 读它 |
| `.../Qwen3.5-122B-A10B_torch_dist` | Megatron torch_dist 权重（228 GB），`--load` 指它 |
| `.../slime-terminal-RL_data/tmax15k_terminal_bench` | 训练任务池（14601 个 task.toml） |
| `.../terminal-bench-2-main` | eval 任务池（89 个任务） |
| `.../slime-terminal-RL` | 代码（含 CP 的新代码，走共享 fs，改一次全节点生效） |
| `.../harbor-code-synthesis/src` | Harbor agent 源码（`HARBOR_SRC`） |

`/path/to/shared_storage` 是 `fuse.dop-fuse` 挂载（rw）；
`/path/to/shared_storage` 本身是 xfs **只读**。

---

## 6. 环境自检脚本

`envs/verify_node_env.sh <ip>` 会核对上面所有版本 + 共享 fs + Megatron patch，
**并且真正 import 一遍 Harbor/CP 的入口模块**（`harbor_imports` 字段）。

> ★ **正确的检查是整包集合 diff，不是列包名，也不是 import 探测。** 2026-08-12 连踩三次：
> 逐项版本表通过 → 缺 `shortuuid`；端到端 import 探测通过 → 又缺 `litellm`；
> 修完仍缺 `dirhash` —— 因为 harbor 在**函数体内惰性 import** 它
> (`harbor/models/task/task.py:168`)，任何模块级探测都看不见。
> 最终发现新节点比旧 worker **少 98 个包**。所以 `verify_node_env.sh` 现在比对
> `pip_names` 全集（只报"缺失"，多出来无害），基准是**已验证可跑 rollout 的旧 worker**
> 而不是 head（head 装了 gradio/nemo/sklearn 等 dev 工具，拿它当基准会全员误报）。
>
> 附带说明，早先的端到端 import 探测仍然保留（`harbor_imports` 字段），因为它能抓
> 版本不兼容这类"包在但 import 失败"的问题。2026-08-12 两个新节点
> 通过了 25 个包的逐项检查，仍然在 rollout 阶段全部失败于
> `ModuleNotFoundError: shortuuid`，修掉后又暂露出 `litellm` 也缺。
不带参数则检查全部 11 个节点并 diff。

```bash
bash envs/verify_node_env.sh              # 全集群
bash envs/verify_node_env.sh node-11  # 单节点
```

**2026-08-12 实测结果：11/11 节点 IDENTICAL**（唯一差异是非交互 shell 的 `nvcc` PATH，无影响）。

---

## 7. 凭证

不写进 git。放在各自机器的 `/tmp/bs512_keys.env`（`umask 077`，mode 600）：

```bash
umask 077
printf 'export DAYTONA_API_KEY=...\nexport WANDB_API_KEY=...\n' > /tmp/bs512_keys.env
```

训练脚本会在 `DAYTONA_API_KEY` 未导出时自动 source 它，缺失则直接报错退出。

> ⚠️ 不要去扫**别人** job 的 `/proc/*/environ` 找 key（会被安全策略拦截）。
> 只能从**自己**起过的 ray job 的 `runtime_env` 里恢复。

---

## 8. 常见故障 → 定位

| 现象 | 根因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: tenacity`/`daytona`/`shortuuid`/`litellm` | 新节点没装 | §3.2 |
| rollout 所有 trial 报 `RayTaskError(ImportError)`，但训练侧正常 | 该节点缺 Harbor 传递依赖；只有被分到 rollout 角色的节点会暴露 | §3.2 + 跑 §6 自检 |
| `ModuleNotFoundError: fla.ops.hyd` / `fla.ops.cp` | fla 还是 0.4.1 | §3.1 |
| CP>1 启动即报 `Hybrid CP requires fla.ops.cp` | 同上；脚本的预检会在秒级拦住 | §3.1 |
| 15 个 rank NCCL 超时 + 1 个静默 | Triton 冷编译掉队 | §3.3 三件套 |
| `OSError: Repo id must be...` | 试图从节点本地 `/tmp` 读 HF 权重 | `STAGE_HF_CKPT_TO_TMP=0` |
| 起训练后 worker 立刻被停 | 上次失败留下 `.stop_workers_*` marker | 脚本会自动清；或手工删 `${BASE_FOLDER}/.stop_workers_*` |
| 整个集群掉线 | `CLEAN_START=1` 杀掉了 head | 复用集群时**必须** `CLEAN_START=0`（脚本已默认） |
| `python3: can't open file '.../train_async.py'` | ray head 的 cwd 不对 | 起集群前先 `cd slime-terminal-RL` |

---

## 9. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-12 (3) | **发现新节点比旧 worker 少 98 个包**，不是先前以为的 3 个。`88cp8-128k` 首启后 rollout 在 N11 上抛 144 次 `ModuleNotFoundError: dirhash` —— 而 `dirhash` 是 harbor 在**函数体内惰性 import** 的，模块级探测永远抓不到。已按"旧 worker 整包集合"一次补齐 98 个包（`--no-deps`，核心版本未变），并把 `verify_node_env.sh` 的判据改为 **pip 全集缺失 diff + 基准换成旧 worker**，`nvitop` 列入白名单。复检 11/11 IDENTICAL。教训：跨节点环境对齐只能用整包集合，逐个列名/import 探测都会漏。 |
| 2026-08-12 (2) | **补 `shortuuid==1.0.11` + `litellm==1.89.3` 到 N10/N11**。起因：`88cp4-84k` 首次启动后 rollout 在 N11 上抛 280 次 `ModuleNotFoundError: shortuuid`，该节点 3 个 trial worker 全废、有效并发 105→70，训练降级但未中断（`harbor_fail_on_trial_error: false` 兜住了）。根因是**自检脚本只逐项列包名，漏掉传递依赖**。已把 `verify_node_env.sh` 改为端到端 import 探测，它随即又发现 `litellm` 也缺。教训：新节点必须跑 §6 自检并看到 `harbor_imports=OK`。 |
| 2026-08-12 | 新增 N10 (node-11) / N11 (node-12)：装 fla 0.4.2 + tenacity 9.1.4 + daytona 0.168.0，建 Triton 缓存目录，join Ray。集群 9→11 节点、72→88 GPU。全 11 节点实测环境一致。 |
| 2026-08-11 | 全部 9 个节点 fla 0.4.1 → 0.4.2（启用 hybrid GDN CP）。 |
