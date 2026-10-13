# Runbook — 从干净容器到跑起 96-GPU 27B run（docker 镜像更新后重装环境）

记录 2026-06-11 一次 docker 镜像 bump 后，从干净容器把 96-GPU Qwen3.5-27B Harbor fully-async run 重新跑起来踩的坑、根因、修复，以及一个可复用的一键安装脚本。**下次镜像更新后重装环境前先读这页。**

配套脚本：`setup_fresh_container_env.sh`（一键装 + 自检，支持 `fanout` 全集群）。
相关已有脚本：`install_apptainer_ubuntu.sh`、`install_harbor_runtime_deps.sh`。
起跑流程另见：`docs/RUNBOOK_96GPU_27B_LAUNCH.md`（master 网卡 / RAY_TEMP_DIR / worker tmux 三个起跑坑）。

---

## TL;DR — 镜像更新后只需做两件事，然后起跑

slimerl 镜像**已经烤进了重型栈**（torch+cu129 / sglang / megatron / slime / flash-attn / TE / apex / mbridge / fla / sglang_router+slime / torch_memory_saver / sgl-deep-gemm）。**不需要重跑 `build_conda.sh`。**

镜像**没有**的、每次都要在每个节点补的，只有两层 runtime：

1. **apptainer / singularity** —— Harbor 每个 terminal 任务都在 SIF sandbox 里跑。镜像不带。
2. **Harbor 的 7 个 python runtime 依赖** —— `tenacity litellm shortuuid toml dirhash claude-agent-sdk supabase`，外加把 `openai` pin 回 `2.6.1`（SGLang 镜像要求）。

一条命令搞定全集群（在 launcher 节点、cluster 内）：

```bash
cd /path/to/workspace/code/slime-v0.3-harbor/examples/harbor_terminal_rl
bash setup_fresh_container_env.sh fanout      # 自动 ssh 到 $NODE_IP_LIST 每个节点装 + 自检
```

跑完每个节点应打印 `NODE READY ✅`。然后照 `RUNBOOK_96GPU_27B_LAUNCH.md` 起跑。

---

## 为什么"重装环境"≠ 重跑 build_conda.sh

这是最容易判断错的地方。直觉是"镜像换了，环境全没了，重跑 build_conda.sh"。**错。**

- `build_conda.sh` 装的是重型 CUDA/训练栈，这些 **slimerl 镜像本身就带**（镜像就是 FROM slimerl/sglang 一路 build 出来的）。在干净容器里 `python3 -c "import torch, sglang, megatron, slime"` 直接通。重跑 build_conda 是几十分钟的浪费，还可能把镜像里调好的 +cu129 native 轮子搞乱。
- 真正每次镜像 bump 会丢的，是**两层从来就没进镜像、一直靠 per-node runtime 安装的东西**：apptainer 和 Harbor 的轻量 python 依赖。`docker/Dockerfile` 里 grep 不到 `apptainer` / `tenacity` / `litellm`，证明它们从设计上就在镜像外。

**判断方法**：进容器先跑 `python3 -c "import torch,sglang,megatron,slime"`。通 → 只需补那两层（本 runbook）。不通 → 那才可能需要 build_conda.sh（或者根本不是 slimerl 镜像）。`setup_fresh_container_env.sh` 第 0 步会自动做这个判断。

---

## 坑 A — Harbor SIF 任务起不来：缺 apptainer/singularity

**症状**：rollout 阶段 agent 无法在容器里执行任务。`which singularity apptainer` 全空。

**根因**：新镜像不带 apptainer。它一直是 per-node 用 `install_apptainer_ubuntu.sh` 装的（apt + PPA，走 star-proxy），不在 docker 镜像里。

**修复**：每个节点跑 `install_apptainer_ubuntu.sh`。它会装 `uidmap/fuse/squashfuse`、给 root 配 subuid/subgid、清理 `/etc/ld.so.preload` 里的 `libtdsp` 残留（否则 singularity 会把它当 bind mount 失败）。装完 `apptainer --version` 应为 1.5.1。

**自检**：用 Harbor 的精确 no-mount flags 跑一次 SIF：
```bash
SIF=$(ls /path/to/workspace/harbor_sif_cache_tb2_full_harbor_runtime/*.sif | head -1)
cd /tmp && singularity exec --no-mount home --no-mount tmp --no-mount bind-paths \
  --no-mount proc --writable-tmpfs --containall --pid "$SIF" echo smoke_ok
```
（`WARNING: Error getting executable path ... readlink /proc/self/exe` 是无害的，旧 run 也有。）

---

## 坑 B — rollout 0/8 groups，trial worker `ModuleNotFoundError: No module named 'tenacity'`

这是这次最关键、最耗时的坑，也最能说明"重装环境"的真正含义。

**症状**：SGLang 全部健康起来了（`/model_info` 200、`/metrics` 200、CUDA graph capture 完成），但 job 在 rollout 阶段崩：
```
RuntimeError: Harbor fully-async rollout could not produce enough complete trainable groups:
0/8 groups after 64 completed group tasks.
```
64 个 task group 跑完，0 个变成可训练 group —— 即**所有 trial 都失败了**。

**根因**：往上翻 launcher log 找 `Harbor trial failed for sample index`，真正的异常是：
```
ray::HarborTrialWorker.run_trial()
  File ".../harbor/src/harbor/trial/trial.py", line 15, in <module>
    from tenacity import (...)
ModuleNotFoundError: No module named 'tenacity'
```
新镜像缺 Harbor 的 python 依赖。`harbor/pyproject.toml` 声明了 22 个依赖，对照当前环境，缺了 **7 个**：`shortuuid toml tenacity litellm dirhash claude-agent-sdk supabase`。`tenacity` 只是 `trial.py` 第一个 import 到的，所以第一个炸。

**修复**：每个节点跑 `install_harbor_runtime_deps.sh`。它装这 7 个 + `openai==2.6.1`。

**关键细节 — openai 版本**：这个脚本会把 `openai` 从镜像里的版本**降到 2.6.1**（SGLang 镜像的硬要求）。pip 会打印冲突警告：
```
litellm x.y requires openai>=2.20.0, but you have openai 2.6.1 which is incompatible.
openai-agents ... requires openai>=2.36.0 ...
```
**这些警告是预期的、无害的** —— slime_sglang 路径不用 litellm 发模型请求（它直接打 SGLang server）。`install_harbor_runtime_deps.sh` 注释里写明了这一点。不要"修"这个 openai 版本，否则 SGLang 会出问题。

---

## 坑 C — `harbor` 源码树没有包元数据：`PackageNotFoundError: No package metadata was found for harbor`

修完坑 B 后**会撞上的下一个**坑（如果你绕过兼容层手动 import 的话）。

**症状**：
```
File ".../harbor/src/harbor/__init__.py", line 83
    __version__ = importlib.metadata.version("harbor")
importlib.metadata.PackageNotFoundError: No package metadata was found for harbor
```

**根因**：`harbor` 是靠 `PYTHONPATH=.../harbor/src` 以**源码**方式用的，没 `pip install`，所以没有 dist 元数据；而 `harbor/__init__.py` 调了 `importlib.metadata.version("harbor")`。

**已有的修复（无需 pip install -e harbor）**：仓库里有兼容层 `slime_plugins/harbor/import_utils.py` 的 `ensure_harbor_distribution_metadata()`，它 monkeypatch `importlib.metadata.version`，对 `harbor` 返回 fallback 版本 `0+slime-src`。`slime_plugins/harbor/rollout.py`（trial worker 实际走的入口）在 import `harbor.trial` 前会调用它，所以**正常运行路径不会触发这个坑**。

> 历史备注：早期文档（`SERVER_SETUP_PITFALLS.md`）写的是 `pip install -e harbor` 然后恢复 `openai==2.6.1`。后来引入了 `import_utils.py` 兼容层来避免装包。**现在不需要 `pip install -e harbor`** —— 兼容层够了。只要你的自检走的是真实 import 链（先 `import slime_plugins.harbor.rollout`），就不会看到这个错。

**自检（决定性的一条）**——复现 trial worker 的真实 import 链：
```bash
PYTHONPATH=/path/to/workspace/code/harbor/src:/path/to/workspace/code/slime-v0.3-harbor \
python3 -c 'import slime_plugins.harbor.rollout; \
from slime_plugins.harbor.import_utils import ensure_harbor_distribution_metadata as e; e(); \
import harbor.trial.trial; \
from harbor.environments.singularity.singularity import SingularityEnvironment; \
import importlib.metadata as m; print("OK harbor="+m.version("harbor"))'
# 期望: OK harbor=0+slime-src
```
`setup_fresh_container_env.sh` 第 3c 步就是这个检查（写成临时 .py 跑，避免 ssh 套引号被吞）。

---

## 起跑前检查清单（preflight）

环境装好后、`bash start_..._tmux.sh launcher` 之前，逐项确认：

| # | 检查 | 命令 / 期望 |
|---|------|------------|
| 1 | 重型栈在镜像里 | `python3 -c "import torch,sglang,megatron,slime"` 通 |
| 2 | apptainer（**每个节点**）| `apptainer --version` → 1.5.1 |
| 3 | SIF smoke test（**每个节点**）| 见坑 A，输出 `smoke_ok` |
| 4 | trial import 链（**每个节点**）| 见坑 C，输出 `OK harbor=0+slime-src` |
| 5 | 27B checkpoint 在共享盘 | `ls -d /path/to/models/Qwen3.5-27B{,_torch_dist_pp8_timeout60}` |
| 6 | SIF cache 完整 | `ls .../harbor_sif_cache_tb2_full_harbor_runtime | wc -l` → 108 |
| 7 | W&B key 文件存在 | `ls -l .../.slime_run_env/wandb_v2_retrain.env`（600 权限，repo 外）|
| 8 | MASTER 网卡 = `$NODE_IP_LIST` 第一个 | 见 `RUNBOOK_96GPU_27B_LAUNCH.md` 坑1（29.68.x，不是 hostname -I 第一个）|
| 9 | RAY_TEMP_DIR 短 | `/tmp/ray27b96fa`（≤39 字节，见坑2/3）|
| 10 | 旧 worker tmux 已清 | 每节点 `tmux kill-session -t slime_full89_96gpu_27b_fa_worker` |

1–4 步直接用 `bash setup_fresh_container_env.sh fanout` 一次性装 + 全集群自检。

**节点 IP 会变**：每次分配的 IP 不同，权威列表是 `$NODE_IP_LIST`（逗号分隔 `ip:gpus`，第一个是 launcher）。本次为例 master = `10.0.0.8`。

---

## 起跑时要注意什么（这次踩到的运行期注意点）

- **27B checkpoint staging 是最长的一步**：HF(52G)+torch_dist(51G) 共 ~103G/节点，从共享盘 rsync 到 `/tmp`，约 30–50 分钟，期间 launcher log 停在 `CLEAN_START` 不动是正常的（不是卡死）。`STAGE_*_TO_TMP=1` 时 12 节点并行 staging。**第二次起跑会秒过**（`.slime_stage_complete` 标记在，跳过 re-stage）。失败重启时不要删 `/tmp/slime-harbor-model-cache`。
- **从 cluster 内 ssh 其它节点**：直接 `ssh <29.68.x ip>`，无需 expect/jizhi。远程命令一律包 `bash --noprofile --norc -c '...'`，绕开 Claude shell snapshot 在 `set -u` 下的 `ZSH_VERSION: unbound variable`（exit 127，但 tmux 往往已 detach 成功）。
- **后台起 launcher 的坑**：用 `Bash(... bash -c "...launcher..." )` 这种把启动塞进一次性子 shell、命令本身又立即返回的方式，子 shell 可能在 tmux detach 前被回收，导致 launcher 没起来（无 log、无 base dir、ray 0 节点）。**对策**：在前台 shell 里 `source envfile; export ...; bash $STARTER launcher $MASTER`（starter 内部自己 `tmux new-session -d` detach），确认打印出 `started slime_..._launcher` + base/log 路径。
- **失败后重启前务必清干净**：launcher 失败会写 stop marker，11 个 worker keep-alive 会随之退出 → 重启要重起全部 worker。每节点先 `tmux kill-session -t ..._worker` + `ray stop --force`，否则同名 worker session 还在会被 starter 直接跳过（表现为 launcher 一直 `1/12 nodes`）。
- **健康校验顺序**（读 log 一律 `grep -avE "HarborTrialWorker|tmux_session"` 过噪声）：
  ```bash
  L=<BASE>/logs/launcher_<RUN_TAG>.log
  grep -ac "master=<MASTER>:6379" "$L"             # 坑(网卡)  >=1
  grep -ac "MetricsHead failed" "$L"                # dashboard  0
  grep -ac "clients joined" "$L"                    # rendezvous 0
  grep -ac "RayTaskError(ImportError)" "$L"         # 坑B        0
  grep -ac "could not produce enough" "$L"          # 坑B(group) 0
  grep -ac "Server process terminated unexpectedly" "$L"  # SGLang init 0
  grep -ac "POST /generate" "$L"                    # rollout 真在跑  >0
  ```
  最终里程碑：起跑后 ~20–40 min，`<BASE>/rollout_dumps/` 出现第一个 `.pt`，训练步开始。

---

## 附录 — 本次失败时间线（供下次对照）

| 尝试 | RUN_TAG | 结果 | 根因 |
|------|---------|------|------|
| 1 | newdkr_122540 | SGLang init 阶段一个 engine（137.182）`Server process terminated` | 当时误判为节点瞬时慢；实际后续重启正常，非硬件问题 |
| 2 | newdkr2_140703 | SGLang 全起，rollout `0/8 groups` | **缺 apptainer + 缺 Harbor 7 依赖**（坑 A/B）|
| 3 | newdkr3_010451 | 装好 apptainer + harbor deps 后，rollout 正常（`POST /generate` 持续、trajectory dump、trial 无失败）✅ | — |

环境补全清单（全 12 节点）：apptainer 1.5.1 ✓ / `tenacity litellm shortuuid toml dirhash claude-agent-sdk supabase` + `openai==2.6.1` ✓ / trial import 链 `0+slime-src` ✓。

本次对比基线：`v2r5_130222`（W&B `ur20fyj0`，23 dumps / iter_0000023 / 跑满 NUM_ROLLOUT=24 ≈ 2 epoch）。本次 ablation：`USE_ROLLOUT_LOGPROBS=0`（基线为 1），故对比口径是"关闭 rollout-logprobs 的影响"。

---

## 已知无害现象（看到别慌，不是 bug）

下面这些是看 log / W&B 时容易被吓到、但实际无害、且**基线 v2r5 同样存在**的现象。判断一个异常值得不得查的第一步：先看基线那条 run 是不是也长这样——是，则是这套 harbor 架构的固有特征，不是本次回归。

### 1. `rollout/prefix_cache_hit_rate` 与 `rollout/avg_cached_tokens_per_sample` 恒为 0

**结论：纯粹是指标埋点缺失，不影响训练正确性，也不代表 GPU 没在复用 prefix cache。无需修。**

- **根因**：这俩指标的填充路径是 slime 核心的 `sglang_rollout.py → Sample.update_from_meta_info() → prefix_cache_info.add()`（`slime/utils/types.py:198`，从 `meta_info["cached_tokens"]` 累加）。但 Harbor 用 `--custom-generate-function-path` 完全替换了 rollout，走自己的客户端 `slime_plugins/harbor/llm.py`——它自己 POST `/generate`、自己解析 `meta_info`，**从不调用 `update_from_meta_info()`，也从不碰 `prefix_cache_info`**。更直接的铁证：`llm.py:180` 把 `cache_tokens=0` **硬编码**写死，压根没读 SGLang 返回的 `cached_tokens`。于是 `_compute_prefix_cache_metrics()`（`slime/ray/rollout.py:1597`）对所有 sample 求和永远得 0。
- **旁证**：(a) SGLang 服务端 radix cache 其实是开着的（server_args `disable_radix_cache=False`、`radix_eviction_policy='lru'`、`page_size=1`），prefix 复用机制正常工作，只是没人统计；(b) `enable_cache_report=False`（默认），即使去读 `meta_info` 也没有 `cached_tokens` 字段；(c) 基线 v2r5 同样恒为 0、同样 `enable_cache_report=False`——这指标在 harbor 路径下**从来没非零过**，不是 docker 变更或关 logprob 引入的。
- **实际上 prefix 复用率很高**：harbor 是 multi-turn terminal agent，每轮 prompt 在上一轮基础上追加（prefix-extension），KV 复用其实很充分——这也是为什么 KV usage 低、decode 还快。只是这个 wandb 数字没记录它。
- **如果非要点亮（不建议）**：需改两处（都在 plugin 内，不动核心）——启动加 `--sglang-enable-cache-report`，并改 `llm.py:180` 为 `cache_tokens=meta_info.get("cached_tokens", 0)` + 在 `atif_adapter.py` 写进 `Sample.prefix_cache_info`。改了会让本次与基线对比口径不一致，且不改变任何训练行为，故建议挂 backlog 即可。

### 2. rollout GPU 平均利用率很低（估计 SM ~15–25%）——是 terminal-agent RL 的预期形态，不是配置问题

**结论：64 张 rollout 卡严重欠载，但瓶颈在 agent/容器侧而非 GPU；基线同形态，新 docker 没有改变利用率特征。**

- **重要前提**：任务结束后 launcher 退出，事后 `nvidia-smi` 看到的不是 rollout 真实负载（要么空闲、要么 gpu_keeper 占着保号）。下面数字全部从 launcher log 回放 SGLang 引擎活动得来。slime perf 字段里**没有直接的 SM util%**，只能用三个代理量推断：每引擎并发 `#running-req`、KV 占用 `full token usage`、decode 吞吐 `gen throughput`。
- **并发请求数 `#running-req`（64 卡 = 32 个 TP2 引擎，每条快照是单引擎瞬时值）**——本次 vs 基线几乎一致：

  | running-req | 本次 newdkr3 | 基线 v2r5 |
  |---|---|---|
  | 0（完全空闲） | 27.0% | 27.1% |
  | 1（单请求 decode） | 56.5% | 58.6% |
  | 2 | 13.4% | 12.6% |
  | ≥3 | 3.1% | 1.7% |

  → 加权平均并发 ≈ 0.95 req/engine，~27% 时间引擎完全空转。
- **KV cache 占用率 `full token usage`**：本次全程中位数 ~0.05、峰值才 0.49，近 30% 快照 ≤0.03——KV 池常年只用个位数百分比，显存远未吃满。
- **decode 吞吐 `gen throughput`**：中位数 ~78 tok/s、p90 ~179、峰值 1065（峰值=偶尔某引擎攒到 10+ 并发）。单 H20 跑 27B 单请求 memory-bound decode ~78 tok/s 属正常。
- **为什么是预期形态**：harbor 节奏 = 模型生成一段（GPU 忙几秒）→ 在 singularity 容器里执行命令/等 shell 输出/等 timeout（CPU/IO 忙几十秒到几分钟）→ 再生成下一段。GPU 大量时间在等容器，故 27% 落在 0。`group_latency_mean ≈ 2000–3000s` 而真正 GPU 生成只占其中一小部分，也印证了这点。
- **若想提高 rollout GPU 利用率（会破坏"与基线同配置对比"，慎用）**：(a) 加大 `fully_async_max_concurrent_groups`（如 8→16/24），让更多 trial 同时在飞，用别的 group 的生成填上 GPU 等容器的间隙，性价比最高且不影响 RL 正确性；(b) 减少 rollout 卡、把卡挪给训练（64 卡只用 ~20%，32 卡也能扛当前并发）。对比期间建议先别动。
