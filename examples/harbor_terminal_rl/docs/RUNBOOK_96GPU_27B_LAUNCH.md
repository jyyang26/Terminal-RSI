# Runbook — 启动 96-GPU (12 节点) Qwen3.5-27B fully-async run

记录 2026-06-09 v2 从头重训起跑时踩的三个坑 + 正确的起跑流程。**下次起 96 卡 run 前先读这页。**

集群：server 018（`8B1D80739E921BCD019E9918BD6E0CAE`），12 节点 × 8 GPU = 96 H20。
- 4 actor 节点（TP2×PP8×CP2=32 train GPU）+ 8 rollout 节点（64 GPU）。
- 启动脚本：`start_full89_96gpu_27b_fully_async_tmux.sh`（launcher + 11 worker，各节点起 tmux）。
- 它 exec `run_full89_grpo_96gpu_27b_..._fully_async.sh`（27B wrapper）→ exec `run_qwen35_9b_terminal_bench2_64h20.sh`（共享 base script）。

---

## TL;DR — 一条命令前必须做对的 4 件事

1. **显式传正确网卡的 `MASTER_ADDR`**（node0 的 `29.68.x` IP，不是 `hostname -I` 第一个）。
2. **显式传短 `RAY_TEMP_DIR=/tmp/ray27b96fa`**（否则 AF_UNIX socket 路径超 107 字节，dashboard 崩）。
3. **RUN_TAG 用短的**（如 `v2r4_HHMMSS`），所有 12 节点用**同一个** RUN_TAG。
4. **起跑顺序**：先起 11 worker（keep-alive 等 head，可先起），最后起 launcher。worker 会重试 120×5s=10min 等 head，所以顺序容错，但别让 worker 等超过 10 分钟。

ssh 到其它节点：**直接 `ssh <29.68.x ip>`**（已在 cluster 内，无需 expect/jizhi）。peer IP 从 `NODE_IP_LIST` 环境变量（权威）或上次 launcher log 的 `ip=29.x` 拿。

---

## 坑 1 — `MASTER_ADDR` 取错网卡 → `8/32 clients joined` 超时

**症状**：job 提交后，模型初始化阶段崩：
```
torch.distributed.DistStoreError: Timed out after 601 seconds waiting for clients. 8/32 clients joined.
  at slime/ray/train_actor.py:63  dist.init_process_group()
```
32 个 train rank 只有 8 个（=1 个节点的 8 GPU）join，其余 24 个跨网段连不上 TCPStore master。

**根因**：节点有双网卡（`29.63.x` + `29.68.x`）。starter line 34 `MASTER_ADDR=$(hostname -I | awk '{print $1}')` 取到的第一个 IP 是 **`29.63.x`**，但 Ray 集群内部 + worker join 走的是 **`29.68.x`** 网段（见 `NODE_IP_LIST` 环境变量）。master 选错网卡 → 跨网段 rendezvous 失败。

**为何旧 run 没事**：之前手动起时传了正确的 29.68 master，或机器网卡顺序不同。不传 = 赌 `hostname -I` 顺序。

**修复**：launcher 和 worker 都显式传 `10.0.0.19`（node0 的 29.68 网卡）作为第 2 个位置参数：
```bash
bash start_...tmux.sh launcher 10.0.0.19
bash start_...tmux.sh worker   10.0.0.19   # 每个 worker
```
确认生效：launcher log 应有 `starting Ray head: master=10.0.0.19:6379`（不是 29.63.x）。
快速核错次数：`grep -ac "clients joined" <launcher.log>` 应为 0。

---

## 坑 2 & 3 — Ray dashboard `MetricsHead failed` → job submit `Connection refused :8265`

**症状**：12 节点都 join 了、Ray runtime 起来了，但 job 提交失败：
```
ConnectionError: Failed to connect to Ray at address: http://127.0.0.1:8265
```
往前翻 launcher log 头部会看到：
```
ERROR services.py:1405 -- Failed to start the dashboard
RuntimeError: Module MetricsHead failed to start. Received EOF from pipe.
```

**真正根因**（在 `<RAY_TEMP_DIR>/session_*/logs/dashboard_MetricsHead.err`）：
```
OSError: validate_socket_filename failed: AF_UNIX path length cannot exceed 107 bytes:
/tmp/r27b96fa-v2retrain3_20260609_094832/session_2026-06-09_09-49-49_892711_1304037/sockets/dash_MetricsHead
```
Ray dashboard 子模块用 **AF_UNIX socket，路径硬上限 107 字节**。路径 = `RAY_TEMP_DIR` + `/session_<ts>_<pid>/sockets/dash_MetricsHead`（后缀约 68 字节）。所以 **`RAY_TEMP_DIR` 必须 ≤ ~39 字节**。

**罪魁**：starter line 42 `RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/r27b96fa-${RUN_TAG}}"` 把 RUN_TAG 拼进路径，且 starter 的 ENV_ASSIGNMENTS（line 89）把这个长值**传给 wrapper，覆盖了** wrapper line 113 的短默认值 `/tmp/ray27b96fa`。RUN_TAG 越长越容易爆。

- `v2retrain3_20260609_094832`（长 RUN_TAG）→ `RAY_TEMP_DIR=/tmp/r27b96fa-v2retrain3_20260609_094832`（38 字节）+ 68 ≈ 106~107 → **崩**。
- `run96retok2`（旧短 RUN_TAG）→ 路径短 → 没事。这就是为什么旧 run 一直正常。

**注意**：坑 2 和坑 3 是同一个 bug。第一次（坑 2）误判为"偶发"想重试，第二次重现才定位到 socket 长度。**不要把 dashboard 崩当偶发——去读 `dashboard_MetricsHead.err`。**

**修复**：起跑时对所有 12 节点显式传短 `RAY_TEMP_DIR=/tmp/ray27b96fa` + 用短 RUN_TAG。
快速核：`grep -ac "MetricsHead failed" <launcher.log>` 应为 0。
socket 预算速算：
```python
suffix = "/session_2026-06-09_09-49-49_892711_1304037/sockets/dash_MetricsHead"  # 68 bytes
# RAY_TEMP_DIR 须 <= 107 - 68 = 39 bytes；/tmp/ray27b96fa (15) 安全
```

---

## 附带坑 — ssh 远程命令的 shell snapshot 噪声

直接 `ssh <ip> "<cmd>"` 时，登录会 source Claude 的 shell snapshot，触发 `ZSH_VERSION: unbound variable`（`set -u` 下），exit 127——**但 `tmux new-session -d` 往往已经执行成功**（命令在报错前已 detach）。

**对策**：远程命令统一用 `bash --noprofile --norc -c '...'` 包裹，绕开 snapshot。
```bash
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 <ip> \
  "bash --noprofile --norc -c 'source <envfile>; export RUN_TAG=... RAY_TEMP_DIR=/tmp/ray27b96fa; bash <starter> worker 10.0.0.19'"
```

---

## 附带坑 — 重起前必须清掉旧 worker tmux session

worker 的 tmux session 名固定 `slime_full89_96gpu_27b_fa_worker`。starter 里有 `if tmux has-session ... exit 0`——**同名 session 已存在就直接跳过，不会用新 RUN_TAG 起**。重起时若上次的 worker session 还活着（keep-alive 等旧 master），新 worker 根本没起，表现为 launcher 一直 `1/12 nodes`。

**对策**：重起前对所有 worker `tmux kill-session -t slime_full89_96gpu_27b_fa_worker`，再起。
验证新 worker 真起来了：worker 节点上应出现 `/tmp/slime_full89_96gpu_27b_fa_<新RUN_TAG>_worker.log`。

---

## 正确起跑流程（copy-paste 模板）

```bash
# ---- 0. 参数 ----
RUN_TAG="v2r4_$(date +%H%M%S)"          # 短！
MASTER=10.0.0.19                       # node0 的 29.68 网卡（不是 hostname -I 第一个）
RAYTMP=/tmp/ray27b96fa                    # 短！<=39 bytes
ENVFILE=/path/to/workspace/.slime_run_env/wandb_v2_retrain.env  # 600 权限, repo 外
STARTER=/path/to/workspace/code/slime-v0.3-harbor/examples/harbor_terminal_rl/start_full89_96gpu_27b_fully_async_tmux.sh
WORKERS="10.0.0.9 10.0.0.18 10.0.0.3 10.0.0.15 10.0.0.8 10.0.0.7 10.0.0.14 10.0.0.16 10.0.0.13 10.0.0.4 10.0.0.20"
# 注意：节点 IP 每次分配可能变 —— 从 `echo $NODE_IP_LIST` 环境变量取权威列表（逗号分隔，第一个是 launcher）。

# ---- 1. 清理（每节点：kill 旧 session + ray stop）----
ray stop --force; pkill -9 -f dashboard.py
for ip in $WORKERS; do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ip" \
    "bash --noprofile --norc -c 'tmux kill-session -t slime_full89_96gpu_27b_fa_worker 2>/dev/null; ray stop --force >/dev/null 2>&1; echo ok'"
done

# ---- 2. 起 11 worker（先起，keep-alive 等 head）----
for ip in $WORKERS; do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ip" \
    "bash --noprofile --norc -c 'source $ENVFILE; export RUN_TAG=$RUN_TAG WANDB_MODE=online RAY_TEMP_DIR=$RAYTMP; bash $STARTER worker $MASTER'"
done

# ---- 3. 起 launcher（最后，本机 node0）----
bash --noprofile --norc -c "source $ENVFILE; export RUN_TAG=$RUN_TAG WANDB_MODE=online RAY_TEMP_DIR=$RAYTMP; bash $STARTER launcher $MASTER"
```

W&B：key 放共享盘 600 权限文件 `.slime_run_env/wandb_v2_retrain.env`（repo 外，不进 git/日志），各节点 `source` 后用环境变量透传。或 `WANDB_MODE=offline` 免 key。

---

## 起跑后健康校验（按顺序确认越过每个坑）

```bash
L=<BASE_FOLDER>/logs/launcher_<RUN_TAG>.log
grep -ac "master=10.0.0.19:6379" "$L"      # 坑1：master 网卡对 (>=1)
grep -ac "MetricsHead failed" "$L"            # 坑2/3：dashboard (0=好)
grep -ac "clients joined" "$L"                # 坑1：rendezvous 超时 (0=好)
grep -ac "Job .* failed" "$L"                 # 0=好
grep -aE "kl_loss_coef|use_kl_loss|ref_load" "$L" | grep -avE "HarborTrialWorker|tmux_session" | head  # v2: KL ref 配置生效
# SGLang 已推理：tail 应见 "Prefill batch ... input throughput"
# 首个 rollout dump：起跑后 20-40 min 出现在 <BASE_FOLDER>/rollout_dumps/
```

读 launcher log 一律加 `grep -avE "HarborTrialWorker|tmux_session"` 过滤容器噪声。

---

## 建议的永久修复（让 starter 默认就安全，不靠记得传参）

`start_full89_96gpu_27b_fully_async_tmux.sh` 两处默认值是坑根源，建议改稳：
1. **line 34** `MASTER_ADDR=$(hostname -I | awk '{print $1}')` → 改成优先取 `NODE_IP_LIST` 第一个，或过滤出 `29.68.x` 网段的 IP。
2. **line 42** `RAY_TEMP_DIR=/tmp/r27b96fa-${RUN_TAG}` → 去掉 `${RUN_TAG}`，固定短路径 `/tmp/ray27b96fa`（多 run 并存时再考虑加短 hash，但务必保证总路径 ≤107）。

改完就不用每次记得显式传这俩参数了。
