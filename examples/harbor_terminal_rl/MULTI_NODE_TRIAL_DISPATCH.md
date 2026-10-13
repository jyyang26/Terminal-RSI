# Harbor Trial 多节点分发 —— 改动说明

> 作者：本次改动 / 日期：2026-05-21
> 目标：让 Harbor singularity rollout trial 自动分散到所有 rollout 节点，而不再
> 全部挤在 RolloutManager 所在的那一台机器。

## 一、为什么做这个改动

### 改之前的现状

`RolloutManager`（`slime/ray/placement_group.py:184`）是一个**唯一**的 Ray actor，被 Ray 调度到
placement group 中的某一个 bundle 上，等价于"挑了一台节点"。在它的进程内，Harbor rollout
（`slime_plugins/harbor/rollout.py`）用 `asyncio.Semaphore` 控制 `harbor_num_concurrent_trials`
个并发 trial。每个 trial 通过 `asyncio.create_subprocess_exec("singularity", ...)`
（`slime_plugins/harbor/singularity_env.py:182`）拉起一个 singularity 容器。

结果：

- Ray 资源账本只看到 RolloutManager 的 `num_cpus=1`。
- 所有 `singularity exec` 子进程都是 fork 自 RolloutManager 的 Python 进程，**都跑在同一台节点**。
- 多节点训练时其他节点的 CPU/IO 全部闲置。
- 单机内的 fakeroot/containall/pid 容器没有 cpuset/memcg 约束，靠 Linux 调度器随机分配，
  容易 OOM-kill / 进程争抢。

### 改之后的效果

- 仍然只有**一个** `RolloutManager`（data source、SGLang router、weight update lock 等所有
  单例不变）。
- `RolloutManager` 启动时会从 placement group 里找出"有多少个不同节点"，每个节点上起
  一个 `TrialWorker` Ray actor。
- Harbor rollout 在派发 trial 时，按 round-robin 把每个 trial 交给一个 `TrialWorker.run_trial.remote(...)`
  执行；singularity 容器就在那个 worker 所在的节点上拉起。
- 每个 worker 的并发上限 = `harbor_trials_per_node`（actor `max_concurrency`）。
- 关闭新行为时（`harbor_trial_node_spread=false`，**默认**）走老路径，行为完全不变。

---

## 二、设计要点

### 为什么不是"多 RolloutManager + 数据分片"

那条路要重做 SGLang router、weight update lock、`train_parallel_config`、data source
分片，改动面大且容易出错。本方案保留 RolloutManager 单例，**只**把 trial 这个"无状态、I/O bound、
跑得起容器就行"的工作单元搬出去。

### 为什么是 long-lived `TrialWorker` actor 而不是 per-trial `@ray.remote` task

- Harbor 包导入（`harbor.trial.trial`、`harbor.environments.singularity.*`）很重，
  per-task 每次都要重新 import。long-lived actor 只 import 一次。
- per-node 并发上限直接用 Ray actor 的 `max_concurrency=K`，权威且简单；per-task 方案
  需要在 manager 端用 asyncio.Semaphore 自己数每个节点几个，麻烦且和 Ray 调度器博弈。
- worker 死了不影响别的 worker；dispatcher 一圈重试到剩下的 worker；都死了再回退到本进程跑。

### 调度细节

- placement group 的每个 bundle 已经有 `(node_ip, gpu_id)` 信息（之前由 `InfoActor.get_ip_and_gpu_id`
  采集）。我们把 `node_ip` 列表暴露出来，然后挑 "每个 IP 第一次出现时的 bundle index"
  作为该节点的代表 bundle。
- `TrialWorker` 用 `PlacementGroupSchedulingStrategy(placement_group=pg,
  placement_group_bundle_index=bundle_idx)` 钉到代表 bundle，从而钉到目标节点。
- `placement_group_capture_child_tasks=False`：worker 内部直接拉子进程而不是 Ray task，
  关掉子任务捕获更省事。

---

## 三、新增的配置参数（YAML）

Harbor 参数走 `--custom-config-path` 指向的 YAML（参考 `harbor_slime_config.yaml`、
`harbor_terminal_bench2_config_gy2_sif.yaml` 等），由 `arguments.py:1813-1819` 处的
`setattr` 注入到 `args`。新增两个 key：

```yaml
# 是否启用多节点 trial 分发；默认 false（保留旧行为）
harbor_trial_node_spread: true

# 每个节点的并发 trial 上限（= 对应 TrialWorker 的 max_concurrency）
harbor_trials_per_node: 4

# TrialWorker 只负责在目标节点启动 Harbor trial，不占 GPU；必须用小 CPU
# 资源量，否则会和同 bundle 上的 SGLangEngine 抢 1 CPU bundle 而 pending。
harbor_trial_worker_num_cpus: 0.05

# worker 启动 / 健康检查保护，避免 Ray actor pending 时 rollout 静默卡住。
harbor_trial_worker_startup_timeout_sec: 30
harbor_trial_worker_healthcheck_timeout_sec: 5

# 单个远程 trial 的兜底超时。Harbor 内部 timeout 仍然是主要保护；
# 这个值主要防止 Ray actor method 排队或异常状态永久不返回。
harbor_trial_remote_timeout_sec: 7200
```

与现有 `harbor_num_concurrent_trials` 的关系：

| 场景 | `harbor_num_concurrent_trials` 的含义 | 节点内并发上限 |
| --- | --- | --- |
| `harbor_trial_node_spread=false`（默认） | 全部 trial 都跑在 manager 节点的全局并发上限（原语义） | 同左 |
| `harbor_trial_node_spread=true` | 全局**在飞**上限（asyncio.Semaphore 兜底） | `harbor_trials_per_node` × 节点数 |

建议在多节点时显式调高 `harbor_num_concurrent_trials`，否则它会限制总吞吐（例如 N 个节点
× per-node 4 但全局只允许 8 在飞，相当于只用 2 个节点的容量）。

> 不需要 argparse 改动：现有 `harbor_*` 参数也都不在 argparse 里，全部走 YAML。

### 2026-05-22 review 后的修正

- `TrialWorker` 不能申请 `num_cpus=1`。placement-group bundle 只有 `CPU: 1`，
  同 bundle 上的 `SGLangEngine` 已经占用约 `0.2 CPU`，所以 worker 申请 1 CPU 会 pending。
  当前默认改为 `harbor_trial_worker_num_cpus=0.05`。
- worker spawn 后会立即调用 `get_node_ip.remote()` 并带
  `harbor_trial_worker_startup_timeout_sec`；启动失败的 worker 会被 kill 并跳过。
- 每次 rollout 前会用 `harbor_trial_worker_healthcheck_timeout_sec` 过滤坏 worker。
- Harbor dispatcher 对远程 `run_trial.remote()` 增加
  `harbor_trial_remote_timeout_sec` 兜底，超时会尝试 cancel Ray task 并换下一个 worker。
- 多节点 spread 时不要把 `trials_dir` 放在 `/tmp`，否则日志/trajectory 分散在各节点本地盘。
  当前 full89 runtime 配置已改到 gy2 共享目录。

---

## 四、改动文件清单

```
M  slime/ray/placement_group.py     # 输出 node_ip 列表 + unique_node_bundles helper
M  slime/ray/rollout.py             # TrialWorker class + spawn 逻辑 + args 注入
M  slime/ray/actor_group.py         # 兼容 pg 4 元组解包（*_）
M  slime_plugins/harbor/rollout.py  # _run_one_with_semaphore 改造为可远程派发
```

下面逐文件展开。

---

### 4.1 `slime/ray/placement_group.py`

#### 改动 1：`_create_placement_group` 多返回一个 `pg_reordered_node_ips`

```python
# 新增一行
pg_reordered_node_ips = [gpu_ids[info[0]][0] for info in sorted_bundle_infos]

# 返回值从 3 元组变 4 元组
return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids, pg_reordered_node_ips
```

`gpu_ids[i]` 之前已经是 `(node_ip, gpu_id)`，所以只是把第一列也暴露出来。

#### 改动 2：新增 `unique_node_bundles()` helper

```python
def unique_node_bundles(reordered_bundle_indices, reordered_node_ips):
    """Return one (bundle_index, node_ip) per distinct node, in reordered order."""
    seen = set()
    out = []
    for bundle_idx, ip in zip(reordered_bundle_indices, reordered_node_ips):
        if ip in seen:
            continue
        seen.add(ip)
        out.append((bundle_idx, ip))
    return out
```

输入 reorder 后的 bundle index 列表 + 平行的 ip 列表；输出"每个节点选一个代表 bundle"
的 `(bundle_idx, node_ip)` 列表。`RolloutManager` 用这个来决定起几个 worker、各钉哪个 bundle。

#### 改动 3：`create_placement_groups` 把 node_ip 列表贯穿到结果字典

```python
# 解包 4 元组
(pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids,
 actor_pg_reordered_node_ips) = _create_placement_group(num_gpus)

# rollout slice 同样切第四个
rollout_pg_reordered_node_ips = actor_pg_reordered_node_ips[rollout_offset:]

result = {
    "actor":   (pg, ..., actor_pg_reordered_node_ips),    # 4 元组
    "rollout": (pg, ..., rollout_pg_reordered_node_ips),  # 4 元组
}
```

### 4.2 `slime/ray/actor_group.py`

只改了一行（兼容 4 元组解包，第四个元素丢弃即可）：

```diff
-        pg, reordered_bundle_indices, _reordered_gpu_ids = pg
+        pg, reordered_bundle_indices, _reordered_gpu_ids, *_ = pg
```

`RayTrainGroup._allocate_gpus_for_actor` 不需要 node IP，所以用 `*_` 吃掉。

### 4.3 `slime/ray/rollout.py`

#### 改动 1：`RolloutServerGroup.start_engines` 解包兼容

```diff
-        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg
+        pg, reordered_bundle_indices, reordered_gpu_ids, *_ = self.pg
```

理由同上：SGLang engine 创建不需要 node IP。

#### 改动 2：新增 `@ray.remote class TrialWorker`

```python
@ray.remote
class TrialWorker:
    """Per-node worker that runs a single Harbor singularity trial.

    Pinned to one placement-group bundle so each `run_trial` call spawns its
    container on the same node. Concurrency is capped by ``max_concurrency``
    set at actor-options time.
    """

    def __init__(self, args):
        configure_logger()
        self.args = args

    async def run_trial(self, rollout_id, group_index, sample_rank, base_sample):
        from slime_plugins.harbor.rollout import _run_one_trial
        node_ip = ray.util.get_node_ip_address()
        logger.info("[TrialWorker %s] running r%s g%s s%s",
                    node_ip, rollout_id, group_index, sample_rank)
        return await _run_one_trial(self.args, rollout_id, group_index, sample_rank, base_sample)

    def get_node_ip(self):
        return ray.util.get_node_ip_address()
```

要点：

- `async def run_trial` —— Ray 支持 async actor 方法，会自动跑在 actor 自己的事件循环里。
  这样 `_run_one_trial` 内部的 `await Trial.create(...)` / `await trial.run()` 都能跑。
- `from slime_plugins.harbor.rollout import _run_one_trial` 是惰性导入：slime 主体不需要
  hard depend on Harbor 插件；只有真用 spread 时才进 worker，进 worker 才 import。
- `configure_logger()` 让 worker 的日志统一格式。

#### 改动 3：`RolloutManager.__init__` 里调用 `_maybe_spawn_trial_workers()`

在原有 `rollout_engine_lock` 初始化之后插入：

```python
self._trial_workers: list = []
self._maybe_spawn_trial_workers()
```

#### 改动 4：新增 `_maybe_spawn_trial_workers()` 方法

```python
def _maybe_spawn_trial_workers(self) -> None:
    if not getattr(self.args, "harbor_trial_node_spread", False):
        return

    from .placement_group import unique_node_bundles

    try:
        _pg, reordered_bundle_indices, _gpu_ids, reordered_node_ips, *_ = self.pg
    except (TypeError, ValueError):
        logger.warning(
            "harbor_trial_node_spread requested but placement group lacks "
            "node-IP info; falling back to in-process trials."
        )
        return

    if not reordered_node_ips:
        return

    node_bundles = unique_node_bundles(reordered_bundle_indices, reordered_node_ips)
    if not node_bundles:
        return

    per_node = int(getattr(self.args, "harbor_trials_per_node", 4) or 4)
    pg_obj = self.pg[0]
    workers = []
    for bundle_idx, node_ip in node_bundles:
        workers.append(
            TrialWorker.options(
                num_cpus=1, num_gpus=0,
                max_concurrency=per_node,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg_obj,
                    placement_group_capture_child_tasks=False,
                    placement_group_bundle_index=bundle_idx,
                ),
            ).remote(self.args)
        )
        logger.info("Spawned TrialWorker on bundle %s (node %s, max_concurrency=%s)",
                    bundle_idx, node_ip, per_node)
    self._trial_workers = workers
    logger.info("Total TrialWorker actors spawned: %d", len(workers))
```

行为：

- 没开 `harbor_trial_node_spread` → 直接返回，`_trial_workers = []`，下游走老路径。
- placement group 是老版本（3 元组）→ 警告后回退。
- 没有可用节点 → 返回空。
- 每个唯一节点起一个 worker，钉到该节点的代表 bundle，`max_concurrency=per_node`。

#### 改动 5：新增 `_attach_trial_workers_to_args()` 方法

```python
def _attach_trial_workers_to_args(self) -> None:
    """Expose worker handles to the harbor rollout function via args."""
    self.args._harbor_trial_workers = list(self._trial_workers)
```

之所以走 `args` 而不是修改 `call_rollout_fn` 签名：rollout 函数签名固定，扩展 args
是最小侵入方式。`list(...)` 是浅拷贝，dispatcher 拿到的是 handle 列表的快照。

#### 改动 6：在两个 `call_rollout_fn` 调用点之前 attach

```python
# eval(self, rollout_id) 内
self._attach_trial_workers_to_args()
result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id,
                         self.data_source, evaluation=True)

# generate(self, rollout_id) 内
self._attach_trial_workers_to_args()
data = call_rollout_fn(self.generate_rollout, self.args, rollout_id,
                       self.data_source, evaluation=False)
```

每次调用都重新写一次，将来支持"动态摘除已死 worker"也只要改这个方法。

#### 改动 7：`dispose()` 里清理 worker

```python
def dispose(self):
    for monitor in self._health_monitors:
        monitor.stop()
    for worker in self._trial_workers:
        try:
            ray.kill(worker, no_restart=True)
        except Exception:
            logger.exception("Failed to kill TrialWorker actor")
    self._trial_workers = []
    logging_utils.finish_tracking(self.args)
```

避免训练结束 / 重启时 worker 残留占着 bundle。

### 4.4 `slime_plugins/harbor/rollout.py`

#### 改动 1：新增 import 和模块级 round-robin 游标

```python
import itertools

# 模块级，跨调用复用；worker 列表长度变了就重置
_worker_cycle: itertools.cycle | None = None
_worker_cycle_len: int = 0
```

#### 改动 2：`_run_one_with_semaphore` 改成"可远程派发 + 兜底"

```python
async def _run_one_with_semaphore(args, rollout_id, group_index, sample_rank,
                                  base_sample, semaphore):
    global _worker_cycle, _worker_cycle_len

    workers = list(getattr(args, "_harbor_trial_workers", None) or [])

    async with semaphore:
        # 没有 worker → 老路径，本进程跑
        if not workers:
            return await _run_one_trial(args, rollout_id, group_index, sample_rank, base_sample)

        # Ray 是 hard dep when workers exist;装不上就老路径
        try:
            import ray
            from ray.exceptions import RayActorError
        except ImportError:
            return await _run_one_trial(args, rollout_id, group_index, sample_rank, base_sample)

        # worker 列表大小变了就重置 cycle
        if _worker_cycle is None or _worker_cycle_len != len(workers):
            _worker_cycle = itertools.cycle(range(len(workers)))
            _worker_cycle_len = len(workers)

        last_error = None
        # 至多扫一圈，避免无限循环
        for _ in range(len(workers)):
            worker = workers[next(_worker_cycle)]
            try:
                ref = worker.run_trial.remote(rollout_id, group_index, sample_rank, base_sample)
                return await asyncio.wrap_future(ref.future())
            except RayActorError as exc:
                last_error = exc
                logger.warning("TrialWorker died for sample %s; trying next worker.",
                               base_sample.index)
                continue

        # 所有 worker 都不可用 → 兜底本进程跑
        logger.warning("All TrialWorkers unavailable (last error: %s); falling back.",
                       last_error)
        return await _run_one_trial(args, rollout_id, group_index, sample_rank, base_sample)
```

要点：

- 外层的 `asyncio.Semaphore(harbor_num_concurrent_trials)` 保留，作为**全局**在飞上限。
- 内层每个 `TrialWorker` 用自己的 `max_concurrency=harbor_trials_per_node` 做节点内上限。
- 异常处理：trial 内部业务异常会经 `await asyncio.wrap_future(...)` 直接抛回，由
  `_run_one_group` 现有的 try/except（rollout.py:170）捕获，逻辑不变；只有 `RayActorError`
  （actor 死了）会触发跨 worker 重试。
- `asyncio.wrap_future(ref.future())`：Ray 的 `ObjectRef.future()` 返回 `concurrent.futures.Future`，
  用 `asyncio.wrap_future` 转成 awaitable，是 ray + asyncio 协作的标准写法。

---

## 五、数据流（开启 spread 后的样子）

```
┌─────────────────────────────────────────────────────────────────────────┐
│ RolloutManager (Ray actor, num_cpus=1, num_gpus=0)                       │
│                                                                          │
│   data_source / SGLang router / weight-update lock 全在这里（唯一）        │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │ generate_rollout / asyncio loop                                  │  │
│   │   asyncio.Semaphore(harbor_num_concurrent_trials)  ← 全局上限     │  │
│   │   for sample: _run_one_with_semaphore(...)                       │  │
│   │     └─ workers[next(cycle)].run_trial.remote(...)                │  │
│   │              │              │              │                    │  │
│   └──────────────┼──────────────┼──────────────┼────────────────────┘  │
└──────────────────┼──────────────┼──────────────┼───────────────────────┘
                   │              │              │
                   ▼              ▼              ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │ TrialWorker 1   │ │ TrialWorker 2   │ │ TrialWorker N   │
   │ node A          │ │ node B          │ │ node Z          │
   │ max_concurrency │ │ max_concurrency │ │ max_concurrency │
   │   = per_node    │ │   = per_node    │ │   = per_node    │
   │                 │ │                 │ │                 │
   │ singularity ... │ │ singularity ... │ │ singularity ... │
   │ singularity ... │ │ singularity ... │ │ singularity ... │
   │ ...             │ │ ...             │ │ ...             │
   └─────────────────┘ └─────────────────┘ └─────────────────┘
```

LLM 调用经过 `args.sglang_router_ip:args.sglang_router_port` 走 cluster-routable HTTP；
trial artifact 写到 `args.harbor_trials_dir`（apdcephfs 共享 FS）；结果对象通过
`samples_from_trial_result` 在内存里序列化回 RolloutManager。所有跨节点路径都不假设 localhost。

---

## 六、对其他能力的兼容

| 能力 | 影响 |
| --- | --- |
| 单节点 / `harbor_trial_node_spread=false` | 完全不变（默认行为） |
| `debug_train_only` / `debug_rollout_only` | placement group 还是 4 元组，spawn 逻辑会查 `harbor_trial_node_spread`，不开就 noop |
| `--colocate` | rollout bundle 切片在 colocate 时就是整组 bundle，spread 自动覆盖所有训练节点 |
| async / colocate rollout 模式 | 互不干涉，本改动只动 trial 派发层 |
| SGLang router、weight update | 单例不变；trial worker 不持有 router/engine handle |
| `harbor_require_full_rollout_groups` / retry / overflow trim | 全部由 `_run_one_group` 主循环管理，与派发位置无关 |
| `rollout_sample_filter_path` / `rollout_all_samples_process_path` | 在 RolloutManager 进程里运行，看到合并后的 output_groups，与派发位置无关 |
| 单元测试 `tests/test_harbor_*` | 不依赖 Ray，已全部通过（11/11） |

---

## 七、验证步骤

1. **配置**：在用到的 YAML（例如 `harbor_terminal_bench2_full89_gy2_runtime_sif_last_turn.yaml`）里加：
   ```yaml
   harbor_trial_node_spread: true
   harbor_trials_per_node: 4
   harbor_num_concurrent_trials: 32   # 多节点时记得调高
   ```

2. **看 RolloutManager 日志**：启动阶段应当能看到
   ```
   Spawned TrialWorker on bundle 0 (node 10.x.x.A, max_concurrency=4)
   Spawned TrialWorker on bundle 8 (node 10.x.x.B, max_concurrency=4)
   ...
   Total TrialWorker actors spawned: N
   ```
   `N` 应等于 rollout placement group 覆盖的不同节点数。

3. **看 trial 日志**：rollout 期间应当能看到
   ```
   [TrialWorker 10.x.x.A] running r0 g0 s0
   [TrialWorker 10.x.x.B] running r0 g0 s1
   [TrialWorker 10.x.x.C] running r0 g0 s2
   ...
   ```
   不同 IP 交替出现。

4. **看每个节点的 singularity 进程**：
   ```bash
   for n in 0 1 2 3; do
     /remote-exec 013:$n "ps -ef | grep 'singularity exec' | grep -v grep | wc -l"
   done
   ```
   各节点都应有 ≤ `harbor_trials_per_node` 个 singularity 进程；多个节点同时 > 0。

5. **验证单例不变**：日志里 "Starting rollout server" / "router on port" / data_source
   初始化只出现一次。

6. **故障注入（可选）**：手动 `ray.kill` 掉一个 TrialWorker，看 rollout 是否继续；
   warning 行应为 `TrialWorker died for sample N; trying next worker.`，吞吐略降但
   不会卡死。

---

## 八、后续可能的改进方向

- **死 worker 自动重生**：当前如果一个 TrialWorker 死了，dispatcher 会绕过它直到下次
  RolloutManager 重启；可以在 `_attach_trial_workers_to_args` 前做存活检查并补一个新 actor。
- **超时**：`worker.run_trial.remote(...)` 没有强制超时；建议在 trial 配置里设 Harbor
  本身的 timeout，或者在 dispatcher 里加 `asyncio.wait_for(...)`。
- **更细的 CPU 配额**：当前 `TrialWorker(num_cpus=1)` 只是占位，singularity 子进程的
  CPU 用量仍由 OS 调度；如需要硬限可在 `singularity exec` 加 `--cpus N` 参数。
- **节点选择策略**：当前是 round-robin；如果发现某些节点持续慢，可改成
  "least-loaded"（看 worker 的 in-flight 计数选最小的）。
- **Argparse 注册**：当前两个新参数只走 YAML；如果以后希望命令行也能覆盖，再补
  `arguments.py` 里的 `add_argument`。

---

## 九、相关文件 / 行号速查

| 文件 | 函数 / 位置 | 作用 |
| --- | --- | --- |
| `slime/ray/placement_group.py:78` | `_create_placement_group` 返回值 | 多输出 node_ip 列表 |
| `slime/ray/placement_group.py:81` | `unique_node_bundles` | 每节点选代表 bundle |
| `slime/ray/placement_group.py:115` | `create_placement_groups` | 把 node_ip 列表填进 `result["rollout"]` |
| `slime/ray/actor_group.py:51` | `_allocate_gpus_for_actor` | 4 元组解包兼容 |
| `slime/ray/rollout.py:89` | `RolloutServerGroup.start_engines` | 4 元组解包兼容 |
| `slime/ray/rollout.py:349-376` | `TrialWorker` | 新增 actor 类 |
| `slime/ray/rollout.py:415-416` | `RolloutManager.__init__` | 调 `_maybe_spawn_trial_workers` |
| `slime/ray/rollout.py:427-472` | `_maybe_spawn_trial_workers` | 起 worker、钉 bundle |
| `slime/ray/rollout.py:474-481` | `_attach_trial_workers_to_args` | 注入 handle 给 harbor rollout |
| `slime/ray/rollout.py:eval/generate` | `eval` / `generate` | 调用前 attach |
| `slime/ray/rollout.py:dispose` | `dispose` | `ray.kill` worker |
| `slime_plugins/harbor/rollout.py:19-23` | 模块级 import 和游标 | `_worker_cycle` 状态 |
| `slime_plugins/harbor/rollout.py:221-265` | `_run_one_with_semaphore` | 派发 + 兜底 |
