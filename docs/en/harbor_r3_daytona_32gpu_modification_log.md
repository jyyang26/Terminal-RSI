# Harbor R3 / Daytona 32GPU modification log

记录时间：2026-06-13，Asia/Beijing。

本文件记录当前这轮 R3 实现、Qwen3.5-35B-A3B 32 卡 Daytona 训练调试相关的代码和运行资产改动。这里不记录任何 Daytona/W&B key；`.secret_env.sh` 只记录文件用途，不写入内容。

## 当前状态

- 已停止继续做功能修改。
- 最新一次 full run 已经通过 SGLang 启动、W&B 初始化、TrialWorker 启动、Megatron HF bridge load，并通过了先前的 `vision_model.*` weight skip。
- 最新一次 full run 随后失败在 `update_weights` 参数名映射：
  - `module.module.language_model.decoder.layers.0.self_attention.dt_bias`
  - 失败原因：Qwen3.5 Megatron-to-HF converter 原先只匹配 `module.module.decoder.layers.*`，不匹配 Megatron Bridge 产出的 `module.module.language_model.decoder.layers.*`。
- 失败后已做了本地修复：在 Qwen3.5 converter 入口加入 Bridge 前缀归一化，并补了对应单测。
- 修复后只完成了 `py_compile`；`tests/test_qwen3_5_mtp_bridge_mapping.py` 的 pytest 在用户中断时未跑完。
- 最新修复后还没有重新启动 full run。

## R3 rollout routing replay 改动

### `slime_plugins/harbor/routing_replay.py`

新增 Harbor 侧 R3 routing payload 工具：

- 校验 SGLang 返回的 base64 `routed_experts` payload。
- 将 payload 解码为 `int32` routing expert tensor。
- 校验 token rows、layer 数、top-k 维度。
- 提供 `RoutingRowsBuilder`，按最终训练样本 token 序列拼接 routing rows。
- 对没有真实首 token routing 的拼接边界使用 placeholder row。
- 记录 placeholder 数量，并在 placeholder 会落到 loss token 时直接报错，避免 silent wrong routing。

### `slime_plugins/harbor/llm.py`

给 Harbor LLM 调用增加 rollout routing replay 开关：

- 新增 `use_rollout_routing_replay` 参数。
- 开启后请求 SGLang 返回 routed experts。
- 从 SGLang response 中读取并校验 `routed_experts`。
- 将 routing payload 和 shape 写入 rollout detail 的 extra 信息，供后续 trajectory/retokenized 转换使用。
- 要求 `num_layers` 和 `moe_router_topk` 为正数，否则启动时报错。

### `slime_plugins/harbor/atif_adapter.py`

在 Harbor trajectory/retokenized 样本转换中接入 R3：

- 从每个 LLM turn 的 detail extra 信息中读取 `routed_experts`。
- 在 `trajectory_retokenized` 拼接时同步拼接 routing rows。
- 将拼好的 routing replay 信息写入训练 sample metadata。
- 写入 placeholder 统计：
  - `harbor_routing_replay_placeholder_rows`
  - `harbor_routing_replay_placeholder_loss_rows`
- 在 `require_routing_replay=True` 时，如果缺少 routing payload 或 routing 无法安全对齐，会 fail fast。

### `slime/backends/sglang_utils/sglang_engine.py`

在 SGLang engine 请求路径中接入 R3 开关：

- 当 `args.use_rollout_routing_replay` 开启时，让 rollout generation 请求返回 routing experts。
- 保持默认关闭；关闭时原 rollout/training 路径不变。

### `slime/rollout/sglang_rollout.py`

普通 SGLang rollout 路径增加 R3 参数传递：

- 开启 `--use-rollout-routing-replay` 时把 routing replay 需求传给 Harbor LLM/backend。
- 默认关闭时不改变旧训练行为。

### `slime/rollout/sglang_streaming_rollout.py`

streaming rollout 路径同样增加 R3 参数传递，保证普通/streaming 两条 rollout 路径行为一致。

### `slime/utils/arguments.py`

新增训练参数：

- `--use-rollout-routing-replay`

参数处理逻辑：

- 开启后自动设置底层 `args.use_routing_replay = True`。
- 默认关闭，保证原训练不受影响。

### `tests/test_harbor_routing_replay.py`

新增 R3 对齐测试：

- 覆盖 strict `trajectory_retokenized` 下 routing rows 精确保留。
- 覆盖拼接边界 placeholder 逻辑。
- 覆盖 placeholder 不允许落到 loss token 的保护。

## Qwen3.5 / Megatron Bridge 调试改动

这些改动是为了在 32 卡上绕过当前 torch_dist sharding 与 Qwen3.5 bridge/provider 的启动问题，让 TB2 full run 能进入训练。

### `slime_plugins/mbridge/qwen3_5.py`

已撤回。

原因：

- 当前 32 卡 full run 实际使用 installed `megatron.bridge` provider，不依赖本地 `slime_plugins/mbridge/qwen3_5.py`。
- 如果模型已经提前转成 torch_dist 并从 torch_dist 直接 load，这个本地 mbridge provider 通常也不会参与主线 load。
- 因此该文件不应混入当前 R3/32GPU 实验主线。

当前状态：

- `git diff -- slime_plugins/mbridge/qwen3_5.py` 为空。
- 该文件目前没有本轮未提交改动。

### `slime/utils/megatron_bridge_utils.py`

Megatron Bridge HF config patch：

- 新增 Qwen3.5 config 检测。
- 默认将 Qwen3.5 `mtp_num_hidden_layers` patch 为 `None`，避免 installed `megatron.bridge` provider 构造 MTP 后在 RL actor load 中失败。
- 如果显式设置 `SLIME_ENABLE_QWEN35_MTP=1`，则保留 MTP。
- 这是因为 wrapper 中的 `SLIME_DISABLE_QWEN35_MTP=1` 不一定能可靠传入 Ray runtime，而 Megatron actor 实际使用的是 installed `megatron.bridge` provider，不是本地 `slime_plugins/mbridge` provider。

### `slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py`

Megatron-to-HF export 兼容改动：

- 跳过 `module.module.vision_model.*` 参数。原因：Megatron Bridge Qwen35 provider 会生成 vision side 参数，但当前 text-only SGLang model 不需要这些权重；不跳过会在 `update_weights` 报 unknown parameter。
- 新增 `_normalize_qwen3_5_bridge_name()`：
  - 将 `module.module.language_model.*` 规范化为 `module.module.*`。
  - 处理多余 `module.module.module.` 前缀。
- 该归一化用于修复 full run 中的失败参数名：
  - `module.module.language_model.decoder.layers.0.self_attention.dt_bias`

### `tests/test_megatron_bridge_utils.py`

新增 Megatron Bridge config patch 测试：

- Qwen3.5 MTP 默认禁用。
- 设置 `SLIME_ENABLE_QWEN35_MTP=1` 时保留 MTP。
- 非 Qwen3.5 config 不受影响。

### `tests/test_qwen3_5_mtp_bridge_mapping.py`

新增/扩展 Qwen3.5 bridge/export 测试：

- MTP 参数名带 `language_model.` 前缀时仍能映射。
- MTP disable 开关生效。
- raw Qwen3.5 export 跳过 `vision_model.*`。
- raw Qwen3.5 export 接受 `module.module.language_model.decoder.layers.*` 前缀。

## 32GPU Daytona run assets

### `.r3_daytona/q35a3b_tb2_32gpu_20260612_daytona_r3/harbor_tb2_full89_daytona_r3.yaml`

新增 32 卡 TB2 Daytona runtime config：

- `environment.type: daytona`
- TB2 full89 task glob。
- `harbor_sample_mode: trajectory_retokenized`
- `harbor_group_unit: trajectory`
- `harbor_num_concurrent_trials: 16`
- `harbor_trials_per_node: 4`
- `override_storage_mb: 5120`
- `max_turns: 120`
- `override_timeout_sec: 3600`
- fully async cleanup function 保持 Harbor 路径。

### `.r3_daytona/q35a3b_tb2_32gpu_20260612_daytona_r3/launch_full_3epoch.sh`

新增 32 卡 full run wrapper，没有重写原 96 卡大脚本，而是包装：

- base script：
  - `examples/harbor_terminal_rl/run_full89_grpo_96gpu_35b_a3b_max120_actor4_rollout8_batch8_n32_64k_timeout3600_trajectory_normalized_fully_async.sh`
- 资源划分：
  - `EXPECTED_NODES=4`
  - `ACTOR_NUM_NODES=2`
  - `ACTOR_NUM_GPUS_PER_NODE=8`
  - `ROLLOUT_NUM_GPUS=16`
- 训练并行：
  - TP=1
  - PP=8
  - CP=1
  - EP=2
  - ETP=1
  - sequence parallel off
- rollout：
  - two SGLang engines x 8 GPUs
  - 128k rollout context
  - 64k train sample cap
- 使用 HF bridge initial load：
  - `--megatron-to-hf-mode bridge`
  - `PRETRAINED_CKPT=/tmp/slime-harbor-model-cache/Qwen3.5-35B-A3B`
- 开启 R3：
  - `--use-rollout-routing-replay`
- W&B online，使用本轮 run group。

### `.r3_daytona/q35a3b_tb2_32gpu_20260612_daytona_r3/launch_actor_load_smoke.sh`

新增 actor-load smoke wrapper：

- 用于快速验证 32 卡 Megatron actor 能否从 HF bridge load Qwen3.5-35B-A3B。
- smoke 已经证明 HF bridge actor load 能进入 `train_wait` 和 `update_weights`，但因为 smoke 没有完整 TrialWorkers/rollout 资源，最终会在 Harbor trial worker availability 处失败，这是预期范围。

### `.r3_daytona/q35a3b_tb2_32gpu_20260612_daytona_r3/.secret_env.sh`

新增本地 secret env 文件：

- 存放 Daytona/W&B credentials。
- 文件内容不应提交、不应复制到文档、不应打印。

## 运行验证记录

### 本地验证

- `py_compile` 通过：
  - `slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py`
  - `slime/utils/megatron_bridge_utils.py`
- `tests/test_megatron_bridge_utils.py` 通过：3 passed。
- 远端节点上 `tests/test_megatron_bridge_utils.py tests/test_qwen3_5_mtp_bridge_mapping.py` 曾通过：16 passed。
- 最新加入 `module.module.language_model.decoder.layers.*` 前缀归一化测试后，pytest 被用户中断，尚未完成。

### 32GPU smoke/full run 记录

- torch_dist checkpoint load 原始失败点：
  - `CheckpointingException Invalid access pattern for ShardedTensor`
  - 涉及 hybrid linear attention `linear_attn.*` layer-axis metadata。
- 因此 32 卡实验切到 HF bridge initial load。
- MTP 问题通过 patch HF config 默认禁用 Qwen3.5 MTP 解决。
- vision side weight 问题通过 export 时跳过 `vision_model.*` 解决。
- 最新 full run job：
  - Ray job id: `raysubmit_y6PxJNhy7gxKfd64`
  - W&B run: `https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/vm3iae3d`
  - launcher log: `/tmp/q35_daytona_r3_launcher.log`
  - worker logs:
    - `/tmp/q35_daytona_r3_worker1.log`
    - `/tmp/q35_daytona_r3_worker2.log`
    - `/tmp/q35_daytona_r3_worker3.log`
- 最新 full run 已到 `Timer update_weights start`，随后因 Qwen3.5 converter 前缀不兼容失败。
- 已在本地修复该前缀问题，但尚未重新跑 full run。

### 2026-06-13 torch_dist 重新转换记录

新增转换 wrapper：

- `.r3_daytona/q35a3b_tb2_32gpu_20260612_daytona_r3/convert_qwen35_35b_a3b_torch_dist_tp1pp8ep2_16g.sh`

成功输出：

- `/path/to/models/Qwen3.5-35B-A3B_torch_dist_tp1pp8ep2_16g_socket`
- tracker: `latest_checkpointed_iteration.txt` 内容为 `release`
- size: 65G

转换拓扑：

- 2 nodes x 8 GPUs = 16 ranks
- TP=1
- PP=8
- CP=1
- EP=2
- ETP=1
- `--moe-token-dispatcher-type flex`
- `--moe-enable-deepep`

转换过程记录：

- 直接传训练参数 `--qkv-format thd` 会失败，因为 `tools/convert_hf_to_torch_dist.py` 不接受该参数；已从转换 wrapper 删除。
- 多节点默认 NCCL/IB 在 `save_checkpoint -> determine_global_metadata -> all_gather_object` 处失败：
  - `Call to ibv_reg_mr_iova2 failed with error Invalid argument`
- 设置 socket-only NCCL 后转换成功：
  - `NCCL_IB_DISABLE=1`
  - `NCCL_NET=Socket`
  - `NCCL_SOCKET_IFNAME=bond1`
  - `GLOO_SOCKET_IFNAME=bond1`
- 成功日志出现：
  - `successfully saved checkpoint from iteration 1 ... [ t 1/1, p 1/8 ]`

后续 smoke 状态：

- 已尝试用新 torch_dist 跑 actor-load smoke。
- 该 smoke 尚未进入 Megatron load；Ray head 的 dashboard/job server 没有成功启动，`ray job submit` 连接 `127.0.0.1:8265` 失败。
- 因此目前只能确认“重新转换成功”，尚未确认“新 torch_dist 可被 actor load”。
- smoke 尝试留下的 Ray 进程已清理。

### 2026-06-13 TB2 Daytona torch_dist full run 入口

新增独立入口，未覆盖旧的 `launch_full_3epoch.sh`：

- `.r3_daytona/q35a3b_tb2_32gpu_20260612_daytona_r3/launch_full_tb2_torchdist_b16_os32_timeout1800_max100.sh`
- `.r3_daytona/q35a3b_tb2_32gpu_20260612_daytona_r3/harbor_tb2_full89_daytona_r3_torchdist_b16_os32_timeout1800_max100.yaml`

本次运行参数：

- Daytona environment: `type: daytona`
- actor/rollout: 16 GPU train + 16 GPU rollout
- checkpoint: `/path/to/models/Qwen3.5-35B-A3B_torch_dist_tp1pp8ep2_16g_socket`
- train topology: TP=1, PP=8, CP=1, EP=2, ETP=1
- `ROLLOUT_BATCH_SIZE=16`
- `OVER_SAMPLING_BATCH_SIZE=32`
- `N_SAMPLES_PER_PROMPT=16`
- `GLOBAL_BATCH_SIZE=256`
- `NUM_ROLLOUT=18`
- agent timeout: 1800s
- max turns: 100
- `harbor_num_concurrent_trials=256`
- `harbor_trials_per_node=64`
- socket-only NCCL/Gloo enabled for consistency with successful conversion.

启动修正：

- `RAY_TEMP_DIR` 从较长路径缩短为 `/tmp/r35b16`，否则 Ray plasma socket 路径超过 107 字节，head 无法启动。
- wrapper 增加 `WORKER_KEEP_ALIVE=1` 默认值；否则 worker `ray start` 后进程退出，head 侧只剩 8 GPU，32-GPU placement group 会 pending。

实际 run：

- Ray job id: `raysubmit_cQRRAPgia8SgaeLm`
- W&B run: `https://wandb.ai/yucheng/slime-harbor-terminal-rl/runs/59edcoi1`
- Ray 资源已正常到 32 GPU，placement group 成功分配到 4 个节点。
- SGLang 两个 8-GPU rollout engine 已成功启动并通过 health check。
- Daytona/Harbor trial worker 已启动：
  - node `node-01`, `max_concurrency=64`
  - node `node-02`, `max_concurrency=64`
- Megatron train actor 已启动到 tokenizer/init，并进入 torch_dist checkpoint load。

最终失败点：

- 新转换的 torch_dist 仍然无法被 16-GPU actor topology 加载。
- 失败仍是 hybrid linear-attention layer-axis metadata：
  - `CheckpointingException: Invalid access pattern for ShardedTensor`
  - key: `decoder.layers.self_attention.linear_attn.dt_bias`
  - `global_shape=(40, 32)`
  - `axis_fragmentations=(40, 1)`
  - invalid access pattern contains repeated `[1], [1], [1], [0]`.
- 这说明重新转换为 `TP1 PP8 EP2` 后，torch_dist 仍保留了按 layer axis 切分的 sharded metadata；加载到 16-GPU actor 时 Megatron dist checkpoint validation 仍不能接受该 fragmentation。
- 失败后已停止 Ray/tmux，node0/node3 GPU 确认为 0 MiB，仅保留 `gpu_keeper`。

## 后续建议

恢复工作时建议按这个顺序继续：

1. 跑完最新单测：
   - `PYTHONPYCACHEPREFIX=/tmp/slime_pycache_pytest python -m pytest tests/test_qwen3_5_mtp_bridge_mapping.py -q`
2. 远端重新跑同一组 Qwen3.5 bridge/export tests。
3. 先用新 torch_dist 做 actor-load smoke，确认不再出现 `Invalid access pattern for ShardedTensor`。
4. 如果 actor-load smoke 通过，再把 32GPU full run 的 `PRETRAINED_CKPT` 切到新 torch_dist。
5. 观察是否通过第一次 `update_weights end`，再确认 Daytona rollout 是否开始产生 samples。
6. 如果再遇到 unknown parameter，优先检查是否仍是 Megatron Bridge 额外前缀/vision side 参数，而不是放宽 converter 到模糊匹配。
