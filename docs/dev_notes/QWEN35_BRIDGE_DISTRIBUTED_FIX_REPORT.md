# Qwen3.5-27B Bridge Distributed Sync Fix Report

Date: 2026-06-19

Remote repo patched for validation:

- `/root/slime`

Shared running repo not touched:

- `/path/to/workspace/code/slime-v0.3-harbor`

## Files Changed On Remote

This run changed these files under `/root/slime`:

- `slime/backends/megatron_utils/actor.py`
- `slime/backends/megatron_utils/update_weight/update_weight_from_distributed_bridge.py`
- `slime/utils/megatron_bridge_utils.py`

Pre-existing dirty files that I did not modify in this run:

- `slime_plugins/models/qwen3_5.py`
- `tools/convert_hf_to_torch_dist.py`
- `slime_plugins/models/qwen3_5.py.bak`

Patch artifact:

- `/tmp/qwen35_bridge_distributed_patch.diff`

## What Changed

### 1. Bridge fully_async now uses a bridge-aware updater

`actor.py` now selects `UpdateWeightFromDistributedBridge` when:

- `not args.colocate`
- `args.megatron_to_hf_mode == "bridge"`
- `args.update_weight_mode != "delta"`

`bridge + delta` now fails fast with a clear error because delta sync has not been adapted or validated for bridge.

### 2. New distributed bridge updater

Added:

- `slime/backends/megatron_utils/update_weight/update_weight_from_distributed_bridge.py`

It uses:

- `HfWeightIteratorBase.create(...)`, so bridge mode uses `HfWeightIteratorBridge`
- `named_params_and_buffers(..., convert_to_global_name=False)`, matching bridge iterator expectations
- one global distributed source rank: `DP=0, TP=0, PP=0`
- group name: `slime`
- existing `connect_rollout_engines_from_distributed(...)`
- existing `update_weights_from_distributed(...)`

It does not use the raw `convert_to_hf` path.

### 3. Qwen3.5 MTP is disabled for Megatron Bridge by default

`slime/utils/megatron_bridge_utils.py` now disables `mtp_num_hidden_layers` for Qwen3.5 configs unless:

- `SLIME_ENABLE_QWEN35_MTP=true`

This matches the earlier successful bridge load/export validation. MTP is still not supported by this fix; this is explicitly only to make the current Megatron Bridge path work for the non-MTP model body and vision tower.

## Why This Fix Works

The previous fully_async path always selected `UpdateWeightFromDistributed`, which hardcoded raw `convert_to_hf`. That raw converter expects raw Megatron names like:

- `self_attention.linear_attn.dt_bias`

But bridge model names look like:

- `self_attention.dt_bias`
- `self_attention.in_proj.weight`
- `self_attention.out_norm.weight`

So the old path failed with `Unknown parameter name`.

The new updater bypasses raw conversion entirely. Megatron Bridge itself exports HF tensors directly, and the sender only forwards those HF chunks to SGLang.

The important detail is that bridge iterator emits a complete HF model on every Megatron rank for this Qwen3.5 TP4/PP2 layout. Therefore the distributed sender must send once from global rank0, not once from each PP source rank. The new updater uses that single-source layout.

## Validation Completed

### Static checks

Ran:

- `python3 -m py_compile slime/utils/megatron_bridge_utils.py`
- `python3 -m py_compile slime/backends/megatron_utils/update_weight/update_weight_from_distributed_bridge.py`
- `python3 -m py_compile slime/backends/megatron_utils/actor.py`

Result: passed.

### 8-GPU updater dry-run on Qwen3.5-27B

Artifact:

- `/tmp/qwen35_bridge_updater_dryrun.py`
- `/tmp/qwen35_bridge_updater_dryrun_summary.json`
- `/root/qwen35_bridge_updater_dryrun.log`

This script loaded `/root/Qwen3.5-27B` with Megatron Bridge and called the new updater's `update_weights()` path. The actual SGLang send was monkeypatched to record what would be sent.

Result:

- source ranks: `[0]`
- total chunks: `53`
- total sent entries: `1184`
- unique sent tensors: `1184`
- duplicate sent tensors: `0`
- group counts: `language=851`, `vision=333`
- raw/Megatron-name sample: `[]`
- weight version advanced to `1`

This confirms the new updater sends exactly one full HF model from rank0, with vision included, and does not send raw Megatron names.

### Real distributed sender smoke test with fake Ray rollout engine

Artifact:

- `/tmp/test_distributed_sender_fake_engine.py`
- `/tmp/test_distributed_sender_fake_engine_summary.json`
- `/root/test_distributed_sender_fake_engine.log`

This used the real slime functions:

- `connect_rollout_engines_from_distributed(...)`
- `update_weights_from_distributed(...)`
- `disconnect_rollout_engines_from_distributed(...)`

It sent two real CUDA tensors through NCCL to a fake Ray rollout engine on another GPU.

Result:

- `ok: true`
- received shape/dtype/sum matched expected for:
  - `model.language_model.layers.0.linear_attn.dt_bias`
  - `model.visual.pos_embed.weight`

This validates the actual distributed sender path used by the new updater.

### 96-GPU real fully_async bridge smoke validation

Date: 2026-06-20

Remote job:

- `self_improving_terminal_agent_rl_Qwen3_5_slime_0_3_0_sglang_0_5_12_96GPU`
- task id: `8B1D80E79EB0707F019EB4A7D28A0A86`

Patched runtime repo:

- `/root/slime` on all 12 nodes

Launch shape:

- 32 Megatron actor GPUs
- 64 SGLang rollout GPUs
- `--megatron-to-hf-mode bridge`
- `--hf-checkpoint /tmp/slime-harbor-model-cache/Qwen3.5-27B`
- no `--ref-load`
- `--calculate-per-token-loss`

Important log:

- `/tmp/q35_bridge96_launcher_20260620_012800.log`

Result:

- SGLang loaded the Qwen3.5-27B HF checkpoint and all rollout engines reached healthy state.
- Megatron Bridge loaded the HF checkpoint into the actor side.
- The first fully_async distributed weight update completed through the new bridge updater.
- Log evidence:
  - `[slime] Update bridge weights` advanced from `0it` to `110it`.
  - SGLang returned `POST /update_weights_from_distributed HTTP/1.1 200 OK` repeatedly across the rollout engines.
  - Reverse grep found no `Unknown parameter`, `ValueError`, or `AssertionError` in the launcher log.

This validates the original failure point: bridge mode now reaches and completes real Megatron-to-SGLang distributed weight sync for Qwen3.5-27B instead of falling into the raw `convert_to_hf` path and failing on names like `self_attention.dt_bias`.

The run then failed after weight sync, during Harbor rollout generation:

- `RuntimeError: Harbor trial node-spread is enabled but no TrialWorkers are available, and in-process fallback is disabled.`
- Final top-level failure: `Harbor fully-async rollout could not produce enough complete trainable groups: 0/1 groups after 192 completed group tasks.`

That failure is outside the Megatron Bridge weight-sync path. It means this launch did not validate a complete train step after rollout data production.

## Not Fully Validated Yet

The 96-GPU job validated the bridge-mode distributed update path against real SGLang rollout engines, but it did not complete a full Harbor train iteration because rollout auxiliary TrialWorkers were not available in that launch.

What is validated:

- bridge load/export works after Qwen3.5 MTP-disable patch
- new actor dispatch selects bridge distributed updater
- new updater emits and routes the correct full HF tensor set from rank0
- existing NCCL distributed send function works with a Ray fake rollout engine
- real 96-GPU fully_async bridge launch completes the first distributed Megatron-to-SGLang `update_weights`

Remaining check:

- run the same bridge launch with Harbor TrialWorkers correctly started, then confirm rollout data generation and the subsequent train step complete.

## Update 2026-06-20: Remaining check run — bridge TRAINING crashes (GatedDeltaNet packed-sequence)

The remaining check was run end to end on the 96-GPU 12-node cluster (run tag `tb2of_bridge_20260620_030923`, with TrialWorkers correctly started this time). Result:

- bridge load: OK
- 89-task step-0 baseline eval: completed (pass-rate 24.8%, 87/89 finished, coordinator total-timeout cancelled 2)
- rollout 0 data generation: OK (16/16 groups, 256 trajectories, reward_mean 0.488, 8/8 TrialWorkers healthy)
- distributed weight-sync: OK
- **first training step: CRASHED** with `NotImplementedError: GDN does not support packed sequence for now.`

So the two previously-documented blockers (distributed weight-sync, TrialWorkers) ARE fixed, but a THIRD blocker — in the training forward path — is now exposed. **Bridge mode cannot train Qwen3.5 yet.**

### Root cause (decisive — real traceback)

The real exception is in the ray JOB-DRIVER log (not the actor stdout, not the launcher log):
`/tmp/rtb2of/session_2026-06-20_03-10-55_523016_1419837/logs/job-driver-raysubmit_jLXQ3XjYm4XNeGSz.log:51366-51473`.

Call stack:

```
MegatronTrainRayActor.train()  (pid=2290599, ip=10.0.0.3)
  actor.py:409  train  ->  actor.py:504  train_actor
  actor.py:387  compute_log_prob  ->  model.py:382  forward_only
  model.py:345  output_tensor = model(**forward_kwargs)        # forward_kwargs carries packed_seq_params (qkv_format=thd)
  megatron/bridge/models/qwen_vl/modelling_qwen3_vl/model.py:626  forward
  .../text_model.py:162  forward
  megatron/core/transformer/transformer_block.py / transformer_layer.py:613  _forward_attention
  .../self_attention
  /root/Megatron-LM/megatron/core/ssm/gated_delta_net.py:300
    raise NotImplementedError("GDN does not support packed sequence for now.")
```

### Why bridge crashes but raw mode trains

The two model-construction paths wire DIFFERENT linear-attention (GatedDeltaNet) modules for Qwen3.5's SSM layers:

- **BRIDGE** (`slime/backends/megatron_utils/model_provider.py:84-125`): builds the model from `AutoBridge.from_hf_pretrained(...).to_megatron_provider()` and **ignores `args.spec`**. The HF config resolves to megatron.bridge's Qwen3-VL provider, whose linear-attn layers are **megatron-core's stock `GatedDeltaNet`** (`/root/Megatron-LM/megatron/core/ssm/gated_delta_net.py`), which unconditionally raises on `packed_seq_params is not None` (line 298-300).
- **RAW** (`--spec slime_plugins.models.qwen3_5 get_qwen3_5_spec`, from `scripts/models/qwen3.5-27B.sh`): installs slime's OWN `Qwen3_5GatedDeltaNet` (`slime_plugins/models/qwen3_5.py:86-191`), whose forward consumes `packed_seq_params.cu_seqlens_q` and runs varlen GDN correctly. thd packing works by design.

slime always builds `packed_seq_params` for thd (`slime/backends/megatron_utils/data.py:116-122`) and passes it into BOTH the log-prob forward (`model.py:340`) and the train forward (`model.py:532`). So the bridge model fails on the very first log-prob microbatch of rollout 0's training step.

### Why it initially looked like a silent 0.3s no-op

Misleading signals that masked the crash (do NOT re-misread these):

- `Timer log_probs end (0.2s)` / `Timer train end (0.3s)` are emitted by `finally`-block `Timer.end()` during exception UNWIND (rank-0 only); they are time-to-exception, not a completed forward.
- The tqdm `actor forward: 0/133` bar froze because `pbar.update(1)` runs only after `forward_step_func` returns (it raised), and `microbatch_pbar.close()` was skipped by the exception — so the `leave=False` bar is never erased.
- NO `actor_train` timer / ZERO `train/` metrics / no grad_norm / no optimizer step / no post-train resync, because execution died in `compute_log_prob` before reaching `with timer("actor_train")` (`actor.py:539`).
- The launcher log shows no error; the `RayTaskError(NotImplementedError)` only surfaces when `ray.get(actor_model.async_train(...))` re-raises on the driver (`train_async.py:78`).

### Fix options (to make bridge trainable)

1. **(preferred)** Register/patch the Qwen3.5 linear-attn submodule in `slime_plugins/megatron_bridge/` so the bridge provider wires slime's varlen `Qwen3_5GatedDeltaNet` instead of stock GDN. Because the bridge branch ignores `args.spec`, this must be injected provider-side (analogous to the GLM-4.6V custom bridge already registered in `slime_plugins/megatron_bridge/__init__.py`).
2. Teach megatron-core's `GatedDeltaNet` to consume `packed_seq_params` (upstream change at `gated_delta_net.py:298`) — large.
3. Stop building `packed_seq_params` for GDN layers (`data.py:116`) — large, broad blast radius.

Until one of these lands, **use raw mode** (`--megatron-to-hf-mode raw` + `--ref-load` torch_dist) for any run that needs to actually train Qwen3.5 — this is the proven path (mid830, tb2of_raw runs trained with 1682-2867s steps and full `train/` metrics).
