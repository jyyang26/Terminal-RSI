# Converting Qwen3.5-122B-A10B torch_dist checkpoints to HF safetensors

How to turn a Megatron `torch_dist` **actor** checkpoint from a slime/Harbor PPO run into
a HuggingFace `safetensors` directory that actually loads.

Everything here was derived from converting 9 real checkpoints (iter 29/39/49/59/69 of the
`88cp4-84k` run, iter 9/19 of `synthRTS_r3`, iter 9/19 of `strategy15k_r10`), including two
runs that **crashed an 84-GPU training job** and one that silently produced a corrupt
checkpoint. The guards below exist because of those failures, not as speculation.

---

## TL;DR

```bash
# 1. create the memory cgroup (NOT persistent -- gone after any container restart)
LIMIT_GB=400 ./ckptconv_cgroup_setup.sh

# 2. convert (actor only; torch_dist is never deleted)
CKPT=/path/to/exp/Qwen3.5-122B-A10B_<run>/ \
OUT_ROOT=/path/to/exp \
STEPS="9 19" \
ALLOW_CONCURRENT_TRAINING=1 \
  ./convert_qwen35_actor_to_hf.sh
```

Success looks like this, per iteration:

```
RAWCHECK indexes=1 keys=37599 shards=47 absent=0
VERIFY keys=1949/1949 keys_equal=True total_size=250173007840 delta=0 missing_shards=0
[ok] iter_19 -> .../q35_122b_iter19_hf   (torch_dist ... left untouched)
```

If `absent=0` / `delta=0` do not both hold, **the output is not usable** — see
[Verification](#verification).

---

## Why two steps are mandatory

`convert_torch_dist_to_hf_parallel.py` alone produces an **unloadable** checkpoint. It has
two independent defects, and the second is silent:

| | step 1 output (raw) | after step 2 (repair) |
|---|---|---|
| keys | 37599 | **1949** |
| expert layout | 36864 per-expert tensors (`...experts.{i}.gate_proj.weight`) | 96 fused (`...experts.gate_up_proj` / `.down_proj`) |
| `mtp.*` | absent (785 keys) | restored from origin |
| `model.visual.*` | absent (333 keys) | restored from origin |
| GDN dtype | bf16 | 72 tensors cast back to origin fp32 |

1. **Missing keys.** Megatron only trains/saves the language tower. `mtp.*` and the entire
   vision tower are never written — 1118 keys must come back from the origin HF checkpoint.
2. **Wrong expert layout.** The converter emits per-expert tensors, but transformers declares
   *fused* parameters — `gate_up_proj [num_experts, 2*intermediate, hidden]` and
   `down_proj [num_experts, hidden, intermediate]`
   (`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:723-724`). Per-expert keys match
   **no** parameter, so every expert would silently keep its randomly-initialised value —
   catastrophic, and invisible unless you diff the key sets.

Fusion layout is pinned by the two directions already in-tree:
`slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py:99` (`gate, up = param.chunk(2, dim=0)`)
and `slime_plugins/mbridge/qwen3_5.py:303` (`expert_w = w[global_expert_id]`), so
`fused[e] == cat([gate_e, up_e], dim=0)` and `down_proj` stacks directly.

> **Do not** use `211_roll_optimizer_off_old_checkpoints.py` for this. It performs the same two
> steps but **deletes the torch_dist tree** afterwards. The driver here never deletes anything.

---

## Sizes: why 228G becomes 234G

This is expected, not padding or overhead:

```
torch_dist weights   227.45 GiB  (du: 228G)    759 keys
HF total_size        232.99 GiB  (du: 234G)   1949 keys
difference             5.54 GiB  = the 1118 restored mtp/visual keys + fp32 GDN casts
```

## No optimizer state? Still convertible

`--no-save-optim` checkpoints convert fine. The converter skips optimizer keys explicitly
(`convert_torch_dist_to_hf_parallel.py:464`), so only the 759 weight keys matter. Verified on a
checkpoint with `optimizer bytes = 0.00 GiB`. Dropping optimizer state costs **resumability**,
not convertibility.

---

## Memory: the part that breaks things

Conversion is **pure CPU + host RAM** (no GPU). `--load-max-workers N` spawns N *processes*,
each holding its slice of the 227GB state dict **plus** its converted copy. Measured peaks:

| workers | peak RAM | step-1 time | safe to use when |
|---|---|---|---|
| 16 | >1 TB | — | **never** (see incident 2) |
| 6 | 831 GB | ~5 min | node is idle |
| 2 | 400 GB | ~12 min | a training job is running |

Peak is transient — once loading finishes RSS drops to ~14GB and the rest is reclaimable page
cache.

### Ray will kill your training job, and a cgroup does not prevent it

Ray's node-memory monitor computes used memory as

```
memory.usage_in_bytes - total_inactive_file - total_active_file
```

(`ray/_private/utils.py:584-606`, `get_used_memory()`), and kills actors above **0.95** of
`MemTotal`. Freshly-mmap'd safetensors count as *active file*, i.e. as "used". On a 2265GB node
that threshold is **2151GB**.

A cgroup limit bounds which process the **kernel** kills. It does **not** stop Ray, which looks
at total node memory. So the cap is about leaving Ray enough headroom — not about immunity.

Budget on the Ray head node: 8 `MegatronTrainRayActor.train` workers hold ~73GB host RAM each
(**~585GB total**, because `--optimizer-cpu-offload` keeps fp32 master weights + Adam m/v on the
CPU and PPO forces offload for actor *and* critic). Hence 400GB:

```
585 (training) + 400 (conversion) + ~200 (ray/system) ≈ 1.2 TB  of 2.27 TB
```

**Check before launching**, don't assume:

```bash
M=/sys/fs/cgroup/memory
u=$(cat $M/memory.usage_in_bytes)
ia=$(awk '/^total_inactive_file /{print $2}' $M/memory.stat)
ac=$(awk '/^total_active_file /{print $2}' $M/memory.stat)
echo "Ray-used = $(( (u-ia-ac)/1024/1024/1024 ))GB  (kill at $(( $(grep MemTotal /proc/meminfo|awk '{print $2}')/1024/1024*95/100 ))GB)"
```

Note `train_async` alternates phases: during **rollout/eval** the train actors are gone and
those ~585GB are not resident (observed Ray-used as low as 152GB); during a **training step**
they come back within seconds. Sizing for the rollout phase only is how you get surprised.

### Prefer a worker node

Conversion needs nothing but the shared filesystem. On a worker node there is no Ray driver and
no offloaded optimizer state (SGLang schedulers are ~4GB each), so you can use `WORKERS=6`
safely and never approach Ray's threshold. The cgroup is per-node, so create it there first.

---

## Verification

Two gates run automatically; both must pass.

**`RAWCHECK`** — every shard named in a worker index must exist on disk. This catches the
silent failure in incident 2: an OOM-killed worker leaves `multiprocessing.Pool` blocked while
its index still *references* shards that were never written.

**`VERIFY`** — for an **actor**, the key set and `total_size` must match origin exactly:

- keys = **1949**
- `total_size` = **250173007840**

`total_size` matching byte-for-byte is the cheapest end-to-end proof that every tensor landed
with origin's shape *and* dtype (it is what caught the 72 fp32 GDN tensors being silently
halved to bf16).

> A **critic** legitimately fails this: its `output_layer` is a `[1, 3072]` value head where
> origin carries the `[248320, 3072]` LM head, so its total is short by exactly one lm_head
> (1525671936 bytes). `210_repair` detects and allows that case. This driver converts actors
> only.

Independent post-hoc check:

```python
import json, os
O = "/path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B"
p = "<your>/q35_122b_iter19_hf"
o = json.load(open(O + "/model.safetensors.index.json"))
i = json.load(open(p + "/model.safetensors.index.json"))
sh = set(i["weight_map"].values())
print("keys      ", len(i["weight_map"]), set(i["weight_map"]) == set(o["weight_map"]))
print("total_size", i["metadata"]["total_size"] == o["metadata"]["total_size"])
print("absent    ", [f for f in sh if not os.path.isfile(os.path.join(p, f))])
print("mtp/visual/fused",
      sum(k.startswith("mtp.") for k in i["weight_map"]),               # 785
      sum(k.startswith("model.visual.") for k in i["weight_map"]),      # 333
      sum(k.endswith(".gate_up_proj") for k in i["weight_map"]))        # 48
```

---

## Loading the result

Use the architecture declared in `config.json` — **`Qwen3_5MoeForConditionalGeneration`**
(or `AutoModelForCausalLM`, which resolves to it):

| class | missing keys | note |
|---|---|---|
| `Qwen3_5MoeForConditionalGeneration` | 3 | ✅ the 3 are `rotary_emb.inv_freq` buffers, `persistent=False`, computed at init — origin's index lacks them too. 785 `mtp.*` are auto-ignored via `_keys_to_ignore_on_load_unexpected`. |
| `Qwen3_5MoeForCausalLM` | **832** | ❌ expects `model.layers.*`, checkpoint has `model.language_model.layers.*` → silently leaves the whole language tower randomly initialised. |

### Does the restored vision tower hurt text-only performance? No.

- The restored `model.visual.*` / `mtp.*` tensors are **bit-identical** to origin
  (`torch.equal` → True on sampled keys; only the language tower differs after training).
- Visual is fully gated: `if pixel_values is not None:` (`modeling_qwen3_5_moe.py:1766`). Pass no
  images → `self.visual` is never called and `inputs_embeds` is untouched. `Qwen3_5MoeTextModel`
  contains **zero** references to `visual`.
- `deepstack_visual_indexes` is `[]`, so there is no mid-layer visual injection path either.
- Cost is only 5.54 GiB of memory/disk for weights that never execute. To drop it, `del
  model.model.visual` after loading.

One real caveat, unrelated to weights: the multimodal wrapper computes 3D M-RoPE position ids.
For pure text `rope_deltas` stays `None` and it falls through to `position_ids = None` (language
tower computes its own), which matches text-model behaviour. Don't mix image and text-only
generate calls on the *same instance* and expect no interaction.

---

## Incident history

**1 — 2026-08-14 / 08-15: killed an 84-GPU PPO run, twice.**
`ray.exceptions.OutOfMemoryError: ... node ... was 2180.05GB / 2265.25GB (0.962388)`. Root cause
was a concurrent conversion on the Ray **head** node. The cgroup was missing (not persistent),
the join degraded to a warning, and the conversion ran uncontained. Ray's threshold was breached
in exactly two windows across the whole multi-day run, and both coincide with a conversion.
→ Hence guard 1 refuses to start uncontained, and guard 2 refuses to run beside `train_async.py`.

**2 — 2026-08-15 17:19-17:28: silent corruption + deadlock.**
16 workers under a 240GB cap; 3 workers OOM-killed. `multiprocessing.Pool` respawned the
processes but **never re-ran their tasks**, so `starmap()` blocked forever with 7 shards
referenced-but-never-written. Symptom: log mtime *and* output-dir size both frozen while workers
sit in `S` state, `memory.oom_control` `oom_kill` > 0.
→ Hence `WORKERS=2`, the `RAWCHECK` gate, and the stall watchdog.

**3 — a 240GB cap was itself the bug.** During incident 2 the host still had 469GB free; the cap
was the binding constraint. Note this node's swap is **zram** (compressed RAM, not disk), so
"spilling to swap" still consumes physical RAM and buys no headroom.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `[cgroup] ERROR ... does not exist` | cgroup is not persistent across restarts | `LIMIT_GB=400 ./ckptconv_cgroup_setup.sh` |
| log mtime + output size both frozen, `oom_kill > 0` | incident 2 (Pool deadlock) | lower `WORKERS`, raise cap; watchdog kills it after `STALL_SECS` |
| `RAWCHECK ... absent=N` (N>0) | worker died mid-save | re-run; do **not** use the raw dir |
| `total_size mismatch ... delta=1525671936` | you converted a **critic**, not an actor | expected for critic; this driver is actor-only |
| 832 missing keys at load time | wrong class | use `Qwen3_5MoeForConditionalGeneration` |
| `[ERROR] cache_position ... not documented` | harmless transformers docstring warning | ignore |
| training dies with `OutOfMemoryError` at ~0.96 | incident 1 | convert on a worker node, or when the trainer is down |

Per-step logs: `/tmp/conv_step1_iter<N>.log`, `/tmp/conv_step2_iter<N>.log`.

---

## Files

| file | role |
|---|---|
| `convert_qwen35_actor_to_hf.sh` | **entry point** — both steps + both gates, keeps torch_dist |
| `ckptconv_cgroup_setup.sh` | creates the `ckptconv` memory cgroup (run first, per node) |
| `convert_torch_dist_to_hf_parallel.py` | step 1 — torch_dist → raw HF (output NOT loadable alone) |
| `210_repair_qwen35_moe_converted_hf.py` | step 2 — fuse experts, restore mtp/visual, fix dtypes |
| `209_repair_hf_missing_weights_from_origin.py` | older/simpler repair: restores missing keys but does **not** fuse experts — insufficient for Qwen3.5-MoE |
| `211_roll_optimizer_off_old_checkpoints.py` | watcher that converts **and deletes** torch_dist to reclaim space — not used here |
| `convert_torch_dist_to_hf.py` | serial reference implementation of step 1 |

Timings (2 workers, alongside training): step 1 ~12 min, repair ~17 min, ~30 min total.
With 6 workers on an idle node: ~15-25 min total.
