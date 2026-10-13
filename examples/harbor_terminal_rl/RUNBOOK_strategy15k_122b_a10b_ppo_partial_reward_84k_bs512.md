# RUNBOOK — Qwen3.5-122B-A10B / strategy-15k / PPO + CTRF dense reward @ 176 GPU, 84k, bs512

> Run the validated PPO recipe at **batch 512** with **84k context on both sides**, on the
> DeepSeek-audited **strategy-15k** pool, with a **warm critic** (iter_69) and a
> **cold RST actor**. R3 routing replay ON, checkpoints weights-only.

---

## §0 TL;DR

```bash
cd /path/to/workspace/slime-terminal-RL

# 0) cluster: 22 nodes / 176 GPU wanted. After the platform grants the extra nodes:
bash /path/to/workspace/stable_rl/scripts/start_ray_cluster.sh
bash envs/setup_new_nodes.sh --from-node-ip-list     # deps + verification (see §2.1)

# 1) dry run (no GPU spent; prints every resolved knob)
DRY_RUN=1 examples/harbor_terminal_rl/run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest_partial_reward.sh

# 2) launch: exactly 3 epochs over 15000 tasks @ bs512
NUM_ROLLOUT=90 examples/harbor_terminal_rl/run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest_partial_reward.sh

# 2') if only 17 nodes / 136 GPU are available (128 train + 8 infer, 1 engine):
TOPOLOGY=136cp4 NUM_ROLLOUT=90 examples/harbor_terminal_rl/run_...strongest_partial_reward.sh

# 3) watch
examples/harbor_terminal_rl/synth_all_rts_122b_partial_reward_health_check.sh
examples/harbor_terminal_rl/r3_disk_watchdog.sh

# 4) stop
ray job list && ray job stop <job_id>
```

Everything below is already the script's default — no env vars needed except `NUM_ROLLOUT`.

---

## §1 Configuration at a glance

| axis | value | where it comes from |
|---|---|---|
| topology | `176cp4`: **128 train** (TP2×PP4×**CP4**⇒DP4, EP16, 12 layers/stage) + **48 infer** (6×TP8) | wrapper §(0) |
| batch | **512** / rollout 512 / oversample 560 | wrapper §(0) |
| context | **86016 (84k) on BOTH sides**; compaction at 65536 → 20480-token buffer | wrapper §(0b) + yaml |
| data | **strategy_deepseek_repaired_v4_final_15k** `materialized_top15k/tasks`, 15000 tasks, **shuffled** | yaml |
| reward | `partial_reward.test_count_progress_rewards`, `passed/20` | wrapper §(1) |
| actor | `RST_models/Qwen3.5-122B-A10B_torch_dist` (**cold**, marker `release`) | load-root |
| critic | tmax15k 88cp4 **iter_0000069** (**warm**, weights-only) | load-root `_critic` sibling |
| critic warm-up | `NUM_CRITIC_ONLY_STEPS=2` — actor joins at rollout_id 2 | wrapper §(3) |
| routing replay | `--use-rollout-routing-replay` **ON** | wrapper §(4) |
| checkpoints | `--no-save-optim` (weights only, **NOT resumable**), every 10 steps | wrapper §(4) |
| lr | critic 30x (1.5e-5) / actor 2x (1.0e-6) | `megatron_roles_tmax15k_criticlr30x_actorlr2x.yaml` |
| schedule | 3 epochs = `ceil(15000/512)=30 × 3` = **90 steps** | `NUM_ROLLOUT=90` |

Projected ~42 min/step at 176 GPU ⇒ **~1.1 days for 90 steps**.
(At `136cp4` the single inference engine makes rollout ~6× slower; expect ~3-4 days.)

---

## §2 Prerequisites

### 2.1 Cluster: 22 nodes / 176 GPU + per-node dependencies

`/usr/local/lib/python3.12/dist-packages` is on **overlayfs (node-local)**, NOT the shared
filesystem, so **every new node needs its own install**. Hand-listing package names does
not work — see `envs/README_cluster_env_setup_zh.md` §6, where a new node passed a
25-package version table and still died in rollout on `shortuuid`, then `litellm`, then
`dirhash` (imported lazily *inside a function*, so no module-level probe can see it). It
was short **98 packages**.

Use the automated path, which diffs the **entire** package set against a worker already
proven to run rollout, and installs with `--no-deps` so pip cannot upgrade torch on one
node and desync the cluster:

```bash
bash /path/to/workspace/stable_rl/scripts/start_ray_cluster.sh   # head + workers
bash envs/setup_new_nodes.sh --from-node-ip-list    # or: setup_new_nodes.sh <ip> <ip> ...
bash envs/verify_node_env.sh                        # must print IDENTICAL for every node
ray status | grep GPU                               # expect 176.0 GPU
```

`setup_new_nodes.sh` also pins **fla 0.4.2** (CP>1 hard dependency — 0.4.1 has no
`fla.ops.cp` at all, and every GDN layer would raise at its first forward, ~20 min into
the run) and pre-creates the Triton cache dirs.

> The launcher **refuses to start** if the live cluster is smaller than the topology needs,
> instead of letting Ray block forever on an unschedulable placement group.

### 2.2 Data

Pool: `.../slime-terminal-RL_data/strategy_deepseek_repaired_v4_final_15k/materialized_top15k/tasks`
— **15000 real task dirs** (not symlinks). Verified: 15000/15000 resolvable `*/task.toml`,
**CTRF present in 93%** of sampled `tests/test.sh` (56/60).

DeepSeek-V4-Pro audited: weighted quality score over 8 dimensions, `hard_reject` for hidden
requirements, test leakage, solution shortcuts, or verifiers too weak to validate the goal.
See the pool's own `README.md`.

`rollout_shuffle: true` is **required**: the pool is materialised in *quality-rank* order
(`quality15k_v4_00001` = best), so unshuffled every 512-batch would come from one narrow
quality band.

### 2.3 Models

```
<LOAD_ROOT>/Qwen3.5-122B-A10B_base        -> RST_models/..._torch_dist     (actor, marker "release")
<LOAD_ROOT>/Qwen3.5-122B-A10B_base_critic -> tmax15k 88cp4 ..._critic dir   (marker 69)
LOAD_ROOT = /path/to/shared_storage/.../.slime_load_roots/strategy15k_ppo_partialreward_warmcritic69
```

Why this shape: `_isolate_critic_checkpoint_dirs` (`slime/ray/placement_group.py:151-224`)
redirects the critic's `--load` to `<load>_critic` **only if that sibling exists**, and
then loads it **weights-only** (`no_load_optim/no_load_rng/finetune`). So the symlink pair
gives a warm value head with zero core changes.

`start_rollout_id` stays **0** even though the critic says iter 69: `arguments.py` sets it
from the **actor's** `release` marker first, and `placement_group` only fills it when still
`None`. The weights-only critic load also means the 69-vs-90 schedule mismatch cannot
assert-crash (the LR scheduler is skipped entirely).

Note the critic symlink points at the **parent** directory (the one holding
`latest_checkpointed_iteration.txt` = 69), not at `iter_0000069/` itself — Megatron reads
that marker to pick the iteration.

---

## §3 Context: 84k both sides, 64k trained + 20k compaction buffer

Previously the two sides **disagreed**: training was walled at 64k, SGLang served 84k, and
the agent believed it had **128k** — so long trajectories were silently truncated. Now:

| knob | value | meaning |
|---|---|---|
| `SEQ_LENGTH`, `HARBOR_MAX_TRAIN_TOTAL_TOKENS` | 86016 | training-side token budget |
| `ROLLOUT_MAX_CONTEXT_LEN`, `SGLANG_CONTEXT_LENGTH`, `SGLANG_MAX_PREFILL_TOKENS` | 86016 | serving-side |
| agent `max_context_tokens` | **86016** (was 128000) | the ceiling the agent believes |
| agent `proactive_summarization_threshold` | **65536** (was 20000) | compact once context hits 64k |

`86016 − 65536 = 20480` tokens of headroom for the compaction/summarisation pass itself, so
it never crosses the 84k wall. The 64k prefix is what is rolled out **and** trained on.

**CP=4 is what makes an 84k training wall fit.** The binding term is the fp32 logits on the
last pipeline stage (vocab 248320), local `T = 86016/CP`:

| CP | local T | logits | last-stage peak (H20 = 95.6 GiB) |
|---|---|---|---|
| 2 | 43008 | 29.8 GB | 89.5 GB (+6.1 — too tight) |
| **4** | 21504 | **14.9 GB** | **64.4 GB (+31.2 — safe)** |

PP must stay 4 (PP2 → 24 layers/stage → OOM). CP=4 numerics are measured, not assumed:
forward bit-exact vs single-rank, grad max diff 1.2e-7 (`tests/test_qwen3_5_gdn_cp_*.py`).

Why 96 or 160 train GPU are **illegal**: they give DP 6 / 10, and `EP*ETP (16)` must divide
`TP*CP*DP`. Legal DP values here are 2/4/8, and `512 % DP == 0` must also hold.

---

## §4 Post-launch checklist

1. `[preflight]` block: cluster size OK, 15000 tasks, ctrf ~19/20, `shuffle = true`,
   context all `86016`, `critic = WARM ... (iter 69, weights-only)`, batch 512,
   `critic-only steps = 2`, misc contains all three of
   `--moe-token-dispatcher-type alltoall --use-rollout-routing-replay --no-save-optim`.
2. `[topology]` line: `train=128 (TP2 x PP4 x CP4 => DP4, EP16), 12 layers/stage, infer=48`.
3. `grep -ac 'successfully loaded checkpoint'` → 3.
4. `grep -acE 'CUDA out of memory|torch.OutOfMemoryError'` → **must stay 0**.
5. **`rollout_id 2` is the first actor+critic joint step.** `train/pg_loss` appears only
   from then on; steps 0-1 log `critic-*` metrics only.
6. R3 health: `grep -acE 'Missing routed_experts|rollout_routed_experts is required|RolloutConversionError'` → **0**.
   Do **not** grep for `placeholder`: the agent writes that word into its own shell
   commands, which produces pure false positives.
7. Dense reward is really firing: `ctrf_probe` line in the health check (want >90%), and
   `harbor/reward_mean` should be **fractional** (a binary reward would land on multiples
   of 1/512). `harbor_progress_*` will **not** appear in wandb — see §5.

Normal-but-alarming: GPU util 0% during rollout (agents solve on remote Daytona); a CP>1
Triton cold-compile storm on the first steps (per-microbatch 400s decaying to ~20s);
`RuntimeError`/`Traceback` text from agent commands; ~5% trial failures
(`No reward file found`) — tolerated by `harbor_fail_on_trial_error: false`, which is what
the 560→512 oversample margin is for.

---

## §5 Known gotchas carried forward

- **`train/kl_loss` is non-zero under R3, by design.** `model.py:805-813` and
  `data.py:375-385` disable their KL≈0 assertions when routing replay is on, because the
  actor replays rollout routing while the reference forward routes normally.
- **Reward-hook metadata never reaches wandb.** `_split_train_data_by_dp`
  (`slime/ray/rollout.py:1073-1087`) rebuilds each DP rank's `rollout_data` from an
  explicit key whitelist that omits `"metadata"`. That whitelist is also load-bearing:
  `log_rollout_data` would do `sum(list-of-dicts)` and raise. Observe progress fields via
  the health check's `ctrf_probe`, `rollout/rewards` continuity, or `rollout_dumps/*.pt`
  (dumped before the DP split).
- **`--no-save-optim` means NOT resumable.** Checkpoints are for eval / fresh cold-start.
  A crash at step 60 means restarting, not resuming.
- **`EXTRA_MISC_ARGS` is assembled by the wrapper**, because the parent *overwrites* it.
  Passing it by hand has historically dropped `--moe-token-dispatcher-type alltoall`.
  Override via `USE_ROUTING_REPLAY=0` / `NO_SAVE_OPTIM=0` instead.
- **Trial dirs are an OUTPUT, never pre-populate them.** `_trial_name()` is
  `slime-r{rollout}-g{group}-s{rank}-i{index}` with no run tag, so reusing a previous run's
  dir would let a new trial that fails to emit ctrf be scored on **another model's**
  leftover `ctrf.json`. The wrapper rewrites `trials_dir` to a `${RUN_TAG}` path and
  refuses non-empty dirs unless `ALLOW_DIRTY_TRIALS_DIR=1`.

---

## §6 ⚠️ Disk

Measured on the 122B runs: **455 GB per save point** (weights-only; 3.2 TB *with*
optimizer) and **~92 GB of trial artifacts per rollout at bs256** — expect roughly
**double that at bs512**. slime has **no checkpoint retention/pruning knob**; every save
point is kept forever.

For 90 steps at `SAVE_INTERVAL=10`: ~9 × 455 GB ≈ **4 TB of checkpoints** plus
~90 × ~180 GB ≈ **16 TB of trials**. Check headroom before launching:

```bash
examples/harbor_terminal_rl/r3_disk_watchdog.sh    # prints remaining-step headroom
```

Largest reclaimable item observed: `slime_harbor_trials/tmax15k_122b_a10b_ppo_cp2_bs256`
(~4.9 TB, 32062 trial dirs from a finished run). Do **not** delete
`slime_harbor_q35_122b_a10b_88cp4-84k_bs256_20260813_004233` — that 10 TB run is the
**source of this run's warm critic** (iter_0000069).
