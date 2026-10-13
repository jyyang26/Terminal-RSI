#!/bin/bash
#
# ====================================================================================
# HOW TO RUN
# ====================================================================================
#   cd /path/to/workspace/slime-terminal-RL
#
#   # 0) one-time prerequisites
#   #    Credentials are read automatically from /tmp/bs512_keys.env (mode 0600) if they
#   #    are not already exported. NEVER put keys in this file — it lives in a git tree.
#   #      umask 077
#   #      printf 'export DAYTONA_API_KEY=...\nexport WANDB_API_KEY=...\n' > /tmp/bs512_keys.env
#   #    CP>1 additionally needs, on EVERY node:
#   #      pip install -U 'fla-core==0.4.2' 'flash-linear-attention==0.4.2'
#
#   # 1) launch. Everything is self-contained: the script sources credentials, creates a
#   #    timestamped run dir, clears stale stop-markers, reuses the running ray cluster
#   #    (CLEAN_START=0 / MASTER_ADDR pinned) and detaches with nohup. No wrapper needed.
#   #    It returns immediately and prints the log path + watch commands.
#
#   #    >>> MAXIMUM-HEADROOM 72-GPU CONFIG (64 train + 8 infer, CP2, bs256, 3 h eval) <<<
#   #    projected peak ~73.8 GB / 95 GB  (+21 GB headroom -- the widest available)
#   TOPOLOGY=72cp2-64tr examples/harbor_terminal_rl/run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest.sh
#
#   #    alternative: 48 train + 24 infer (3x faster rollout, but only ~+8 GB headroom)
#   TOPOLOGY=72cp2-bs256 examples/harbor_terminal_rl/run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest.sh
#
#   #    other presets
#   TOPOLOGY=72    ...   # 48 train + 24 infer, CP1, batch 128 (the config that OOM'd)
#   TOPOLOGY=72cp2 ...   # 48 train + 24 infer, CP2, batch 128
#   TOPOLOGY=96    ...   # 64 train + 32 infer, CP1
#   TOPOLOGY=128   ...   # 96 train + 32 infer, CP1
#
#   #    check a config without spending GPUs (prints the resolved plan, exits):
#   DRY_RUN=1 TOPOLOGY=72cp2-bs256 examples/harbor_terminal_rl/run_...strongest.sh
#   #    run in the foreground instead of detaching:
#   BACKGROUND=0 TOPOLOGY=72cp2-bs256 examples/harbor_terminal_rl/run_...strongest.sh
#
#   TOPOLOGY       nodes  train GPU (parallelism)       infer GPU      L/stage  batch
#   -------------  -----  ---------------------------   ------------   -------  -----
#      72            9    48 (TP2 x PP3 x CP1 => DP8)   24 (3 x TP8)     16      128
#      72cp2         9    48 (TP2 x PP3 x CP2 => DP4)   24 (3 x TP8)     16      128
#      72cp2-bs256   9    48 (TP2 x PP3 x CP2 => DP4)   24 (3 x TP8)     16      256
#      72cp2-64tr    9    64 (TP2 x PP4 x CP2 => DP4)    8 (1 x TP8)     12      256
#      88cp2        11    64 (TP2 x PP4 x CP2 => DP4)   24 (3 x TP8)     12      256  <= 88-GPU 主力
#      88cp4-84k    11    64 (TP2 x PP4 x CP4 => DP2)   24 (3 x TP8)     12      256  <= 84k 上下文
#      88cp8-128k   11    64 (TP2 x PP4 x CP8 => DP1)   24 (3 x TP8)     12      128  <= 128k, DP1 慎用
#      96           12    64 (TP2 x PP4 x CP1 => DP8)   32 (4 x TP8)     12      128
#     128           16    96 (TP2 x PP6 x CP1 => DP8)   32 (4 x TP8)      8      128
#
#   The cp2 rows use native GatedDeltaNet context parallelism (fla.ops.cp), which halves
#   the GDN activation peak — the term that actually OOMs on long trajectories. Measured
#   on the real 122B GDN dims: 10.57 -> 5.34 GB at T=33792 (1.98x), forward BIT-EXACT vs
#   CP1, grad max diff 1.2e-7 (tests/test_qwen3_5_gdn_cp_*.py). Cost: DP 8 -> 4 and two
#   extra all-gathers per GDN layer. At batch 256 with DP4 that is 64 grad-accum
#   micro-batches per step (vs 16 at bs128/DP8) — roughly 4x the step time. Deliberate.
#
#   Every divisibility constraint, the GPU budget, and the presence of fla.ops.cp are
#   validated BEFORE the ~15-20 min init, so a bad override fails in seconds.
#
#   # 2) watch it. The script prints the exact log path and watch commands on launch.
#   L=$(ls -td /path/to/workspace/slime_harbor_q35_122b_a10b_*/launch_122b_*.log | head -1)
#   tail -f "$L"
#   grep -acE "CUDA out of memory|torch.OutOfMemoryError" "$L"  # must stay 0
#   grep -aoE "perf [0-9]+:" "$L" | tail -3                     # rollout progress
#   grep -ac "Timer update_weights end" "$L"                    # completed train steps
#
#   The step to watch is rollout_id 2: NUM_CRITIC_ONLY_STEPS=2 means steps 0 and 1 are
#   critic-only, so rollout_id 2 is the first actor+critic joint step — that is where the
#   32-train-GPU topology OOM'd. Getting past it is the real validation.
#
#   # 3) stop it
#   ray job list                        # find the raysubmit_... id
#   ray job stop <job_id>
#
#   Overrides: every knob below is `${VAR:-default}`, so anything can be changed from the
#   environment, e.g. `TOPOLOGY=72 GLOBAL_BATCH_SIZE=64 MAX_TOKENS_PER_GPU=16384 ./...sh`.
#   If you override the topology by hand, keep DP = train_gpus/(TP*PP*CP) dividing
#   GLOBAL_BATCH_SIZE, and keep EP*ETP dividing TP*CP*DP.
#
#   Design rationale, the calibrated memory model and the full investigation:
#   claude_sessions/2026-08-06_122b_a10b_oom_topology_investigation_zh.md
# ====================================================================================
#
# tmax-15k + Qwen3.5-122B-A10B (MoE) + strongest PPO recipe — SCALABLE topology,
# OOM-safe at 72 / 96 / 128 GPU. STANDARD (skim-oversample) rollout, DAYTONA backend.
#
# Cloned from run_tmax15k_122b_a10b_56gpu_ppo_criticlr30x_strongest.sh. The RL recipe is
# IDENTICAL (PPO eps-clip 0.2, KL OFF, entropy OFF, cold-start critic, critic lr 30x /
# actor lr 2x, NUM_CRITIC_ONLY_STEPS=2, n=1 train / pass@3 eval, 64k walls, FT on).
# ONLY the parallel topology and the memory knobs change.
#
# ====================================================================================
# WHY THIS FILE EXISTS: the 56/72-GPU run OOMs at rollout_id 2
# ====================================================================================
# The 32-actor-GPU topology (TP2 x PP2 x CP1 x EP16 => DP8) crashes in the CRITIC's
# GatedDeltaNet forward at the first actor+critic joint step:
#
#   fla/ops/common/chunk_delta_h.py:493  v_new = torch.empty_like(u)
#   torch.OutOfMemoryError: Tried to allocate 632.00 MiB.
#   GPU 1 ... Process ... has 90.56 GiB memory in use   (H20 = 95 GiB usable)
#
# Three facts drive the fix (all verified in code, not guessed):
#
# (1) THE CRITIC IS COLOCATED ON THE ACTOR'S GPUs, and it is a FULL 122B model.
#     slime/utils/arguments.py:1820  critic_num_gpus_per_node = actor_num_gpus_per_node
#     slime/utils/arguments.py:1821  critic_num_nodes         = actor_num_nodes
#     slime/ray/placement_group.py:135  result["critic"] = result["actor"]   <- SAME pg
#     So every "actor GPU" holds TWO 122B models. --offload is forced ON for PPO
#     (arguments.py:1862-1863: `if args.use_critic: args.offload_train = True`), so they
#     alternate rather than co-reside, but each one alone must still fit in 95 GiB.
#
# (2) EXPERT OPTIMIZER MEMORY DEPENDS ONLY ON TOTAL TRAIN GPUs — NOT on EP.
#     Expert params are 116.0B of the 121.4B total (95.5%). Per-rank expert params scale
#     as 1/(EP*ETP), but the distributed-optimizer shard factor EDP = TP*CP*DP/(EP*ETP)
#     scales the SAME way, so the two cancel exactly:
#
#         expert optimizer bytes/GPU = P_expert * 12 / (total train GPUs)
#
#     => Tuning EP CANNOT reduce optimizer memory. Only ADDING TRAIN GPUs can.
#     => LOWERING EP is actively HARMFUL: it doubles per-rank expert weights+grads
#        (6 B/param) while leaving the optimizer term unchanged. Keep EP maximal.
#
# (3) RAISING TP DOES NOT HELP THIS MODEL. The GatedDeltaNet is implemented with plain
#     nn.Linear (slime_plugins/models/qwen3_5.py — no ColumnParallel/RowParallel, no
#     parallel_state), and slime_plugins/models/hf_attention.py:107-113 ALL-GATHERS the
#     sequence back to full length before calling it:
#         hidden_states = gather_from_sequence_parallel_region(..., tensor_parallel_output_grad=False)
#         # comment in-repo: "the linear attention after this gather is NOT TP-sharded
#         #                   (duplicated on all ranks)"
#     So at TP>2 the 2.94B GDN params are REPLICATED on every TP rank and the GDN
#     activation peak does NOT shrink. Worse, num_query_groups=2 < TP means
#     megatron/core/transformer/attention.py:289 takes the KV-replication branch.
#     => TP stays 2. The only useful lever for the dense/activation side is PP.
#
# CP: GatedDeltaNet's recurrent scan is not CP-safe in the generic path, so CP was pinned
# to 1 historically. TOPOLOGY=72cp2 now uses NATIVE GDN CP via fla.ops.cp instead (see the
# CP_SIZE section below) -- verified bit-exact vs CP1 with a ~1.98x activation reduction.
#
# ====================================================================================
# THE MEMORY MODEL (calibrated against the real crash, +-4 GB)
# ====================================================================================
#   static/GPU = (P_ep/PP/(EP*ETP) + P_gdn/PP + P_tp/PP/TP) * 6 B      # bf16 w + fp32 grad
#              + P_ep*12/(PP*EP*ETP*EDP) + (P_gdn+P_tp)*12/(PP*DP)     # adam m,v,master
#   activ/GPU  ~ (L/PP)*(T/TP)*h*2                                     # recompute inputs
#              + 2 * max(GDN_peak(T), MoE_peak(T/TP))                  # fwd recompute+bwd
#   overhead   ~ 7 GB  (measured: 4.33 GB fragmentation + NCCL/ctx + 1 GB memory_saver margin)
#
#   P_ep = 116.0B (expert)   P_gdn = 2.94B (TP-replicated)   P_tp = 2.92B (TP-shardable)
#   GDN activation is INDEPENDENT of TP (post-all-gather). MEASURED on the real 122B GDN
#   dims: ~330 KiB/token, NOT the 80 KiB/token this model originally assumed -- that 4x
#   error is why the earlier peak-memory predictions came out systematically low. At
#   CP=2 it halves to ~165 KiB/token. Per-layer, per-micro-batch, set by the LARGEST
#   SINGLE SAMPLE, so GLOBAL_BATCH_SIZE does not enter it.
#
#   Validation on the crashed config (TP2 PP2 DP8 EP16, 32 train GPU):
#     predicted @40k-token sample = 90.5 GB   vs   MEASURED 90.56 GB   <- matches
#     predicted @64k-token sample = 96.7 GB   ->   exceeds 95 GB       <- explains OOM
#   The 632 MiB v_new alloc corresponds to a ~40k-token sample (632 MiB / 16 KiB per
#   token), i.e. an ordinary tmax-15k trajectory. NOTE: --max-tokens-per-gpu CANNOT
#   prevent this. first_fit_pack (slime/utils/seqlen_balancing.py) documents that "an
#   oversized sample lands alone in its own bin with sum equal to its length" — samples
#   are indivisible, so a 40k sample forms its own micro-batch regardless of the cap.
#   That is why lowering MAX_TOKENS_PER_GPU 8192 -> 4096 did NOT fix the crash.
#
# ====================================================================================
# TOPOLOGIES (TP2 x CP1 fixed; PP absorbs the growth; EP kept maximal)
# ====================================================================================
#  total  train  infer  PP  DP  EP  L/stage   static  activ(64k)  peak    headroom
#  ----------------------------------------------------------------------------------
#    56*    32     24    2   8  16     24      78.8     14.5     100.3   -5.3  OOM (current)
#    72     48     24    3   8  16     16      54.3     13.0      74.3   +20.7 OK  <- TOPOLOGY=72
#    96     64     32    4   8  16     12      42.1     12.2      61.3   +33.7 OK  <- TOPOLOGY=96
#   128     96     32    6   8  16      8      29.8     11.5      48.3   +46.7 OK  <- TOPOLOGY=128
#
# Every row keeps DP=8, so the gradient semantics, GLOBAL_BATCH_SIZE=128 divisibility
# (128 % 8 == 0) and the per-step loss reduction are IDENTICAL across all three scales —
# only memory headroom and throughput change. Constraint checks for each row:
#   - TP*PP*CP divides train GPUs                    (launcher asserts this, line ~171)
#   - EP*ETP divides TP*CP*DP  (16 | 16)             (megatron initialize.py)
#   - 48 layers divisible by PP  (16/12/8 per stage)
#   - layers/stage % full_attention_interval(4) == 0 -> the [lin,lin,lin,full] pattern
#     splits evenly, so every stage gets the same linear/full attention mix. This matters
#     because qwen3_5.py:225-232 selects the attention class by GLOBAL layer index
#     (layer_id + get_transformer_layer_offset).
#   - each PP stage spans exactly 2 nodes at 72/96 GPU, keeping EP16 all-to-all node-local.
#
# WHY NOT more DP instead of more PP: DP does not reduce the per-micro-batch activation
# peak, and raising DP forces EP down (EP*ETP must divide TP*CP*DP), which per fact (2)
# inflates weights+grads for zero optimizer benefit.
#
# MAX_TOKENS_PER_GPU is restored to 8192. Per the note above it cannot bound an oversized
# single sample, so 4096 bought nothing while doubling the micro-batch count (slower).
# PYTORCH_CUDA_ALLOC_CONF stays max_split_size_mb:256 — expandable_segments:True is
# INCOMPATIBLE with the torch_memory_saver that PPO's forced offload requires and kills
# the job at startup (see the allocator section below).
#
# USAGE
#   TOPOLOGY=72  ./run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest.sh   # 48 train + 24 infer
#   TOPOLOGY=96  ./run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest.sh   # 64 train + 32 infer
#   TOPOLOGY=128 ./run_tmax15k_122b_a10b_scalable_ppo_criticlr30x_strongest.sh   # 96 train + 32 infer
#   Any individual knob can still be overridden from the environment.
#
# BEFORE LAUNCH export: DAYTONA_API_KEY, HARBOR_SRC, WANDB (logged in).

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export SLIME_DIR="${SLIME_DIR:-/path/to/workspace/slime-terminal-RL}"
export CODE_ROOT="${CODE_ROOT:-${SLIME_DIR}}"
export HARBOR_SRC="${HARBOR_SRC:-/path/to/workspace/harbor-code-synthesis/src}"

# --- ROLLOUT MODE: force standard --------------------------------------------------
export HARBOR_ROLLOUT_MODE="standard"

# --- MODEL: Qwen3.5-122B-A10B (RAW torch_dist load; cold-start critic) -------------
# RAW mode: load the pre-converted Megatron torch_dist (built by
# convert_qwen35_122b_a10b_torch_dist.sh). The RUNTIME bridge path (megatron.bridge)
# fails for this 122B model at PP>1, so we do NOT use bridge.
# The torch_dist is PP-agnostic: release/.metadata is present, so Megatron's
# torch_dist loader RESHARDS freely to any TP/PP/EP at load time. It was written with
# PP8 and is being read here at PP3/PP4/PP6 — no reconversion needed.
export MODEL_SCRIPT="${MODEL_SCRIPT:-${SLIME_DIR}/scripts/models/qwen3.5-122B-A10B.sh}"
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
export HF_CKPT="${HF_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/path/to/shared_storage/hf_models/Qwen/Qwen3.5-122B-A10B_torch_dist}"
export STAGE_HF_CKPT_TO_TMP="${STAGE_HF_CKPT_TO_TMP:-0}"
export STAGE_TORCH_DIST_CKPT_TO_TMP="${STAGE_TORCH_DIST_CKPT_TO_TMP:-0}"

export OVERRIDE_OPT_PARAM_SCHEDULER="${OVERRIDE_OPT_PARAM_SCHEDULER:-1}"

# --- per-role lr: critic 30x / actor 2x — reuse the source YAML --------------------
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_tmax15k_criticlr30x_actorlr2x.yaml}"
export LR="${LR:-5e-7}"
# NOTE: NUM_CRITIC_ONLY_STEPS is deliberately NOT defaulted here -- the TOPOLOGY case
# block below needs to be able to set it (e.g. the -joint preset uses 0), and an
# `export VAR="${VAR:-2}"` at this point would pin it to 2 first and make the preset's
# `: "${VAR:=0}"` a no-op. It is defaulted right after the case block instead.

# ====================================================================================
# TOPOLOGY SELECTION
# ====================================================================================
export TOPOLOGY="${TOPOLOGY:-72}"

case "${TOPOLOGY}" in
  72)
    # 9 nodes: 6 actor nodes (48 GPU) + 3 rollout nodes (24 GPU)
    _EXPECTED_NODES=9;  _ACTOR_NODES=6; _ROLLOUT_GPUS=24; _PP=3; _CP=1   # 48/(2*3)=DP8, 16 layers/stage
    ;;
  72cp2)
    # 9 nodes: 6 actor nodes (48 GPU) + 3 rollout nodes (24 GPU), hybrid GDN CP=2.
    # Same GPU split as TOPOLOGY=72, but CP2 halves the GDN activation peak -- which is
    # what actually OOMs on long trajectories -- at the cost of DP 8 -> 4.
    # REQUIRES flash-linear-attention >= 0.4.2 (fla.ops.cp) on EVERY node, and
    # FLA_INTRACARD_CP=0 (see the CP section below).
    _EXPECTED_NODES=9;  _ACTOR_NODES=6; _ROLLOUT_GPUS=24; _PP=3; _CP=2   # 48/(2*3*2)=DP4, 16 layers/stage
    ;;
  72cp2-bs256)
    # TOPOLOGY=72cp2 plus the batch-256 / 3-hour-eval preset. One-command launch of the
    # configuration validated on 2026-08-11:
    #   48 train (TP2 x PP3 x CP2 => DP4, EP16) + 24 infer (3 x TP8)
    #   batch 256, oversample 280, save+eval every 10 steps, eval total timeout 3 h
    # NOTE batch 256 with DP4 means 64 micro-batches of grad-accum per step (vs 16 at
    # bs128/DP8), i.e. roughly 4x the step time. Deliberate: correctness over speed.
    _EXPECTED_NODES=9;  _ACTOR_NODES=6; _ROLLOUT_GPUS=24; _PP=3; _CP=2
    : "${GLOBAL_BATCH_SIZE:=256}"
    : "${ROLLOUT_BATCH_SIZE:=256}"
    : "${OVER_SAMPLING_BATCH_SIZE:=280}"
    : "${SAVE_INTERVAL:=10}"
    : "${EVAL_INTERVAL:=10}"
    : "${NUM_CRITIC_ONLY_STEPS:=2}"
    ;;
  72cp2-64tr)
    # 9 nodes on 72 GPU: 8 actor nodes (64 GPU) + 1 rollout node (8 GPU), hybrid GDN CP=2.
    #   64 train (TP2 x PP4 x CP2 => DP4, EP16), 12 layers/stage + 8 infer (1 x TP8)
    # THE MAXIMUM-HEADROOM 72-GPU CONFIG. Two independent levers vs the crashed run:
    #   (a) 48 -> 64 train GPUs cuts the expert optimizer 27.0 -> 20.25 GB. That term is
    #       P_exp*12/TRAIN_GPUS and cancels PP/EP/CP/DP exactly, so ONLY adding train GPUs
    #       can shrink it -- and it was the single largest consumer in the 89.5 GB crash.
    #   (b) even PP4 (12 layers/stage) instead of the uneven 12/16/16/4 split, which had
    #       loaded the middle stages with +12.25 GB.
    #   (c) CP2 halves the GDN activation peak (10.48 -> 5.24 GB at T=33287).
    # Projected peak ~73.8 GB vs 95 GB usable (+21 GB headroom), the widest available.
    # COST: only ONE inference engine, so rollout is ~3x slower than the 24-GPU split and
    # rollout already dominates step time. Chosen deliberately: correctness over speed.
    _EXPECTED_NODES=9;  _ACTOR_NODES=8; _ROLLOUT_GPUS=8;  _PP=4; _CP=2   # 64/(2*4*2)=DP4
    : "${GLOBAL_BATCH_SIZE:=256}"
    : "${ROLLOUT_BATCH_SIZE:=256}"
    : "${OVER_SAMPLING_BATCH_SIZE:=280}"
    : "${SAVE_INTERVAL:=10}"
    : "${EVAL_INTERVAL:=10}"
    : "${NUM_CRITIC_ONLY_STEPS:=2}"
    ;;
  72cp2-64tr-joint)
    # Same 64-train/8-infer CP2 topology as 72cp2-64tr, but NUM_CRITIC_ONLY_STEPS=0 so
    # rollout_id 0 is ALREADY the joint actor+critic step. Purpose: reach the
    # historically-fatal code path in ONE rollout (~1 h) instead of three (~4 h).
    # Use this to validate that CP=2 trains at all; switch back to the =2 preset for a
    # real run (a cold critic gives garbage advantages on the first steps).
    _EXPECTED_NODES=9;  _ACTOR_NODES=8; _ROLLOUT_GPUS=8;  _PP=4; _CP=2
    : "${GLOBAL_BATCH_SIZE:=256}"
    : "${ROLLOUT_BATCH_SIZE:=256}"
    : "${OVER_SAMPLING_BATCH_SIZE:=280}"
    : "${SAVE_INTERVAL:=10}"
    : "${EVAL_INTERVAL:=10}"
    : "${NUM_CRITIC_ONLY_STEPS:=0}"
    ;;
  88cp2)
    # 11 nodes / 88 GPU: 8 actor nodes (64 GPU) + 3 rollout nodes (24 GPU), hybrid GDN CP=2.
    #   64 train (TP2 x PP4 x CP2 => DP4, EP16), 12 layers/stage + 24 infer (3 x TP8)
    # THE 88-GPU WORKHORSE. 3 inference engines instead of 1 cuts rollout 58.9 -> 19.6 min
    # (measured single-engine rate 4.35 groups/min), so a bs256 step drops ~60 -> ~23 min.
    #
    # PP MUST stay 4 at CP2. PP2 gives DP8 (nicer gradient stats) but puts 24 layers on
    # every stage, and the last stage then needs 105.7 GB > 95 GB -> guaranteed OOM.
    # The binding term is the fp32 logits on the last stage (vocab 248320):
    #   local T = 65536/2 = 32768  ->  32768 x 248320/2 x 6B = 22.7 GB
    # PP4 keeps it to 89.3 GB (+5.7 GB headroom). Verified in practice: the CP2/PP4/64k
    # run on 2026-08-12 completed 9 joint actor+critic steps with peak allocated 17-19 GB
    # (real trajectories top out ~33k tokens, far below the 64k theoretical wall).
    #
    # CONTEXT WALL MUST STAY 64k HERE. At 84k the last stage is 100.3 GB -> OOM.
    # For an 84k wall use CP=4 instead (81.8 GB, +13.2 GB) -- see the header table.
    _EXPECTED_NODES=11; _ACTOR_NODES=8; _ROLLOUT_GPUS=24; _PP=4; _CP=2   # 64/(2*4*2)=DP4
    : "${GLOBAL_BATCH_SIZE:=256}"
    : "${ROLLOUT_BATCH_SIZE:=256}"
    : "${OVER_SAMPLING_BATCH_SIZE:=280}"
    : "${SAVE_INTERVAL:=10}"
    : "${EVAL_INTERVAL:=10}"
    : "${NUM_CRITIC_ONLY_STEPS:=2}"
    ;;
  88cp4-84k)
    # 11 nodes / 88 GPU: 8 actor nodes (64 GPU) + 3 rollout nodes (24 GPU), GDN CP=4.
    #   64 train (TP2 x PP4 x CP4 => DP2, EP16), 12 layers/stage + 24 infer (3 x TP8)
    #   84k training-token wall (vs 64k on the cp2 presets)
    #
    # WHY CP=4 HERE. The binding term is the fp32 logits on the LAST pipeline stage
    # (vocab 248320, bf16+fp32 co-resident at module.py:495). At an 84k wall a single
    # oversized trajectory owns its whole micro-batch, so local T = 86016/CP:
    #   CP2 -> local 43008 -> logits 29.8 GB -> last stage 100.3 GB  = OOM
    #   CP4 -> local 21504 -> logits 14.9 GB -> last stage  81.8 GB  = +13.2 GB OK
    # PP MUST be 4: PP2 leaves 24 layers/stage and lands at 96.4 GB (-1.4 GB) = OOM.
    #
    # CP=4 is measured, not assumed: forward BIT-EXACT vs single-rank reference,
    # grad max diff 1.2e-7, GDN activation 10.57 -> 2.70 GB (3.92x), relayout exact,
    # E2E varlen PASS (tests/test_qwen3_5_gdn_cp_*.py at --nproc_per_node=4).
    #
    # COST: DP 4 -> 2, so bs256 means 128 grad-accum micro-batches per rank and the
    # step time roughly doubles vs the cp2 presets. Deliberate: 84k without truncation.
    _EXPECTED_NODES=11; _ACTOR_NODES=8; _ROLLOUT_GPUS=24; _PP=4; _CP=4   # 64/(2*4*4)=DP2
    : "${SEQ_LENGTH:=86016}"
    : "${HARBOR_MAX_TRAIN_TOTAL_TOKENS:=86016}"
    : "${GLOBAL_BATCH_SIZE:=256}"
    : "${ROLLOUT_BATCH_SIZE:=256}"
    : "${OVER_SAMPLING_BATCH_SIZE:=280}"
    : "${SAVE_INTERVAL:=10}"
    : "${EVAL_INTERVAL:=10}"
    : "${NUM_CRITIC_ONLY_STEPS:=2}"
    ;;
  88cp8-128k)
    # 11 nodes / 88 GPU: 8 actor nodes (64 GPU) + 3 rollout nodes (24 GPU), GDN CP=8.
    #   64 train (TP2 x PP4 x CP8 => DP1, EP16), 12 layers/stage + 24 infer (3 x TP8)
    #   128k training-token wall -- the longest this cluster can host at all.
    #
    # THE ONLY LEGAL 128k CONFIG ON 64 TRAIN GPUs, and it is genuinely marginal:
    #   PP2 -> DP2, 24 layers/stage, last stage 97.8 GB  = OOM (-2.8 GB)
    #   PP4 -> DP1, 12 layers/stage, last stage 85.7 GB  = +9.3 GB, the only survivor
    # local T = 131072/8 = 16384, so logits = 16384 x 248320/2 x 6B = 11.4 GB.
    #
    # CP=8 is measured, not assumed: forward BIT-EXACT vs single-rank reference,
    # grad max diff 1.2e-7, GDN activation 10.25 -> 1.34 GB (7.67x), relayout exact,
    # E2E varlen PASS (tests/test_qwen3_5_gdn_cp_*.py at --nproc_per_node=8).
    #
    # ⚠ DP=1 IS THE REAL COST, not the memory:
    #   * all grad-accum micro-batches run serially on one DP rank -- bs128 means 128
    #     of them, so expect ~90+ min per step.
    #   * balance_data's Karmarkar-Karp partitioning degenerates to a single partition,
    #     so a long-tail sample can no longer be balanced away. That is exactly the
    #     straggler mechanism behind the 2026-08-11 NCCL-watchdog kill, so the Triton
    #     anti-straggler trio below matters even more here.
    # Prefer 88cp4-84k unless you specifically need a 128k wall.
    _EXPECTED_NODES=11; _ACTOR_NODES=8; _ROLLOUT_GPUS=24; _PP=4; _CP=8   # 64/(2*4*8)=DP1
    : "${SEQ_LENGTH:=131072}"
    : "${HARBOR_MAX_TRAIN_TOTAL_TOKENS:=131072}"
    : "${ROLLOUT_MAX_CONTEXT_LEN:=131072}"
    : "${SGLANG_CONTEXT_LENGTH:=131072}"
    : "${SGLANG_MAX_PREFILL_TOKENS:=131072}"
    : "${GLOBAL_BATCH_SIZE:=128}"
    : "${ROLLOUT_BATCH_SIZE:=128}"
    : "${OVER_SAMPLING_BATCH_SIZE:=144}"
    : "${SAVE_INTERVAL:=10}"
    : "${EVAL_INTERVAL:=10}"
    : "${NUM_CRITIC_ONLY_STEPS:=2}"
    ;;
  96)
    # 12 nodes: 8 actor nodes (64 GPU) + 4 rollout nodes (32 GPU)
    _EXPECTED_NODES=12; _ACTOR_NODES=8; _ROLLOUT_GPUS=32; _PP=4; _CP=1   # 64/(2*4)=DP8, 12 layers/stage
    ;;
  128)
    # 16 nodes: 12 actor nodes (96 GPU) + 4 rollout nodes (32 GPU)
    _EXPECTED_NODES=16; _ACTOR_NODES=12; _ROLLOUT_GPUS=32; _PP=6; _CP=1  # 96/(2*6)=DP8, 8 layers/stage
    ;;
  *)
    echo "ERROR: TOPOLOGY must be 72, 72cp2, 72cp2-bs256, 72cp2-64tr, 72cp2-64tr-joint, 88cp2, 88cp4-84k, 88cp8-128k, 96 or 128 (got '${TOPOLOGY}')." >&2
    echo "       For a custom size, set EXPECTED_NODES / ACTOR_NUM_NODES /" >&2
    echo "       ROLLOUT_NUM_GPUS / PIPELINE_MODEL_PARALLEL_SIZE explicitly and keep" >&2
    echo "       GLOBAL_BATCH_SIZE % DP == 0 with DP = actor_gpus/(TP*PP*CP)." >&2
    exit 1
    ;;
esac

# Defaulted AFTER the case block so a preset can override it (see note near LR above).
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}"

export EXPECTED_NODES="${EXPECTED_NODES:-${_EXPECTED_NODES}}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${_ACTOR_NODES}}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${_ROLLOUT_GPUS}}"

# TP2 is deliberate and should NOT be raised — see fact (3) in the header.
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-${_PP}}"
export CP_SIZE="${CP_SIZE:-${_CP}}"
# ---- CP for GatedDeltaNet: NATIVE (hybrid) CP, not the duplicate-compute fallback ----
# CP used to be pinned at 1 because GDN's recurrent scan is not CP-safe. That is now
# handled: slime_plugins/models/cp_utils.py detects GDN modules and wires fla.ops.cp
# state-passing CP into them, with a zigzag<->packed relayout in hf_attention.py to
# bridge slime's zig-zag CP layout to the contiguous shard fla expects.
# Verified on the real 122B GDN dims (tests/test_qwen3_5_gdn_cp_*.py, torchrun 2/4 ranks):
#   T=33792 and T=65536, CP=2 vs single-rank reference:
#     forward BIT-EXACT (max abs diff 0.0), grad max diff 1.2e-7
#     GDN activation 10.57 -> 5.34 GB (1.98x) and 20.50 -> 10.31 GB (1.99x)
# FLA_INTRACARD_CP=0 is REQUIRED: fla 0.4.2 auto-enables an alternate forward algorithm
# under inference_mode for sequences >= 24576 tokens, which changes numerics and would
# desync train-side logprobs from the SGLang rollout (SGLang vendors its own fla copy).
if [ "${CP_SIZE}" -gt 1 ]; then
  export FLA_INTRACARD_CP="${FLA_INTRACARD_CP:-0}"
fi
# EP kept maximal — lowering it doubles expert weights/grads for zero optimizer gain.
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-16}"
export EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export QKV_FORMAT="${QKV_FORMAT:-thd}"

# Rollout: independent TP8 engines (eval-proven; no dp-attention/EP).
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
export EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:---sglang-mamba-scheduler-strategy extra_buffer --sglang-disable-custom-all-reduce}"
# DeepEP OFF (alltoall dispatcher) — matches the 72-GPU nodeepep runs.
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---moe-token-dispatcher-type alltoall}"

# --- context walls: TRAINING token budget 64k ---------------------------------------
# SEQ_LENGTH bounds the worst-case single-sample activation, which is what actually
# decides OOM here (an oversized sample cannot be split by the dynamic packer). The
# headroom table in the header is computed at the full 64k wall.
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export HARBOR_MAX_TRAIN_TOTAL_TOKENS="${HARBOR_MAX_TRAIN_TOTAL_TOKENS:-65536}"
# compact
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-86016}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-86016}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-86016}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
# ★ 0.6, NOT the 0.75 default. At 0.75 SGLang locks ~71 GB into the KV static pool and the
# GatedDeltaNet/FLA long-sequence temporaries (l2norm_fwd_kernel, chunk_delta_h) then blow
# through what is left -- an OBSERVED crash on the pure-rollout nodes, not a theoretical
# one (claude_sessions/HOWTO_launch_122b_a10b_72gpu_training.md section 3.4).
# The cost of 0.6 is a smaller KV pool, i.e. SGLang queues more requests. That is a
# throughput hit; 0.75 is a crash. Prefer the throughput hit.
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"

# Dynamic batching ON. Restored to 8192 (from the 4096 experiment): per
# first_fit_pack's documented behaviour an oversized sample lands alone in its own bin
# at its FULL length, so 4096 could not bound the 40k-token sample that OOM'd — it only
# doubled the micro-batch count. The real bound now comes from the PP-reduced
# layers/stage, which lowers activation for EVERY sample including oversized ones.
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"

# --- batch / n / oversample / schedule ---------------------------------------------
# GLOBAL_BATCH_SIZE must stay divisible by DP (=8 at every TOPOLOGY) — dp_schedule.py:248
# asserts each step has >= dp_size samples, and the dynamic packer must reach a
# multiple of dp_size micro-batches (dp_schedule.py:264-271).
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export USE_DYNAMIC_GLOBAL_BATCH_SIZE="${USE_DYNAMIC_GLOBAL_BATCH_SIZE:-1}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-140}"
export HARBOR_LENGTH_REWARD_LAMBDA="${HARBOR_LENGTH_REWARD_LAMBDA:-0.0}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"

# --- eval: Terminal-Bench pass@3 ---------------------------------------------------
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-3}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
export EVAL_TOP_K="${EVAL_TOP_K:-20}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-tb2}"
export SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-1}"

# --- PPO knobs (identical to source recipe) ----------------------------------------
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-ppo}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export USE_CUSTOM_NO_ENTROPY_LOSS="${USE_CUSTOM_NO_ENTROPY_LOSS:-1}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
export USE_KL_LOSS="${USE_KL_LOSS:-0}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
export KL_COEF="${KL_COEF:-0.00}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
export KEEP_OLD_ACTOR="${KEEP_OLD_ACTOR:-1}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"
export ROLLOUT_ALL_SAMPLES_PROCESS_PATH="${ROLLOUT_ALL_SAMPLES_PROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples}"
export ROLLOUT_DATA_POSTPROCESS_PATH="${ROLLOUT_DATA_POSTPROCESS_PATH:-slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data}"

# --- allocator: MUST NOT use expandable_segments with PPO ---------------------------
# The crash log reported "4.33 GiB is reserved by PyTorch but unallocated", so
# expandable_segments:True looks tempting. IT IS INCOMPATIBLE and hard-fails at startup:
#   torch_memory_saver/entrypoint.py:199 _sanity_checks()
#     if "expandable_segments:True" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]:
#         raise RuntimeError("TorchMemorySaver is disabled ... not supported yet")
# PPO forces offload_train=True (arguments.py:1862), and actor.py:76-78 then sets
# torch_memory_saver.memory_margin_bytes, which triggers that sanity check on EVERY
# train actor -> job dies before the checkpoint even loads (observed 2026-08-06 17:32).
# Keep max_split_size_mb:256, the value the 32-GPU runs used. Fragmentation is instead
# addressed by the PP-reduced per-stage footprint.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

# --- CP>1 anti-straggler settings: the Triton cold-compile storm ---------------------
# ROOT CAUSE of the 2026-08-11 hang (72cp2-64tr, killed at 20:32): NOT a deadlock and
# NOT OOM. It was a Triton JIT COLD-COMPILE STRAGGLER exceeding the NCCL watchdog.
#
#   Evidence: the critic's per-microbatch times decayed 489s -> 216s -> 128s -> 88s
#   -> ~22s, the classic JIT warm-up curve. Variable-length terminal trajectories
#   (min ~800, max 35855 tokens) + dynamic batching + pad_size=TP*128=256 produce
#   ~60 DISTINCT padded thd shapes per step, and EVERY new shape triggers a fresh
#   GDN/Triton compile (tens..hundreds of seconds). CP=2 halves the per-rank length,
#   doubling the distinct-shape count.
#   When the actor started, ONE rank (30) sat in a >30 min compile while its 15
#   expert-parallel peers had already enqueued the MoE all-to-all. The 30-minute
#   watchdog fired on THEM; rank 30 never reached the collective so it never tripped
#   its own watchdog -- which is exactly why the log showed 15 timeouts and 1 silent
#   rank, and why 'actor forward' never advanced past 0/46.
#
# This failure mode was hit before on 27B CP2@96k and diagnosed then; the fixes live
# in run_tb2_full89_cp2_96k_dbgtrain_{padmult_tritoncache,newgroup_timeout}.sh.
# We apply all three mitigations here:
if [ "${CP_SIZE}" -gt 1 ]; then
  # (1) SHAPE CONVERGENCE. Coarser padding collapses ~60 distinct shapes into a
  #     handful, so the compile storm happens a few times instead of per-shape.
  #     Costs padding waste (<= mult tokens per microbatch); buys orders of
  #     magnitude fewer JIT compiles. 2048 was the value validated on 27B CP2.
  export DATA_PAD_SIZE_MULTIPLIER="${DATA_PAD_SIZE_MULTIPLIER:-2048}"
  # (2) PERSISTENT, NODE-LOCAL TRITON CACHE. Compiled kernels then survive across
  #     microbatches, steps, actor/critic switches AND process restarts, so only the
  #     very first run pays. /root is node-local overlayfs (NOT the shared fs), so
  #     there is no cross-node lock contention.
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.triton_cache_122b_cp${CP_SIZE}}"
  # (3) WATCHDOG HEADROOM. The compile window is legitimate work, not a hang. Both
  #     knobs matter: DISTRIBUTED_TIMEOUT_MINUTES only covers INITIAL group creation,
  #     while every offload reload rebuilds ~148 groups using the slime default
  #     (30 min, reloadable_process_group.py:27). Raise both.
  export SLIME_NCCL_GROUP_TIMEOUT_MINUTES="${SLIME_NCCL_GROUP_TIMEOUT_MINUTES:-120}"
  export SLIME_NEW_GROUP_TIMEOUT_MINUTES="${SLIME_NEW_GROUP_TIMEOUT_MINUTES:-120}"
  # (4) DIAGNOSTICS. If it still hangs, dump the NCCL flight recorder so the next
  #     investigation starts from a named process group instead of a 30-min mystery.
  export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
  export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
  export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-/root/nccl_trace_rank}"
fi
export USE_FAULT_TOLERANCE="${USE_FAULT_TOLERANCE:-1}"
export SLIME_GLOO_GROUP_TIMEOUT_MINUTES="${SLIME_GLOO_GROUP_TIMEOUT_MINUTES:-60}"

# --- data/timeout config yaml ------------------------------------------------------
# Default follows CP: the cp2 topology ships with the 3-hour-eval / bs256-concurrency
# yaml, the CP1 topologies keep the original 2-hour one. Override CONFIG to force either.
if [ -z "${CONFIG:-}" ]; then
  if [ "${CP_SIZE}" -gt 1 ]; then
    CONFIG="${SCRIPT_DIR}/harbor_tmax15k_72gpu_122b_a10b_daytona_ppo_cp2_bs256_evaltimeout10800.yaml"
  else
    CONFIG="${SCRIPT_DIR}/harbor_tmax15k_56gpu_122b_a10b_daytona_ppo_evaltimeout7200.yaml"
  fi
fi
export CONFIG

# --- cluster reuse defaults (this cluster is long-lived; see claude_sessions/HOWTO) --
# ★★ CLEAN_START MUST be 0 when reusing a running cluster. The base launcher defaults it
# to 1, which runs `ray stop --force` + `pkill raylet` BEFORE the head-reuse branch —
# and since the head lives on this box, that kills the head and drops all 8 workers.
export CLEAN_START="${CLEAN_START:-0}"
export USE_EXISTING_RAY_HEAD="${USE_EXISTING_RAY_HEAD:-1}"
# ★ MASTER_ADDR must be explicit: the launcher's default `hostname -I | awk '{print $1}'`
# picks 29.133.201.52 on this box (10 interfaces), but the real head is 29.127.33.153.
export MASTER_ADDR="${MASTER_ADDR:-29.127.33.153}"
export DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-180}"

# --- credentials -------------------------------------------------------------------
# Keys are NEVER stored in this file (it lives in a git worktree). They come from a
# 0600 env file, or from whatever is already exported.
export KEYS_FILE="${KEYS_FILE:-/tmp/bs512_keys.env}"
if [ -z "${DAYTONA_API_KEY:-}" ] && [ -f "${KEYS_FILE}" ]; then
  # shellcheck disable=SC1090
  set +u; source "${KEYS_FILE}"; set -u
fi
if [ -z "${DAYTONA_API_KEY:-}" ]; then
  echo "ERROR: DAYTONA_API_KEY is not set and ${KEYS_FILE} did not provide it." >&2
  echo "       Create it with:  umask 077; printf 'export DAYTONA_API_KEY=...\\nexport WANDB_API_KEY=...\\n' > ${KEYS_FILE}" >&2
  exit 1
fi

# --- run identity ------------------------------------------------------------------
# Each launch gets its own timestamped BASE_FOLDER so logs, stop-markers and trial
# artifacts never collide with a previous run. Override BASE_FOLDER to resume into an
# existing dir.
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export BASE_FOLDER="${BASE_FOLDER:-/path/to/workspace/slime_harbor_q35_122b_a10b_${TOPOLOGY}_bs${GLOBAL_BATCH_SIZE}_${RUN_TAG}}"
mkdir -p "${BASE_FOLDER}"
# ★ Stale stop-markers from a previously FAILED launch will stop the workers within
# seconds of start-up. Clear them for this run dir.
rm -f "${BASE_FOLDER}"/.stop_workers_* 2>/dev/null || true
export SAVE_SUBDIR="${SAVE_SUBDIR:-Qwen3.5-122B-A10B_tmax15k_${TOPOLOGY}gpu_ppo_criticlr30x_actorlr2x_strongest}"
# DP is derived, not always 8 (CP2 halves it) -- compute it so the run name is truthful.
_WANDB_DP=$(( ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE / (TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CP_SIZE) ))
export WANDB_GROUP="${WANDB_GROUP:-q35-122b-a10b-tmax15k-${TOPOLOGY}g-ppo-dp${_WANDB_DP}-ep16-pp${PIPELINE_MODEL_PARALLEL_SIZE}-cp${CP_SIZE}-nodeepep-bs${GLOBAL_BATCH_SIZE}}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-terminal-rl}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rtmax15k122b${TOPOLOGY}}"
export RAY_CLUSTER_WAIT_SECONDS="${RAY_CLUSTER_WAIT_SECONDS:-1800}"

export HARBOR_DAYTONA_S3_PROXY_URL="${HARBOR_DAYTONA_S3_PROXY_URL:-http://star-proxy.oa.com:3128}"
export HARBOR_CREATE_RATE_PER_SEC="${HARBOR_CREATE_RATE_PER_SEC:-6.0}"

# --- topology self-check (fail fast, before the 15-20 min init) --------------------
_TRAIN_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
_MP=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CP_SIZE))
if [ $((_TRAIN_GPUS % _MP)) -ne 0 ]; then
  echo "ERROR: train GPUs ${_TRAIN_GPUS} not divisible by TP*PP*CP=${_MP}." >&2; exit 1
fi
_DP=$((_TRAIN_GPUS / _MP))
_EPTOT=$((EXPERT_MODEL_PARALLEL_SIZE * EXPERT_TENSOR_PARALLEL_SIZE))
_ATTN_DP=$((TENSOR_MODEL_PARALLEL_SIZE * CP_SIZE * _DP))
if [ $((_ATTN_DP % _EPTOT)) -ne 0 ]; then
  echo "ERROR: EP*ETP=${_EPTOT} must divide TP*CP*DP=${_ATTN_DP}." >&2; exit 1
fi
if [ $((48 % PIPELINE_MODEL_PARALLEL_SIZE)) -ne 0 ]; then
  echo "ERROR: 48 layers not divisible by PP=${PIPELINE_MODEL_PARALLEL_SIZE}." >&2; exit 1
fi
if [ $(( (48 / PIPELINE_MODEL_PARALLEL_SIZE) % 4 )) -ne 0 ]; then
  echo "WARN: layers/stage $((48 / PIPELINE_MODEL_PARALLEL_SIZE)) is not a multiple of" >&2
  echo "      full_attention_interval=4; the [lin,lin,lin,full] pattern will split" >&2
  echo "      unevenly across pipeline stages (load imbalance, not incorrectness)." >&2
fi
if [ $((GLOBAL_BATCH_SIZE % _DP)) -ne 0 ]; then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} not divisible by DP=${_DP}." >&2; exit 1
fi
if [ $((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
  echo "ERROR: ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} not divisible by engine size" \
       "${ROLLOUT_NUM_GPUS_PER_ENGINE}." >&2; exit 1
fi
if [ $((_TRAIN_GPUS + ROLLOUT_NUM_GPUS)) -gt $((EXPECTED_NODES * 8)) ]; then
  echo "ERROR: train ${_TRAIN_GPUS} + infer ${ROLLOUT_NUM_GPUS} exceeds" \
       "$((EXPECTED_NODES * 8)) GPU on ${EXPECTED_NODES} nodes." >&2; exit 1
fi
# CP>1 needs fla.ops.cp (flash-linear-attention >= 0.4.2). Without it every GDN layer
# raises at its first forward, ~20 min into the run after the checkpoint load. Fail now.
if [ "${CP_SIZE}" -gt 1 ]; then
  if ! python3 -c "import fla.ops.cp" >/dev/null 2>&1; then
    echo "ERROR: CP_SIZE=${CP_SIZE} requires fla.ops.cp (flash-linear-attention >= 0.4.2)," >&2
    echo "       but 'import fla.ops.cp' failed. Installed: $(python3 -c 'import fla;print(fla.__version__)' 2>/dev/null || echo none)" >&2
    echo "       Install on EVERY node (site-packages is node-local):" >&2
    echo "         pip install -U 'fla-core==0.4.2' 'flash-linear-attention==0.4.2'" >&2
    exit 1
  fi
fi
echo "[topology] TOPOLOGY=${TOPOLOGY}: train=${_TRAIN_GPUS} (TP${TENSOR_MODEL_PARALLEL_SIZE}" \
     "x PP${PIPELINE_MODEL_PARALLEL_SIZE} x CP${CP_SIZE} => DP${_DP}, EP${EXPERT_MODEL_PARALLEL_SIZE})," \
     "$((48 / PIPELINE_MODEL_PARALLEL_SIZE)) layers/stage, infer=${ROLLOUT_NUM_GPUS}" \
     "($((ROLLOUT_NUM_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE)) x TP${ROLLOUT_NUM_GPUS_PER_ENGINE}), nodes=${EXPECTED_NODES}"
echo "[run]      base=${BASE_FOLDER}"
echo "[run]      batch=${GLOBAL_BATCH_SIZE} rollout=${ROLLOUT_BATCH_SIZE} oversample=${OVER_SAMPLING_BATCH_SIZE}" \
     "grad-accum-microbatches~=${GLOBAL_BATCH_SIZE}/${_DP}=$((GLOBAL_BATCH_SIZE / _DP)) per DP rank"
echo "[run]      save_interval=${SAVE_INTERVAL} eval_interval=${EVAL_INTERVAL}" \
     "critic_only_steps=${NUM_CRITIC_ONLY_STEPS} num_rollout=${NUM_ROLLOUT}"
echo "[run]      config=$(basename "${CONFIG}")  eval_total_timeout=$(grep -oP 'harbor_eval_total_timeout_sec:\s*\K[0-9]+' "${CONFIG}" || echo '?')s"
echo "[run]      cluster: CLEAN_START=${CLEAN_START} USE_EXISTING_RAY_HEAD=${USE_EXISTING_RAY_HEAD} MASTER_ADDR=${MASTER_ADDR}"
echo "[run]      wandb=${WANDB_GROUP}"

# --- dry-run escape hatch ----------------------------------------------------------
# DRY_RUN=1 prints the resolved plan and exits WITHOUT touching the cluster. Use it to
# sanity-check a topology/batch override before committing 48 GPUs to it.
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[dry-run] all pre-flight checks passed; exiting without launching."
  exit 0
fi

# ====================================================================================
# LAUNCH
# ====================================================================================
# BACKGROUND=1 (default) detaches via nohup and returns immediately, which is what you
# want for a many-hour run — the caller's shell can die without killing training.
# BACKGROUND=0 runs in the foreground (exec), for interactive debugging.
LOG_FILE="${LOG_FILE:-${BASE_FOLDER}/launch_122b_${TOPOLOGY}.log}"

if [ "${BACKGROUND:-1}" = "1" ]; then
  nohup "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh" > "${LOG_FILE}" 2>&1 &
  _PID=$!
  echo "[launch]   detached pid=${_PID}"
  echo "[launch]   log=${LOG_FILE}"
  cat <<WATCH

  # watch progress
  tail -f ${LOG_FILE}

  # milestones (expect: 5 'server is fired up' -> 3 'loaded checkpoint' -> rollout -> perf 0)
  L=${LOG_FILE}
  grep -ac 'successfully loaded checkpoint' "\$L"   # 3 = actor+critic loaded
  grep -aoE 'perf [0-9]+:' "\$L" | tail -3          # rollout completions
  grep -ac 'Timer update_weights end' "\$L"         # completed train steps
  grep -acE 'CUDA out of memory|torch.OutOfMemoryError' "\$L"   # MUST stay 0

  # NUM_CRITIC_ONLY_STEPS=${NUM_CRITIC_ONLY_STEPS} -> rollout_id ${NUM_CRITIC_ONLY_STEPS} is the first
  # actor+critic joint step. That is the step that OOM'd before; clearing it is the real test.

  # stop
  ray job list && ray job stop <job_id>
WATCH
  exit 0
fi

exec "${SCRIPT_DIR}/run_qwen35_9b_terminal_bench2_64h20.sh"
