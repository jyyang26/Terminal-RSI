#!/usr/bin/env bash
# launch_27b_run2_warmcritic_sc.sh
#
# RUN2 recipe (27B dense Terminal PPO 256k, converted SFT iter263, WARM critic, original
# test_count_progress_rewards @ SCALE 82) + PPO Score-Centering (SC) turned ON.
#
# This is a thin launcher: it sets only what differs from launch_27b_run2_warmcritic_ctrf82.sh
# and then calls the same wrapper run_27b_dense_128gpu_ppo_256k.sh, so the proven RUN2
# topology / memory / proxy / keepalive behaviour is the single source of truth.
#
# =============================== WHAT SC ADDS vs RUN2 ===============================
# SC is an ACTOR-side loss addend layered on top of the unchanged PPO objective (see
# SC_IMPLEMENTATION_REPORT_20260929.md section 3.1). It does NOT change the reward, the
# advantage estimator, the critic, the data, or the topology. Concretely this launcher only:
#   (1) points MEGATRON_CONFIG_PATH at the SC-active per-role yaml (scale 1.0), and
#   (2) appends --no-enable-tb-gae to the load-side flags.
# Everything else is inherited from the RUN2 warm-critic path.
#
# WHY THIS IS RUNNABLE (unlike TB-GAE):
#   SC's upstream data contract is top-k logprobs, requested from SGLang as
#   top_logprobs_num=k with return_logprob=true (slime_plugins/harbor/llm.py:135-150) and
#   read back from meta_info.output_top_logprobs in json mode. That is a NATIVE SGLang
#   capability and was really GPU-smoke-tested on 2026-09-26 (single node, TP2xPP2,
#   sc_applied=1.0, sc_head_hit=1.0). Every SC validation gate in
#   slime/utils/score_centering_validate.py passes against this exact RUN2 argv (verified:
#   ppo advantages, custom_loss on the audited Harbor mount, trajectory_retokenized,
#   temperature/top_p/top_k = 1/1/-1, eps_clip 0.3<1, keep_old_actor, permitted cap hooks).
#
# ================================ SC PARAMETERS: SOURCE ================================
# All SC numeric params come from ONE source only: the score_centering: block in the
# SC-active role yaml. Per the report's guidance ("建议每次只选择一个主要 SC 参数来源"),
# nothing SC-numeric is passed on the CLI here, to avoid the CLI>role-yaml>custom-yaml
# priority interacting confusingly. The role yaml sets:
#   enable_score_centering: true, sc_correction_scale: 1.0 (REAL SC, not shadow),
#   sc_formula: score, sc_wire_format: json (NOT the 'binary' default), sc_top_k: 32.
# --no-enable-tb-gae is the one SC/TB-related CLI flag, and it belongs to the separate
# TB-GAE switch system; it is belt-and-braces in case an inherited YAML ever set TB on.
#
# ================================ CRITIC WARMUP ================================
# NUM_CRITIC_ONLY_STEPS defaults to 2 (RUN2). During critic-only steps the actor does NOT
# train, so NO SC loss/metrics appear yet: sc_applied first shows up at rollout_id >=
# NUM_CRITIC_ONLY_STEPS. To see SC from the very first step (as the 2026-09-26 smoke did),
# set NUM_CRITIC_ONLY_STEPS=0 -- but that is a training decision, not an SC requirement.
#
# ================================ HOW TO CONFIRM SC IS LIVE ================================
# After launch, grep the log (report section 7 -- four layers of evidence):
#   [preflight]/args : enable_score_centering true, sc_formula score, sc_correction_scale 1.0,
#                      sc_wire_format json ; critic side false
#   per Actor step   : train/sc_applied = 1.0  (plus sc_head_hit / sc_head_logp_deviation /
#                      sc_correction). sc_applied=1.0 alone only proves the FORWARD mount ran;
#                      a finite grad_norm + successful optimizer.step proves the backward.
#
# Usage (multi-node cluster, RUN2 prerequisites -- load roots, data pool, auth -- must exist):
#   bash examples/harbor_terminal_rl/launch_27b_run2_warmcritic_sc.sh
#   NUM_ROLLOUT=40 NUM_CRITIC_ONLY_STEPS=0 bash examples/harbor_terminal_rl/launch_27b_run2_warmcritic_sc.sh
#
# To resume SC from a different start instead of the RUN2 converted-iter263 load root,
# override PRETRAINED_CKPT (+ its _critic sibling) and START_ROLLOUT_ID accordingly.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_ROOT=/path/to/workspace

# ---- RUN2 LOAD ROOT: converted SFT iter263 + warm critic sibling --------------------
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${USER_ROOT}/checkpoints/q38_27b_ppo256k_run2_loadroot}"
export WARM_CRITIC="${WARM_CRITIC:-1}"
export START_ROLLOUT_ID="${START_ROLLOUT_ID:-0}"

# ---- CRITIC WARMUP (RUN2 default) ---------------------------------------------------
export NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}"

# ---- REWARD: RUN2 original (unchanged by SC) ----------------------------------------
export CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/harbor_mixed3000_128gpu_27b_dense_daytona_ppo_cp4_256k_run2_ctrf82.yaml}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime_plugins.harbor.partial_reward.test_count_progress_rewards}"
unset HARBOR_REWARD_MODE
export HARBOR_TESTCOUNT_SCALE="${HARBOR_TESTCOUNT_SCALE:-82}"

# ---- SC: the ONLY behavioural change. Per-role yaml carries the whole SC block. ------
export MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${SCRIPT_DIR}/megatron_roles_27b_criticlr30x_warmcritic_sc_active.yaml}"

# ---- LOAD-SIDE FLAGS: RUN2 fresh-optimizer + explicit TB-GAE off ---------------------
# RUN2 loads a converted SFT checkpoint with no compatible optimizer state, so the
# optimizer/RNG are NOT loaded (they are still built fresh and SAVED -- NO_SAVE_OPTIM stays 0).
# --no-enable-tb-gae is appended so no inherited YAML can flip TB-GAE on under SC.
export EXTRA_MISC_ARGS="${EXTRA_MISC_ARGS:---no-load-optim --no-load-rng --no-enable-tb-gae}"

# ---- PPO clipping: unchanged from RUN2 ----------------------------------------------
export EPS_CLIP="${EPS_CLIP:-0.3}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-5.0}"

# ---- TOPOLOGY: 64 train + 56 infer on the 15 surviving nodes ------------------------
export NODE_IP_LIST="${NODE_IP_LIST:-$(cat /tmp/node_ip_list_15.txt 2>/dev/null || echo "")}"
export EXPECTED_NODES="${EXPECTED_NODES:-15}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-56}"

# ---- RUN LENGTH + IDENTITY ----------------------------------------------------------
export NUM_ROLLOUT="${NUM_ROLLOUT:-40}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export RUN_TAG="${RUN_TAG:-RUN2_SC_$(date +%Y%m%d_%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-terminal3_rl}"
export WANDB_GROUP="${WANDB_GROUP:-q38-27b-dense-ppo-run2-warmcritic-sc}"

echo "[sc] ============ RUN2 WARM-CRITIC + SCORE-CENTERING (scale 1.0) ============"
echo "[sc] load root    = ${PRETRAINED_CKPT} (iter $(cat "${PRETRAINED_CKPT}/latest_checkpointed_iteration.txt" 2>/dev/null)) + _critic sibling"
echo "[sc] warm critic  = ${WARM_CRITIC}; critic-only steps = ${NUM_CRITIC_ONLY_STEPS} (SC metrics appear at rollout_id >= this)"
echo "[sc] reward       = RUN2 original: passed / SCALE(family), scale ${HARBOR_TESTCOUNT_SCALE} (unchanged by SC)"
echo "[sc] SC role cfg  = ${MEGATRON_CONFIG_PATH}"
echo "[sc] SC params    = score / json / k32 / scale 1.0  (from the role yaml's score_centering block)"
echo "[sc] TB-GAE       = OFF (--no-enable-tb-gae; mutually exclusive with SC)"
echo "[sc] misc flags   = ${EXTRA_MISC_ARGS}"
echo "[sc] num_rollout  = ${NUM_ROLLOUT} (END INDEX => ${NUM_ROLLOUT} steps: 0..$(( NUM_ROLLOUT - 1 )))"
echo "[sc] save_interval= ${SAVE_INTERVAL} (~1059 GB per save, NO checkpoint rotation)"
echo "[sc] confirm SC   : grep 'sc_applied' in the run log; expect 1.0 once actor trains"
echo "[sc] ======================================================================"

exec bash "${SCRIPT_DIR}/run_27b_dense_128gpu_ppo_256k.sh"
