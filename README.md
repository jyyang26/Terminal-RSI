<div align="center">

# Terminal-RSI

### Recursive Self-Improving Terminal Agents via Recursive Self-Rewrite and Policy-Refine

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2610.22192-b31b1b.svg)](https://arxiv.org/abs/2610.22192v1)
[![Project Page](https://img.shields.io/badge/Project-Page-0052D9.svg)](https://jyyang26.github.io/terminal-rsi)
[![Model Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-ffc107.svg)](https://huggingface.co/Teriminal-RSI)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

### T1: Terminal Agent Reinforcement Learning for Long-Horizon Tasks

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2610.22192-b31b1b.svg)](https://arxiv.org/abs/2609.11042)
[![Project Page](https://img.shields.io/badge/Project-Page-0052D9.svg)](https://jyyang26.github.io/t1/)
[![Model Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-ffc107.svg)](https://huggingface.co/Teriminal-RSI)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>



## 📰 News

- **[2026-10]** Code release: Terminal-RSI and RPR training recipes and framework.
- **[2026-10]** Code release: T1 training recipes and framework.

## 📖 Overview

Terminal agency is judged by persistent changes to an executable environment rather than by the plausibility of a response. Hundreds of tokens may express one action, thousands of observation tokens may encode no decision, and decisive feedback often arrives many turns later. Effective learning must therefore address both the **scarcity of reproducible successes** and the **alignment of optimization signals with terminal outcomes**.

**Terminal-RSI** alternates the adopted **Recursive Self-Rewrite (RSR)**, which turns harness-assisted successes into verified public demonstrations, with our proposed **Recursive Policy-Refine (RPR)**, a reinforcement-learning phase that couples a recursively handed-off critic with turn-aware credit assignment and phase-adaptive rewards:

- **Turn-Boundary Skip-Observation GAE (TB-GAE)** keeps token-specific value baselines but applies discount and trace decay only across interaction turns, so observation length no longer attenuates credit. Successor values *and* advantages are carried across computational chunks, so splitting a trajectory introduces no truncation error.
- **Family Dense Reward (FDR)** normalizes semantic-check progress by a per-family divisor, removing the up-to-795× reward-scale spread between task families.
- **Success-Prioritized Reward (SPR)** adds a strict-success offset and a saturating progress term, so an incomplete trajectory can never outrank a successful one by accumulating partial checks.
- **Recursive critic handoff.** The critic is carried unchanged through actor-only RSR stages and briefly recalibrated (four critic-only updates) at the start of every RPR phase.

Starting from Qwen3.8-27B, the sequence **Base → RSR-1 → RPR-1 → RSR-2 → RPR-2** yields **Terminal-RSI-27B** (RPR-2, step 18), which reaches **86.52%** on Terminal-Bench 2.1, **16.95%** on Terminal-Bench 3.0, **19.84%** on Terminal-Bench 4.0 and **79.00%** on Terminal-Bench Hard. Its **50.58%** four-suite macro-average is up from 20.68% at Base and 34.25% at RSR-1. Separately, rescoring 4,352 fixed trajectories with SPR reduces the failed trajectories' share of reward mass from **50.09%** to **28.72%**.

<div align="center">
<img src="assets/figure1_rsi_main_results.png" alt="Figure 1" width="100%"/>
<p><b>Figure 1. Retained-stage gains coexist with nonmonotonic exploration.</b> Positions denote Base, RSR-1, RPR-1 (step 20), RSR-2 and RPR-2 (step 18, Terminal-RSI-27B). These are stage positions, not equal compute intervals. Blue diamonds mark RSR, stars the retained RPR checkpoints, and gray branches unretained controls or continuations. LHTB shows completed tasks (left axis) and process reward (right axis).</p>
</div>

## 📊 Results

Performance of the five retained checkpoints marked in Figure 1, using the checkpoint scores from the accompanying results table. Terminal-Bench and SWR100 scores are success rates (%); LHTB reports average process reward, not a percentage.

| Checkpoint | Terminal-Bench 4.0 | Terminal-Bench 2.1 | Terminal-Bench 3.0 | Terminal-Bench Hard | LHTB (process reward) | SWR100 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Base (Qwen3.8-27B) | 0.00 | 51.70 | 0.00 | 31.00 | 0.21 | 3.00 |
| RSR-1 | 6.10 | 70.10 | 6.80 | 54.00 | 0.29 | 5.00 |
| RPR-1 (step 20) | 16.67 | 77.53 | 14.29 | 69.50 | 0.30 | 6.00 |
| RSR-2 | 17.46 | **86.52** | 15.00 | 76.00 | 0.31 | 6.00 |
| **Terminal-RSI-27B** (RPR-2, step 18) | **19.84** | **86.52** | **16.95** | **79.00** | **0.37** | **6.00** |

## 🗂️ Repository Structure

Terminal-RSI is built on [slime](https://github.com/THUDM/slime) (Megatron training + SGLang rollout over Ray). The agentic RL integration is a plugin layer, so slime core stays close to upstream.

```text
Terminal-RSI/
├── train_async.py                 # RL entry point: rollout of step t+1 overlaps training on step t
├── train.py                       # synchronous / eval-only entry point
├── examples/
│   ├── terminal_rsi/              # ★ Terminal-RSI recipes (start here)
│   │   ├── run_rpr1_fdr.sh        #   RPR-1 (Qwen3.8-27B): Family Dense Reward, actor lr 5e-7
│   │   ├── run_rpr2_spr.sh        #   RPR-2 (Qwen3.8-27B): Success-Prioritized Reward, actor lr 1e-7
│   │   ├── run_122b_terminal.sh   #   Qwen3.5-122B-A10B Terminal RL (MoE, 84k context)
│   │   ├── run_eval.sh            #   rollout-only evaluation on a Harbor task pool
│   │   ├── prepare_load_root.sh   #   actor / critic load-root pair for warm handoff
│   │   ├── common.sh              #   shared argument builder + Ray job submission
│   │   └── configs/               #   Harbor configs (FDR / SPR divisor tables, 122B), per-role optimizer configs
│   └── harbor_terminal_rl/        # Harbor runtime deps + HF -> torch_dist conversion (27B, 122B)
├── slime_plugins/
│   ├── harbor/                    # ★ Terminal-agent RL plugin
│   │   ├── rollout.py             #   Harbor/Terminus-2 agentic rollout (generate_rollout)
│   │   ├── data_source.py         #   task-pool data source with per-epoch shuffling
│   │   ├── partial_reward.py      #   FDR (test_count_progress_rewards) and SPR (success_smoothed_progress_rewards)
│   │   ├── tb_gae*.py             #   TB-GAE: turn-boundary advantages, chunk boundaries, termination
│   │   ├── atif_adapter.py        #   Harbor trial → token-exact training samples
│   │   ├── sample_postprocess.py  #   256k token caps and response trimming
│   │   ├── train_loss.py          #   PPO policy loss used by RPR
│   │   └── agent.py, llm.py, ...  #   Terminus-2 agent bound to the SGLang policy server
│   └── models/                    # Qwen3.5/3.8 (GatedDeltaNet hybrid) Megatron specs
├── slime/                         # training framework (Megatron backend, SGLang rollout, Ray orchestration)
│   ├── backends/megatron_utils/   #   actor/critic training, TB-GAE bootstrap values
│   ├── ray/                       #   placement groups, actor/critic groups, warm-critic loading
│   └── utils/                     #   arguments (incl. TB-GAE flags), PPO utilities
├── tools/                         # checkpoint conversion (torch_dist ↔ HF), reward analysis scripts
├── tests/                         # CPU unit tests (rewards, TB-GAE math, Harbor adapters) + GPU E2E tests
├── docker/  build_conda.sh        # environment
├── docs/                          # slime documentation and development notes
└── assets/                        # paper figures
```

## ⚙️ Installation

Terminal-RSI needs a CUDA build of SGLang and Megatron-LM. The most reliable path is the slime Docker image; a conda recipe pinning the same commits is also provided.

**Option A: Docker (recommended).**

```bash
docker pull slimerl/slime:latest           # see docker/ for the pinned tag and patches
docker run --gpus all --ipc=host --shm-size=64g -it slimerl/slime:latest bash
git clone https://github.com/jyyang26/Terminal-RSI.git && cd Terminal-RSI
pip install -e . --no-deps
```

**Option B: Conda / micromamba.**

```bash
git clone https://github.com/jyyang26/Terminal-RSI.git && cd Terminal-RSI
bash build_conda.sh                        # env "slime": Python 3.12, CUDA 12.9, PyTorch, pinned SGLang + Megatron-LM
micromamba activate slime
pip install -e . --no-deps
```

**Agent runtime.** Install [Harbor](https://github.com/harbor-framework/harbor) (the Terminal-Bench harness) and the context-parallel kernels for the GatedDeltaNet layers:

```bash
git clone https://github.com/harbor-framework/harbor.git /opt/harbor
export HARBOR_SRC=/opt/harbor/src                         # put on PYTHONPATH by the launchers
INSTALL_DAYTONA=1 bash examples/harbor_terminal_rl/install_harbor_runtime_deps.sh
pip install "fla-core>=0.4.2" "flash-linear-attention>=0.4.2"   # fla.ops.cp is required for CP>1
```

Rollouts execute in [Daytona](https://www.daytona.io/) sandboxes (`environment.type: daytona`), so export `DAYTONA_API_KEY`. Set `WANDB_API_KEY` for logging, or `USE_WANDB=0` to disable it.

**Sanity check (CPU only).**

```bash
python -m pytest tests/test_terminal_rsi_rewards.py tests/test_tb_gae_math.py tests/test_harbor_partial_reward.py -q
```

## 🚀 Quickstart

### 1. Prepare checkpoints and data

For HF-to-Megatron conversion, set `NPROC_PER_NODE`, `TENSOR_MODEL_PARALLEL_SIZE`,
`PIPELINE_MODEL_PARALLEL_SIZE`, and `MODEL_SCRIPT` for your environment as described
in [`examples/harbor_terminal_rl/README.md`](examples/harbor_terminal_rl/README.md).

```bash
# HF -> Megatron torch_dist (configure conversion resources for your environment)
HF_CKPT=/path/to/Qwen3.8-27B SAVE_DIR=/path/to/Qwen3.8-27B_torch_dist \
    bash examples/harbor_terminal_rl/convert_q38_27b_base_torch_dist.sh

export HF_CKPT=/path/to/Qwen3.8-27B
export TASKS_DIR=/path/to/swr_tasks              # Harbor tasks: <TASKS_DIR>/*/task.toml with CTRF-emitting verifiers
export EVAL_TASKS_DIR=/path/to/terminal-bench-2  # held-out pool:  <EVAL_TASKS_DIR>/*/task.toml
export HARBOR_SRC=/opt/harbor/src
export DAYTONA_API_KEY=...
```

FDR/SPR read the task family from `tests/private/manifest.json` (`family_id`) and the passed-check count from the verifier's `verifier/ctrf.json`. Tasks without a valid CTRF report fall back to the binary success reward.

### 2. RL training (RPR)

The training launchers require a running Ray cluster and an explicit resource configuration for your environment. Set `ACTOR_NUM_NODES`, `ACTOR_NUM_GPUS_PER_NODE`, `ROLLOUT_NUM_GPUS`, `ROLLOUT_NUM_GPUS_PER_ENGINE`, `TENSOR_MODEL_PARALLEL_SIZE`, `PIPELINE_MODEL_PARALLEL_SIZE`, and `CONTEXT_PARALLEL_SIZE` before launching; no experiment-specific resource defaults are provided. RPR uses a 256k context window and a batch of 256 trajectories per step. See [`examples/terminal_rsi/README.md`](examples/terminal_rsi/README.md) for resource requirements. Use `DRY_RUN=1` to inspect the generated command without launching training.

```bash
# RPR-1: RSR-1 actor + Warm-up Critic, Family Dense Reward
bash examples/terminal_rsi/prepare_load_root.sh \
     /ckpt/rsr1/iter_XXXXXXX /ckpt/warmup_critic/iter_XXXXXXX /ckpt/loadroot_rpr1
LOAD_ROOT=/ckpt/loadroot_rpr1 bash examples/terminal_rsi/run_rpr1_fdr.sh

# RPR-2: RSR-2 actor + critic retained from RPR-1, Success-Prioritized Reward
bash examples/terminal_rsi/prepare_load_root.sh \
     /ckpt/rsr2/iter_XXXXXXX /ckpt/rpr1/checkpoints_critic/iter_XXXXXXX /ckpt/loadroot_rpr2
LOAD_ROOT=/ckpt/loadroot_rpr2 bash examples/terminal_rsi/run_rpr2_spr.sh
```

| | RPR-1 | RPR-2 |
| --- | --- | --- |
| Actor init / critic init | RSR-1 / Warm-up Critic | RSR-2 / retained RPR-1 critic |
| Reward hook | `partial_reward.test_count_progress_rewards` (FDR) | `partial_reward.success_smoothed_progress_rewards` (SPR) |
| Harbor config | `configs/harbor_rpr1_fdr.yaml` | `configs/harbor_rpr2_spr.yaml` |
| Actor / critic lr | 5e-7 / 1.5e-5 | 1e-7 / 1.5e-5 |
| Critic-only calibration steps | 4 | 4 |
| PPO clip (low / high), value clip | 0.3 / 5.0, 4.0 | 0.3 / 5.0, 4.0 |
| Rollouts, batch, oversampling | 36, 256, 288 | 36, 256, 288 |
| Retained checkpoint | step 20 | step 18 (**Terminal-RSI-27B**) |

The **Warm-up Critic** uses the same launcher with a cold value head: build an actor-only root (`prepare_load_root.sh <actor> none <root>`) and run with `WARM_CRITIC=0 NUM_CRITIC_ONLY_STEPS=<N> NUM_ROLLOUT=<N>`, where `N` covers one epoch of the task pool. **TB-GAE** is enabled by default for RPR-1 in `configs/harbor_rpr1_fdr.yaml`, with `tb_gae_gamma_turn: 1.0` and `tb_gae_lambda_turn: 1.0`. Override the turn discounts with `EXTRA_ARGS="--tb-gae-gamma-turn <γ> --tb-gae-lambda-turn <λ>"`, or explicitly disable it with `EXTRA_ARGS="--no-enable-tb-gae"`. The **RSR** stages are supervised fine-tuning on verified rewritten trajectories (see slime's SFT recipes, e.g. `scripts/run-qwen3-4B-base-sft.sh`); the RSR data pipeline itself follows the RSR paper and is not part of this release.

**Qwen3.5-122B-A10B Terminal RL.** The same stack also trains the 256-expert MoE model at
84k context with batch 512, the CTRF progress reward, and rollout routing replay.
Configure the training and rollout resources for your environment, including
`EXPERT_MODEL_PARALLEL_SIZE` and `EXPERT_TENSOR_PARALLEL_SIZE` for MoE training:

```bash
HF_CKPT=/path/to/Qwen3.5-122B-A10B SAVE_DIR=/path/to/Qwen3.5-122B-A10B_torch_dist \
    bash examples/harbor_terminal_rl/convert_qwen35_122b_a10b_torch_dist.sh
bash examples/terminal_rsi/prepare_load_root.sh \
     /path/to/Qwen3.5-122B-A10B_torch_dist/iter_XXXXXXX none /ckpt/loadroot_122b
WARM_CRITIC=0 HF_CKPT=/path/to/Qwen3.5-122B-A10B LOAD_ROOT=/ckpt/loadroot_122b \
TASKS_DIR=/path/to/strategy15k/tasks bash examples/terminal_rsi/run_122b_terminal.sh
```

See [`examples/terminal_rsi/README.md`](examples/terminal_rsi/README.md) for all options (resume, dry run, TB-GAE, ...).

### 3. Evaluation

Training evaluates on `EVAL_TASKS_DIR` every 10 steps (3 samples per task, temperature 1.0, top-p 0.95, top-k 20). To evaluate a checkpoint standalone, convert it to HF format and run the rollout-only evaluator:

```bash
# Megatron torch_dist -> HuggingFace
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir /ckpt/rpr2/checkpoints/iter_0000018 \
    --output-dir /ckpt/terminal_rsi_27b_hf \
    --origin-hf-dir /path/to/Qwen3.8-27B --vocab-size 248320

# Rollout-only evaluation on any Harbor task pool (e.g. Terminal-Bench 2.x / 3.0 / 4.0 / Hard)
HF_CKPT=/ckpt/terminal_rsi_27b_hf EVAL_TASKS_DIR=/path/to/terminal-bench-2 \
    bash examples/terminal_rsi/run_eval.sh
```

### 4. Inference

The converted checkpoint is a standard HuggingFace model and can be served with SGLang, then driven by any Harbor agent (Terminus-2 in our experiments):

```bash
python -m sglang.launch_server --model-path /ckpt/terminal_rsi_27b_hf \
    --tp "${ROLLOUT_NUM_GPUS_PER_ENGINE}" --context-length 262144 --mamba-scheduler-strategy extra_buffer --port 30000
```

## 📝 Citation

If you find Terminal-RSI useful, please cite [our paper](https://arxiv.org/abs/2610.22192v1):

```bibtex
@misc{yang2026terminalrsi,
  title         = {Terminal-RSI: Recursive Self-Improving Terminal Agents via Recursive Self-Rewrite and Policy-Refine},
  author        = {Yang, Junyao and Li, Zongxia and Shi, Yucheng and Li, Zhongzhi and Wang, Ruhan and Zhou, Ruiwen and Mi, Haitao and Yan, Shuicheng and Liang, Leowei},
  year          = {2026},
  eprint        = {2610.22192},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.22192v1}
}
```

Terminal-RSI and [T1](https://arxiv.org/abs/2609.11042) share the same terminal-agent reinforcement-learning training framework. If you use this training framework, please also cite T1:

```bibtex
@misc{yang2026t1terminalagentreinforcement,
  title         = {T1: Terminal Agent Reinforcement Learning for Long-Horizon Tasks},
  author        = {Junyao Yang and Yucheng Shi and Zhongzhi Li and Ruhan Wang and Zongxia Li and Haitao Mi and Leowei Liang},
  year          = {2026},
  eprint        = {2609.11042},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.11042}
}
```

## 📄 License

This project is released under the [Apache License 2.0](LICENSE). It is derived from [slime](https://github.com/THUDM/slime), which is also Apache-2.0 licensed; third-party components keep their original licenses.

## 🙏 Acknowledgements

We sincerely thank the authors and contributors of [slime](https://github.com/THUDM/slime), [SGLang](https://github.com/sgl-project/sglang), and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for their open-source work. These projects provide the foundation for Terminal-RSI's reinforcement-learning framework, rollout inference, and distributed model training, respectively.

Terminal-RSI also builds on [Ray](https://github.com/ray-project/ray). Agent execution uses [Harbor](https://github.com/harbor-framework/harbor), [Terminal-Bench](https://github.com/laude-institute/terminal-bench), and [Daytona](https://www.daytona.io/) sandboxes. We follow the distributed terminal-RL infrastructure of T1 and adopt the Recursive Self-Rewrite procedure of RSR. We thank the authors and contributors of these projects as well.

## 📬 Contact

Junyao Yang: [junyaoyang@u.nus.edu](mailto:junyaoyang@u.nus.edu)
