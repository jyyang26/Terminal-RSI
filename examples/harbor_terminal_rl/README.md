# Harbor Terminus RL Rollout

This example wires Harbor Terminus 2 trials into slime through extension hooks.
It does not require changes to slime core or Harbor core.

For the current implementation and validation status, see
[`INTEGRATION_SUMMARY.md`](./INTEGRATION_SUMMARY.md).

Minimal slime arguments:

```bash
export CODE_ROOT=/path/to/workspace/code
export PYTHONPATH=$CODE_ROOT/harbor/src:$CODE_ROOT/slime:$PYTHONPATH

python train.py \
  --rollout-function-path slime_plugins.harbor.rollout.generate_rollout \
  --data-source-path slime_plugins.harbor.data_source.HarborTaskDataSource \
  --custom-config-path examples/harbor_terminal_rl/harbor_slime_config.yaml
```

For the current single-node Qwen3.5-4B probe, use the helper script:

```bash
cd /path/to/workspace/code/slime
DEBUG_ROLLOUT_ONLY=1 \
NUM_ROLLOUT=1 \
CONFIG=$PWD/examples/harbor_terminal_rl/harbor_slime_config.yaml \
BASE_FOLDER=/path/to/workspace/slime_harbor_qwen35_4b \
bash examples/harbor_terminal_rl/run_qwen35_4b_single_node.sh
```

The script defaults to:

- code root: `/path/to/workspace/code`
- HF checkpoint: `/path/to/models/Qwen3.5-4B`
- model script: `scripts/models/qwen3.5-4B.sh`
- one node, eight GPUs, colocated Megatron actor and SGLang rollout engines
- one SGLang engine across 8 GPUs by default:
  `ROLLOUT_NUM_GPUS_PER_ENGINE=8`
- smoke-sized rollout: `NUM_ROLLOUT=2`, `ROLLOUT_BATCH_SIZE=1`,
  `N_SAMPLES_PER_PROMPT=2`
- `DEBUG_ROLLOUT_ONLY=1` can be used for the first model smoke so rollout data
  is dumped without running actor train/update/save.

Before starting the model, validate the Apptainer task layer with the NOP smoke:

```bash
cd /tmp
PYTHONPATH=/path/to/workspace/code/slime:/path/to/workspace/code/harbor/src:$PYTHONPATH \
python /path/to/workspace/code/slime/examples/harbor_terminal_rl/smoke_stable_nop.py \
  --stable-tasks-path /path/to/workspace/code/stable_rl_tb_rl/data/train.parquet \
  --materialized-tasks-dir /tmp/slime-harbor-stable-tasks-smoke \
  --sif-cache-dir /path/to/workspace/code/stable_rl_tb_rl/data/sif_cache \
  --trials-dir /tmp/slime-harbor-trials-smoke
```

Latest verified NOP smoke on the launcher reached Harbor verifier successfully:
`exception_info=null`, `reward=0.0` as expected for a no-op agent.

Directly loading checkpoints from `/gy_1` is very slow on the probed node and
can look like SGLang is hanging at `Multi-thread loading shards`. Copying the
model to local node storage fixed this for the standard Qwen3-4B checkpoint:

```bash
rsync -a /path/to/models/Qwen3-4B/ /tmp/slime-harbor-model-cache/Qwen3-4B/
```

The latest verified model smoke used `/tmp/slime-harbor-model-cache/Qwen3-4B`,
`DEBUG_ROLLOUT_ONLY=1`, `NUM_ROLLOUT=1`, `N_SAMPLES_PER_PROMPT=1`, and
`ROLLOUT_MAX_RESPONSE_LEN=128`. It completed SGLang generation through Harbor
and saved `/tmp/slime_harbor_qwen3_4b_rollout_smoke_after_tokenfix/rollout_dumps/0.pt`
with 20 trainable samples. Reward was `0.0` because the base Qwen3-4B output did
not produce valid Terminus JSON in this smoke.

To try a standard Qwen3 checkpoint instead:

```bash
cd /path/to/workspace/code/slime
DEBUG_ROLLOUT_ONLY=1 \
NUM_ROLLOUT=1 \
N_SAMPLES_PER_PROMPT=1 \
GLOBAL_BATCH_SIZE=1 \
ROLLOUT_MAX_RESPONSE_LEN=128 \
HF_CKPT=/tmp/slime-harbor-model-cache/Qwen3-4B \
MODEL_SCRIPT=$PWD/scripts/models/qwen3-4B.sh \
MEGATRON_TO_HF_MODE= \
QKV_FORMAT= \
SAVE_SUBDIR=Qwen3-4B_harbor_slime \
BASE_FOLDER=/tmp/slime_harbor_qwen3_4b_rollout_smoke \
CONFIG=$PWD/examples/harbor_terminal_rl/harbor_slime_config.yaml \
bash examples/harbor_terminal_rl/run_qwen35_4b_single_node.sh
```

## Jizhi Path Notes

On the current Taiji/Jizhi job, `/path/to/shared_storage` is a stale
mount on the remote nodes. Use `/path/to/shared_storage` inside the job instead.

Before launching a multi-node run, verify every selected node can see the same
code tree:

```bash
ls /path/to/workspace/code/slime/slime_plugins/harbor
```

In the probed job, the launcher can see `user/<user>/code`, while worker-0
only sees `models` and `_tmp_bench` under `user/<user>`. That mount mismatch
must be fixed before running distributed slime training from the shared tree.

If the cluster only provides `apptainer` but Harbor's built-in HPC environment
invokes `singularity`, add a shim before launching slime:

```bash
mkdir -p /tmp/slime-harbor-bin
ln -sf "$(command -v apptainer)" /tmp/slime-harbor-bin/singularity
export PATH=/tmp/slime-harbor-bin:$PATH
```

On the current launcher node, `apptainer 1.5.0` is installed and the
`singularity` command resolves to Apptainer. `root` also needs subordinate
uid/gid mappings for `--fakeroot`; the tested values are
`root:100000:65536` in both `/etc/subuid` and `/etc/subgid`.

For a fresh Ubuntu launcher pod, the same setup is captured in:

```bash
APT_PROXY=http://star-proxy.oa.com:3128 \
SIF_SMOKE_TEST=/path/to/workspace/code/stable_rl_tb_rl/data/sif_cache/ubuntu-24-04_latest.sif \
bash examples/harbor_terminal_rl/install_apptainer_ubuntu.sh
```

The current Jizhi pod cannot mount a fresh `/proc` inside the container under
Harbor's `--containall --pid` execution mode. Keep the proc mount suppressed,
then bind host `/proc` back in so tmux can resolve process state:

```yaml
singularity_no_mount: home,tmp,bind-paths,proc
singularity_bind_host_proc: true
```

For offline clusters, set the Harbor task `[environment].docker_image` to a
pre-built `.sif` path. The built-in Singularity/Apptainer environment can run
`.sif` files without a Docker daemon.

## Sample Modes

- `turn`: one slime `Sample` per LLM call. This is the most robust mode because
  each sample is exactly `prompt_token_ids[i] + completion_token_ids[i]`.
- `last_turn`: only train on the final LLM call from each Harbor trial.
- `trajectory`: one slime `Sample` per ATIF rollout detail. Model completion
  tokens get `loss_mask=1`; environment observation deltas get `loss_mask=0`.
  This mode requires adjacent Harbor prompt token histories to be strict prefix
  extensions.
- `trajectory_normalized`: opt-in trajectory mode for Terminus2/Qwen template
  boundaries. It reuses trajectory context when adjacent prompts align after
  small prompt-suffix or response-edge normalization, and starts a new chunk when
  a transition still does not align. Chunks from the same trial share the same
  trajectory group id. Metadata marks whether the chunk is an approximate
  conditioning match.
- `trajectory_retokenized`: trajectory mode for Terminus2/Qwen runs where the
  next prompt replays the same assistant text after re-tokenizing message
  history. The rollout LLM records completion text plus the previous assistant
  message span in the next prompt; conversion merges only when those texts match
  exactly, skips the re-tokenized history span, and keeps later context tokens
  masked with `loss_mask=0`. Content rewrites still split.
