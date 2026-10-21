# Harbor terminal-RL utilities

Helpers for the launchers in [`../terminal_rsi`](../terminal_rsi).

| Script | Purpose |
| --- | --- |
| `install_harbor_runtime_deps.sh` | Python runtime deps for running Harbor from source (`INSTALL_DAYTONA=1` also installs the Daytona SDK). |
| `convert_q38_27b_base_torch_dist.sh` | Qwen3.8-27B HF → Megatron `torch_dist` with user-configured resources. |
| `convert_qwen35_122b_a10b_torch_dist.sh` | Qwen3.5-122B-A10B HF → Megatron `torch_dist` with user-configured resources. |

```bash
INSTALL_DAYTONA=1 bash examples/harbor_terminal_rl/install_harbor_runtime_deps.sh
HF_CKPT=/path/to/Qwen3.8-27B       SAVE_DIR=/path/to/Qwen3.8-27B_torch_dist       bash examples/harbor_terminal_rl/convert_q38_27b_base_torch_dist.sh
HF_CKPT=/path/to/Qwen3.5-122B-A10B SAVE_DIR=/path/to/Qwen3.5-122B-A10B_torch_dist bash examples/harbor_terminal_rl/convert_qwen35_122b_a10b_torch_dist.sh
```

Before converting, set `NPROC_PER_NODE`, `TENSOR_MODEL_PARALLEL_SIZE`, and
`PIPELINE_MODEL_PARALLEL_SIZE` for your environment, and provide `MODEL_SCRIPT` pointing
to a shell file defining the matching model's `MODEL_ARGS` array. The conversion
scripts do not supply experiment-specific resource defaults.

A `torch_dist` checkpoint can be resharded on loading into a compatible training
layout. Both conversion and training layouts must satisfy the model and backend
constraints; tensor-parallel scatter compatibility should be checked for
GatedDeltaNet models.
