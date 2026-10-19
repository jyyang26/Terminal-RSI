# Harbor terminal-RL utilities

Helpers for the launchers in [`../terminal_rsi`](../terminal_rsi).

| Script | Purpose |
| --- | --- |
| `install_harbor_runtime_deps.sh` | Python runtime deps for running Harbor from source (`INSTALL_DAYTONA=1` also installs the Daytona SDK). |
| `convert_q38_27b_base_torch_dist.sh` | Qwen3.8-27B HF → Megatron `torch_dist` (TP1 × PP8 on one 8-GPU node). |
| `convert_qwen35_122b_a10b_torch_dist.sh` | Qwen3.5-122B-A10B HF → Megatron `torch_dist` (TP1 × PP8). |

```bash
INSTALL_DAYTONA=1 bash examples/harbor_terminal_rl/install_harbor_runtime_deps.sh
HF_CKPT=/path/to/Qwen3.8-27B       SAVE_DIR=/path/to/Qwen3.8-27B_torch_dist       bash examples/harbor_terminal_rl/convert_q38_27b_base_torch_dist.sh
HF_CKPT=/path/to/Qwen3.5-122B-A10B SAVE_DIR=/path/to/Qwen3.5-122B-A10B_torch_dist bash examples/harbor_terminal_rl/convert_qwen35_122b_a10b_torch_dist.sh
```

A `torch_dist` checkpoint reloads at any TP/PP/CP layout, so converting with TP1 × PP8 is
fine for the training layouts used by the launchers.
