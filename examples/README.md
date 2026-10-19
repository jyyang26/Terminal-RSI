# Examples

| Directory | Contents |
| --- | --- |
| [`terminal_rsi/`](./terminal_rsi) | The three training launchers: **RPR-1** (Qwen3.8-27B, FDR), **RPR-2** (Qwen3.8-27B, SPR) and **Qwen3.5-122B-A10B Terminal RL**, plus rollout-only evaluation, checkpoint load-root helper and all configs. |
| [`harbor_terminal_rl/`](./harbor_terminal_rl) | Environment and checkpoint utilities used by those launchers: Harbor runtime dependencies and HF → Megatron `torch_dist` conversion for both models. |

Start with [`terminal_rsi/README.md`](./terminal_rsi/README.md).
