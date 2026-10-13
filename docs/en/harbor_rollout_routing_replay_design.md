# Rollout Routing Replay for the Harbor Terminal-Agent Path (Qwen3.5-35B-A3B MoE)

> Status: **design / not yet implemented**. This document is the result of a
> read-only investigation of how `--use-rollout-routing-replay` would have to be
> wired into the Harbor (Terminus2) rollout path for the Qwen3.5-35B-A3B MoE run.
> It records the verified data-path facts, the alignment contract, the two
> candidate approaches (and why one is wrong for fully-async), and a concrete
> implementation + validation plan.
>
> Scope: `slime_plugins/harbor/`, `slime/agent/trajectory.py`, the training-side
> consumer in `slime/backends/megatron_utils/actor.py`, and the SGLang 0.5.10
> `/generate` contract. Target run: the 96-GPU fully-async trajectory_retokenized
> recipe in `examples/harbor_terminal_rl/run_full89_grpo_96gpu_35b_a3b_*.sh`.

---

## 1. Background — what routing replay is and why we want it

`--use-rollout-routing-replay` implements the rollout routing-replay technique
(paper arXiv:2510.11370). For an MoE model the router's top-k expert selection is
discrete and non-differentiable; between rollout (inference on the SGLang engine)
and training (the Megatron actor forward/backward), the **same tokens can route to
different experts** because of numerical drift, kernel differences, or — under
async training — a weight update in between. That mismatch makes the importance
ratio `π_train / π_rollout` noisy precisely on MoE layers.

Routing replay fixes the routed experts: it records the expert indices the rollout
engine actually used, then **forces the training forward (and backward) to reuse
those same indices** instead of recomputing top-k. The actor only recomputes the
gate *probabilities* on the recorded indices (`scores.gather(1, top_indices)`),
keeping the path consistent and the gradient well-defined.

The mechanism on the training side lives in `slime/utils/routing_replay.py`:

- `RoutingReplay.record(top_indices)` stores per-layer expert indices (CPU pinned).
- `get_routing_replay_compute_topk` wraps the MoE `compute_topk`. Driven by env
  `ENABLE_ROUTING_REPLAY=1` + `ROUTING_REPLAY_STAGE ∈ {record, replay_forward,
  replay_backward, fallthrough}`, the `replay_*` stages **pop the recorded
  indices** and gather scores on them instead of recomputing argmax-topk.
- `register_routing_replay(module)` attaches a `RoutingReplay` per MoE layer via a
  forward pre-hook.

---

## 2. The native (non-Harbor) data path — the reference implementation

In the stock slime rollout (`slime/rollout/sglang_rollout.py`), routing replay is a
straight pass-through:

1. **Request** (`sglang_rollout.py:179-180`): when `args.use_rollout_routing_replay`,
   add `payload["return_routed_experts"] = True`.
2. **Response → Sample** (`sglang_rollout.py:224-232`):
   ```python
   if "routed_experts" in output["meta_info"]:
       sample.rollout_routed_experts = np.frombuffer(
           pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
           dtype=np.int32,
       ).reshape(
           len(sample.tokens) - 1,   # full sequence minus one
           args.num_layers,
           args.moe_router_topk,
       )
   ```
   The streaming rollout does the same at `sglang_streaming_rollout.py:77-78,156`.
3. **Sample → train_data** (`slime/ray/rollout.py:1012-1013`):
   ```python
   if samples[0].rollout_routed_experts is not None:
       train_data["rollout_routed_experts"] = [s.rollout_routed_experts for s in samples]
   ```
4. **train_data → tensors** (`slime/backends/megatron_utils/actor.py:283-286`):
   `np.ndarray` per sample → `torch.from_numpy`.
5. **Consumed** (`actor.py:295-366`, `fill_routing_replay`): asserts presence,
   then per-microbatch reshapes `[seq, layer, topk]`, pads to the last token,
   CP-slices, SP-slices, and records into the per-layer `RoutingReplay` buffers
   (skipping dense layers via `config.moe_layer_freq`).

**Hard contract (`actor.py:296-330`):**

- If `use_rollout_routing_replay` is set but `rollout_routed_experts` is absent
  from `rollout_data` → `raise ValueError`. **The training step crashes on step 1.**
- `assert len(rollout_routed_experts) == len(tokens)` (one array per sample).
- `assert a.shape[0] == b.shape[0] - 1` — each sample's routing array length must
  equal `len(sample.tokens) - 1`. **The array spans the full sequence
  (prompt + response) minus the last token**, NOT response-only.

This last point is the crux: `rollout_routed_experts` is aligned to the **full
token axis**, whereas `sample.rollout_log_probs` is **response-only**. The two are
NOT the same length and cannot be threaded identically.

---

## 3. Why Harbor has no data path today

The Harbor path does **not** use `sglang_rollout.py`. The Terminus2 agent talks to
the policy through a custom LLM backend (`slime_plugins/harbor/llm.py:SlimeSGLangLLM`)
over HTTP, and trial results are converted to `Sample`s by
`slime_plugins/harbor/atif_adapter.py`. Confirmed by grep: **there is zero
`routed_experts` handling anywhere under `slime_plugins/harbor/`**.

Consequence: every Harbor `Sample.rollout_routed_experts` stays `None`
(the field exists — `slime/utils/types.py:35` — but is never populated) →
`rollout_data` never gets `rollout_routed_experts` →
`actor.py:298 raise ValueError` on the **first training step**.

**Therefore the flag cannot just be added to the shell script. It requires
plugin-code changes to build the data path.**

---

## 4. Verified facts (read from source)

### 4.1 SGLang 0.5.10 `/generate` contract

- **Server launch:** `enable_return_routed_experts` (`srt/server_args.py:663`).
  slime sets this automatically when `args.use_rollout_routing_replay`
  (`slime/backends/sglang_utils/sglang_engine.py:605-606`:
  `kwargs["enable_return_routed_experts"] = True`).
- **Per-request key:** top-level `return_routed_experts: true`
  (`srt/managers/io_struct.py:171`), **NOT** inside `sampling_params`.
- **Response:** `meta_info["routed_experts"]`, base64-encoded `int32`.
- **Coverage:** **full sequence** — io_struct comment: *"The routed experts for
  each token, including both input and output tokens … shape (token, layer,
  top_k)."* Matches the native reshape to `[len(tokens)-1, L, topk]`.
- **Model:** capture is generic in the MoE top-k layer; `qwen3_5_moe`
  (`srt/models/qwen3_5.py:Qwen3_5MoeForCausalLM`) is covered with no per-model opt-in.

### 4.2 Model shape (Qwen3.5-35B-A3B)

- `num_layers = 40`, **all MoE** (`moe_layer_freq = [1]*40`), `num_experts = 256`,
  `moe_router_topk = 8`. So each sample's routing array is `[len(tokens)-1, 40, 8]`
  `int32`. `fill_routing_replay`'s dense-layer skip (`actor.py:355-362`) is a no-op
  here (no dense layers).

### 4.3 Injection point for runtime args → agent

`slime_plugins/harbor/rollout.py:_build_trial_config` (lines 872-928) is the
**single** place (used by both the standard and fully-async paths via the shared
`_run_one_group`, `rollout.py:677`) where `args` is poured into the agent's
`llm_kwargs`:

```python
llm_kwargs.setdefault("hf_checkpoint", args.hf_checkpoint)
llm_kwargs.setdefault("router_url", _router_url(args))
llm_kwargs.setdefault("max_new_tokens", getattr(args, "rollout_max_response_len", 1024))
if (max_context_tokens := getattr(args, "rollout_max_context_len", None)) is not None:
    llm_kwargs.setdefault("max_context_tokens", max_context_tokens)
```

This is where the routing-replay flag + `num_layers` + `moe_router_topk` must be
threaded in (trial workers run in subprocesses and have no `args` handle).

### 4.4 Read channel: per-turn `extra`

The LLM backend returns a Harbor `LLMResponse` whose `extra` dict is recorded
per-turn into `rollout_details[t]["extra"]`. The adapter reads it via
`_detail_extra_list(detail, key)` (`atif_adapter.py:991-996`) — the same channel
already used for `completion_text`, `last_assistant_message_content`,
`last_assistant_message_span` (`atif_adapter.py:658-660`). routed_experts can ride
the same channel, one base64 string per turn.

### 4.5 Reconstruction: how tokens are assembled (the alignment problem)

`trajectory_retokenized` does NOT use `merge_turns`. It uses its own chunk loop in
`_normalized_trajectory_samples_from_detail` (`atif_adapter.py:640-826`). Per turn:

- inter-turn `context_ids` (the new non-model prompt suffix) → `loss_mask 0`,
  `rollout_log_probs 0.0` (`atif_adapter.py:723-726`);
- `turn.output_ids` → `loss_mask 1`, real logprobs (`atif_adapter.py:746-748`);
- a `split` transition starts a fresh chunk (`atif_adapter.py:710-721`).

Final per chunk (`atif_adapter.py:757-768`):
`sample.tokens = prompt_ids + response_ids`,
`sample.response_length = len(response_ids)`,
and the invariant `len(rollout_log_probs) == response_length`.

The transition strategies (`_normalized_transition_alignment`,
`atif_adapter.py:860-942`) are `strict_prefix` / `normalized_prefix` /
`retokenized_prefix` / `split`. The `normalized`/`retokenized` strategies can
**drop** prompt-suffix / response edge tokens, so the per-turn full-sequence
routing array does not map 1:1 onto the rebuilt chunk tokens — alignment must be
reconstructed, not copied.

---

## 5. Two approaches — and why fully-async forces approach A

### Approach B — post-hoc rebuild (rejected for this run)

After a trajectory finishes, re-send the rebuilt `sample.tokens` as a single
prefill-only request with `return_routed_experts=true`, take the full-sequence
routing back, store `[:-1]`.

- ✅ Alignment is trivially correct (one request, one contiguous sequence).
- ✅ Tiny code change, isolated to the adapter.
- ❌ **Wrong under fully-async.** Between trajectory completion and the post-hoc
  request, async training may have **updated the weights**. The replayed routing
  would then reflect the *new* policy, not the policy that actually generated the
  rollout — defeating the entire point of routing replay (which is to pin the
  *rollout-time* routing). Acceptable only for fully synchronous runs.

### Approach A — capture at generation time (chosen)

Capture routing on **every `/generate` call**, exactly like `rollout_log_probs` is
captured today, and thread it through the same turn→chunk reconstruction.

- ✅ Correct under fully-async: routing is recorded against the exact weights used.
- ✅ Consistent with the existing per-turn logprob handling.
- ❌ More code; the chunk-boundary alignment is the error-prone part (covered by a
  unit test).

**Our target run is fully-async, so Approach A is the only correct choice.**

---

## 6. Token-alignment derivation (Approach A, trajectory_retokenized)

Per turn `i`, SGLang returns `R_i` of shape `[P_i + O_i - 1, L, topk]` where
`P_i = len(prompt_ids_i)`, `O_i = len(output_ids_i)`. Index `R_i[j]` is the routing
used to **predict token j+1** of that call's full sequence
`seq_i = prompt_ids_i + output_ids_i`.

We rebuild a chunk routing array `RE` aligned to `chunk_tokens = prompt_ids +
response_ids` (length `T`), targeting final `sample.rollout_routed_experts =
RE[:T-1]` so that `shape[0] == len(tokens) - 1`.

Sources for each rebuilt position:

- **First turn of a chunk:** `chunk_tokens` starts as `prompt_ids_0 = seq_0[:P_0]`,
  then `output_ids_0`. For positions `0 .. P_0 + O_0 - 2`, copy `R_0` directly
  (same sequence). Position `P_0 + O_0 - 1` (the last output token's prediction of
  the *next* token) is unknown from `R_0`; fill from the next turn (below) or, if
  this is the chunk's last token, it falls into the dropped `[:-1]`.
- **`strict_prefix` continuation** (`next.prompt = prompt + output + context`):
  the `context_ids` and the next `output_ids` all live in `seq_{i+1}`. The next
  turn's `R_{i+1}` covers `seq_{i+1}` fully, and because the prefix is strict,
  `R_{i+1}`'s leading rows **also re-cover** the prior turn's last position — use
  `R_{i+1}` to fill the one previously-missing boundary row, then continue with
  `R_{i+1}` for `context_ids` + `output_ids_{i+1}`.
- **`normalized_prefix` / `retokenized_prefix`:** edge tokens were dropped, so the
  next turn's sequence is a *near*-prefix. Map by the recorded drop counts
  (`prompt_suffix_dropped_tokens`, `response_prefix/suffix_dropped_tokens`,
  `retokenized_prompt_history_tokens`) to offset into `R_{i+1}`. Any position that
  still cannot be mapped (rare drift) gets a **placeholder routing row** (e.g. the
  previous row, or expert 0..topk-1) and is **counted in sample metadata**
  (`harbor_routing_replay_placeholder_rows`). These rows sit on `loss_mask 0`
  context positions or on dropped edges, so they do not steer the policy gradient;
  routing replay still benefits from being correct on the ≥99% real positions.
- **`split`:** a new chunk begins; its first turn is handled as "first turn of a
  chunk" above. Routing never crosses a split.

Final per chunk: assert `RE.shape == (T, 40, 8)`, set
`sample.rollout_routed_experts = RE[:T-1].astype(np.int32)`, assert
`shape[0] == len(sample.tokens) - 1`.

`turn` / `last_turn` modes are trivial: a single turn's `seq = prompt + output`,
`sample.tokens == seq`, so `sample.rollout_routed_experts = R[:len(tokens)-1]`
directly.

For strict `trajectory` mode (which DOES use `merge_turns`), the same logic is
folded into `merge_turns` so the rebuilt routing tracks the token-prefix stitching
there.

---

## 7. Implementation plan

Five files + one CPU unit test + shell wiring + an 8-GPU smoke.

| # | File | Change |
|---|------|--------|
| 1 | `slime_plugins/harbor/llm.py` | When routing-replay is on: add `return_routed_experts: true` to the payload; read `meta_info["routed_experts"]`, validate it decodes to `[P+O-1, num_layers, topk]` int32; put the base64 string (or decoded array) on `LLMResponse.extra["routed_experts"]`. New ctor kwargs: `use_rollout_routing_replay`, `num_layers`, `moe_router_topk`. |
| 2 | `slime_plugins/harbor/rollout.py` | In `_build_trial_config`, `setdefault` `args.use_rollout_routing_replay`, `args.num_layers`, `args.moe_router_topk` into `llm_kwargs`. |
| 3 | `slime_plugins/harbor/agent.py` | `_init_llm` already does `dict(llm_kwargs)`; ensure the new keys pass through to `SlimeSGLangLLM` (near-zero change). |
| 4 | `slime/agent/trajectory.py` | Add optional `routed_experts` to `TurnRecord`; rebuild it in `merge_turns` parallel to `rollout_log_probs` (for strict `trajectory` mode). |
| 5 | `slime_plugins/harbor/atif_adapter.py` | Core. Decode per-turn routing in `_turn_records_from_detail`; rebuild the full-sequence routing in the chunk loop per §6; set `sample.rollout_routed_experts`, assert `shape[0] == len(tokens)-1`; count placeholder rows into metadata. Guard unsupported modes (grouped-step) with a clear error when the flag is on. |
| 6 | `tests/test_harbor_routing_replay.py` | Synthetic 3-turn strict-prefix detail with known `R_i`; assert exact reconstruction, correct shape, and that every loss-masked output position equals its source routing row. Add a normalized-drift case asserting placeholder count and shape. |
| 7 | shell wrapper | Add `--use-rollout-routing-replay` to the 35B-A3B wrapper (via `EXTRA_MISC_ARGS` or a dedicated `USE_ROLLOUT_ROUTING_REPLAY` env toggle in the base script). |
| 8 | 8-GPU smoke | `--debug-rollout-only`, one MoE engine; verify SGLang returns `routed_experts`, that `num_layers` seen == 40 and the off-by-one (`len(tokens)-1`) holds end-to-end before scaling to 96 GPU. |

---

## 8. Risks & open questions

1. **Chunk-boundary alignment** is the single most error-prone area. Mitigation:
   the unit test (#6) plus the `harbor_routing_replay_placeholder_rows` metric
   surfaced to W&B so we can see how often we fall back to placeholders.
2. **Layer count seen by SGLang vs Megatron.** SGLang may or may not count an MTP
   layer; the run config has `mtp_num_hidden_layers: 1`. Must confirm in smoke that
   the decoded second-axis is exactly `args.num_layers` (40) and not 41. If SGLang
   includes MTP, slice to the 40 transformer layers before storing.
3. **Memory.** `[~64k, 40, 8] int32 ≈ 80 MB/sample` before `[:-1]`; with 256
   in-flight trajectories this is non-trivial host RAM during conversion. Decode
   lazily per chunk and store only `[:T-1]` per emitted sample.
4. **Placeholder rows on dropped edges** are acceptable only because they land on
   `loss_mask 0` or dropped positions. If a future change moves them onto trained
   tokens, the assumption breaks — keep the metric and alert if placeholders ever
   coincide with `loss_mask 1`.

---

## 9. Validation plan (before 96-GPU launch)

1. `python -m pytest tests/test_harbor_routing_replay.py -v` (CPU).
2. 8-GPU `--debug-rollout-only` smoke: one Qwen3.5-35B-A3B MoE SGLang engine,
   a handful of trials, assert:
   - `meta_info["routed_experts"]` present and decodes to int32;
   - second axis == 40, third axis == 8;
   - for every emitted sample, `rollout_routed_experts.shape[0] == len(tokens)-1`;
   - placeholder-row fraction is ≈0 on strict_prefix trajectories.
3. Single end-to-end train step with the flag on (tiny `num_rollout`), confirm
   `fill_routing_replay` does NOT raise and the step completes.
4. Only then enable on the full 96-GPU run.

---

## 10. Appendix — key source references

| Concern | Location |
|---------|----------|
| Flag definition | `slime/utils/arguments.py:1028-1033` (`--use-rollout-routing-replay`) |
| Training consumer / hard asserts | `slime/backends/megatron_utils/actor.py:283-368` |
| Replay mechanism | `slime/utils/routing_replay.py` |
| SGLang server flag | `srt/server_args.py:663` (`enable_return_routed_experts`) |
| slime sets server flag | `slime/backends/sglang_utils/sglang_engine.py:605-606` |
| Native request/response | `slime/rollout/sglang_rollout.py:179-232` |
| Sample → train_data | `slime/ray/rollout.py:1012-1013` |
| Sample field (exists) | `slime/utils/types.py:35` (`rollout_routed_experts`) |
| Harbor LLM backend | `slime_plugins/harbor/llm.py:58-155` |
| Harbor args→llm_kwargs | `slime_plugins/harbor/rollout.py:872-928` (`_build_trial_config`) |
| Harbor agent init | `slime_plugins/harbor/agent.py:11-53` |
| Per-turn extra read | `slime_plugins/harbor/atif_adapter.py:991-996` (`_detail_extra_list`) |
| Turn records | `slime_plugins/harbor/atif_adapter.py:258-281` |
| Chunk reconstruction | `slime_plugins/harbor/atif_adapter.py:640-826` |
| Transition alignment | `slime_plugins/harbor/atif_adapter.py:860-988` |
| Trajectory merge | `slime/agent/trajectory.py:85-151` (`merge_turns`) |
