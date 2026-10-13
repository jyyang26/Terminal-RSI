# Normalized Trajectory Stitching Audit

- Generated: 2026-06-05T08:57:19
- Raw export dir: `/path/to/workspace/code/slime/examples/harbor_terminal_rl/trajectory_turn_io_raw`
- Records: 5
- Turns / transitions: 349 / 344

## Summary

| classification | count |
| --- | ---: |
| `strict_prefix` | 0 |
| `qwen_template_normalized_prefix` | 338 |
| `unresolved_normalized_mismatch` | 6 |
| `reset_or_summary_candidate` | 0 |

## Token Estimate

| metric | estimated tokens |
| --- | ---: |
| raw full-prompt context inserted on non-strict turns | 9130670 |
| normalized tail/reset context | 228626 |
| estimated context tokens saved | 8902044 |
| normalized chunk count | 11 |

Interpretation:

- `qwen_template_normalized_prefix` means the next prompt is continuous after stripping only assistant generation prefill and leading think-boundary artifacts.
- `unresolved_normalized_mismatch` means safe Qwen/template normalization is not enough, but overlap is still high. These are likely parser/template formatting differences and need manual inspection before automatic training conversion.
- `reset_or_summary_candidate` means overlap is low after safe normalization. Treat the next prompt as a new context base rather than continuing one causal sequence.
- The token estimate is approximate because this audit uses exported decoded text plus per-turn prompt token counts, not a live tokenizer.

## Per Record

| file | turns | strict | qwen-normalized | unresolved | reset-candidate | est raw ctx | est normalized ctx | est saved | chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dump01_build_pmars_success_g4_s1.json | 71 | 0 | 69 | 1 | 0 | 1555665 | 35273 | 1520392 | 2 |
| dump01_circuit_fibsqrt_fail_longest.json | 100 | 0 | 99 | 0 | 0 | 2745409 | 17942 | 2727467 | 1 |
| dump20_scheme_fail_g3_s1.json | 100 | 0 | 95 | 4 | 0 | 3123515 | 119082 | 3004433 | 5 |
| dump20_sqlite_success_g6_s7.json | 18 | 0 | 17 | 0 | 0 | 165470 | 11587 | 153883 | 1 |
| dump20_video_fail_g11_s10.json | 60 | 0 | 58 | 1 | 0 | 1540611 | 44742 | 1495869 | 2 |

## Unresolved Examples

### dump01_build_pmars_success_g4_s1.json

- turn 13->14: `unresolved_normalized_mismatch`, prefix_ratio=0.9797, note=high-overlap mismatch remains after safe normalization; inspect parser/template differences

Expected normalized projection around mismatch:

```text
he B-operand (NEW_MODES).
653:#define ADDRB_AVALUE AB_Value
1014:   if (AB_Value == 1)
1055:   if ((AB_Valu ... lue' undeclared suggests the code structure. Need to build with only the essential flags without graphics s
```

Actual next prompt around mismatch:

```text
he B-operand (NEW_MODES).
653:#define ADDRB_AVALUE AB_Value
1014:   if (AB_Value == 1)
1055:   if ((AB_Valu ... e' undeclared suggests the code structure. Need to build with only the essential flags without graphics sup
```

### dump20_scheme_fail_g3_s1.json

- turn 6->7: `unresolved_normalized_mismatch`, prefix_ratio=0.9413, note=high-overlap mismatch remains after safe normalization; inspect parser/template differences

Expected normalized projection around mismatch:

```text
 (cdr line))))))
   ((= (car line) '*)
    (display (* (car (cdr line)) (car (cdr (cdr line))))))))
(newlin ...  It demonstrates the Scheme-like language syntax: (print display), (newline), (let), (read), (cond), (car),
```

Actual next prompt around mismatch:

```text
 (cdr line))))))
   ((= (car line) '*)
    (display (* (car (cdr line)) (car (cdr (cdr line))))))))
(newlin ... nterpreter interp.py can execute Scheme files with primitives: car, cdr, cons, read, display, newline, let,
```

- turn 14->15: `unresolved_normalized_mismatch`, prefix_ratio=0.9637, note=high-overlap mismatch remains after safe normalization; inspect parser/template differences

Expected normalized projection around mismatch:

```text
ter JSON object

New Terminal Output:

Apptainer> cat > eval.scm << 'EOF'
> (define (eval expr)
>   (displa ... .scm with the evaluator to read file path of STDIN:
./interp.py scheme
```
The exec code with interp.py
ech
```

Actual next prompt around mismatch:

```text
ter JSON object

New Terminal Output:

Apptainer> cat > eval.scm << 'EOF'
> (define (eval expr)
>   (displa ... e. Let us see the  and interpret the file to execute the straightforward eval.scm. The interp.py is a  sche
```

- turn 66->67: `unresolved_normalized_mismatch`, prefix_ratio=0.9850, note=high-overlap mismatch remains after safe normalization; inspect parser/template differences

Expected normalized projection around mismatch:

```text
expr))
>   ((if* expr) (car expr))
>   (else #T)))
> EOF
> echo '(eval 1 2)' | python3 interp.py eval.scm
> ... and the code runs without error. The evaluator is working.

The interpreter:

```
I have a file `eval.scm`

```

Actual next prompt around mismatch:

```text
expr))
>   ((if* expr) (car expr))
>   (else #T)))
> EOF
> echo '(eval 1 2)' | python3 interp.py eval.scm
> ...  eval.scm, that is an interpreter written in Scheme that interprets Scheme programs. A metacircular evaluat
```

### dump20_video_fail_g11_s10.json

- turn 38->39: `unresolved_normalized_mismatch`, prefix_ratio=0.9999, note=high-overlap mismatch remains after safe normalization; inspect parser/template differences

Expected normalized projection around mismatch:

```text
  },
    {
      "keystrokes": "python3 /app/jump_analyzer.py\n",
      "duration": 2.0
    },
    {
      "keystrokes": "cat /app/output.toml\n",
      "duration": 0.5
    }
  ]
}
<|im_end|>
```

Actual next prompt around mismatch:

```text
  },
    {
      "keystrokes": "python3 /app/jump_analyzer.py\n",
      "duration": 2.0
    },
    {
       ... Invalid JSON: Expecting ',' delimiter: line 7 column 7 (char 1568) | Content preview: '{\n  "analysis": "Th
```

## Training Implication

A normalized trajectory stitcher can recover most apparent raw-token prefix mismatches in these exports, especially the Qwen think-template boundary cases. It should still be treated as an approximate fast path unless the training pipeline can either:

1. train on the same normalized assistant content that the next prompt renderer replays, or
2. reset/split the causal sequence whenever generated raw tokens are removed from later rendered history.

For summary or low-overlap resets, the correct behavior is to start a new segment from the summary-rendered prompt. Continuing the old raw trajectory would put tokens in attention that the rollout policy did not condition on.

## 2026-06-06 Rollout 1 Follow-up

Target dump:

```text
/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_actor1_rollout3_batch3_n32_fully_async_trajectory_normalized_timeout3600_20260606_123111/rollout_dumps/1.pt
```

The dump contains `644` training chunks grouped into `96` trajectory group ids. The extra chunks are exactly the `548` transitions where the current `trajectory_normalized` token-prefix alignment returns `strategy == "split"`:

| metric | count |
| --- | ---: |
| dump samples/chunks | 644 |
| trajectory group ids | 96 |
| selected raw turns | 6676 |
| selected raw transitions | 6580 |
| current `normalized_prefix` transitions | 6032 |
| current `split` transitions | 548 |

The raw `result.json` details for this run contain only `prompt_token_ids`, `completion_token_ids`, `logprobs`, and `extra.finish_reason`. They do not contain `prompt_text`, `completion_text`, `history_content`, or message-history fields, so the follow-up audit decoded token ids with the Qwen3.5-9B tokenizer on the Taiji job node.

### Root Cause

The main failure mode is not terminal summary or context reset. It is token-boundary drift after Terminus2 re-renders the same assistant message into the next prompt. At the decoded-text level:

| text-level classification | count |
| --- | ---: |
| next prompt starts with the exact decoded completion text | 546 |
| low-overlap / real history rewrite | 2 |

For the 546 text-exact cases, `completion_token_ids` and the next prompt's assistant-history token ids decode to the same text, but they are not the same token sequence. Examples:

| decoded text | raw completion tokenization | next prompt tokenization |
| --- | --- | --- |
| `Adjustment` | `Adjust` + `ment` | `Adjustment` |
| `__.VALUE` | `__` + `.VALUE` | `__.` + `VALUE` |
| `XSS` | `X` + `SS` | `XSS` |

This happens because the model output is generated token by token, while the next prompt is built from message text and then tokenized again. BPE token boundaries are not stable across those two paths even when the decoded text is identical.

The shared prompt suffix for all 96 selected trajectories was:

```text
<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The existing `_normalized_transition_alignment` handles this Qwen assistant prefill by dropping prompt suffix tokens, but it still requires the raw generated completion token ids to be a prefix of the next prompt's re-tokenized assistant history. That strict token-prefix check is what turns 96 trajectories into 644 chunks.

### Real Rewrite Cases

Only 2 of the 548 split transitions were content-level rewrites:

- `slime-r1-g2-s15-i79`, turn 85: raw completion starts with a long CSS `javascript:` analysis; next prompt history rewrites it to a shorter “CSS approach doesn't work” analysis.
- `slime-r1-g2-s23-i87`, turn 21: raw completion starts with natural-language reasoning; next prompt history starts from a JSON `analysis` object.

Those should remain splits unless the rollout detail carries an explicit rendered-history projection that can be safely trained against.

### Mask and Logprob Check

The existing dump does not show a mask/logprob length bug:

| check | result |
| --- | ---: |
| samples with bad `loss_mask` length | 0 |
| samples with missing `rollout_log_probs` | 0 |
| samples with bad `rollout_log_probs` length | 0 |
| raw turns with missing logprobs | 0 |
| raw turns with logprob/completion length mismatch | 0 |

The converter now treats a provided per-turn logprob list with the wrong length as a hard `RolloutConversionError`; missing logprobs are still zero-filled, but malformed logprobs are not silently dropped.

Totals in the dump:

| token category | tokens |
| --- | ---: |
| response tokens | 5,349,260 |
| loss-mask tokens | 2,990,103 |
| zero-mask bridge/context tokens | 2,359,157 |

### Implementation Direction

The fix should not make `trajectory_normalized` silently accept arbitrary token drift. The safer path is a separate mode that has access to message-level evidence:

1. During rollout, record per-call `completion_text`.
2. During the next call, record the previous assistant message content and its token span inside the next prompt.
3. Merge a split transition only when the raw completion text exactly matches that next-prompt assistant history content.
4. Use the next-prompt assistant-history span to skip the re-tokenized history tokens and append only the later context/terminal-observation tokens with `loss_mask=0`.
5. Keep content-level rewrites as real splits.

This is implemented as the new `trajectory_retokenized` mode rather than changing `trajectory_normalized`.

### Retokenized Mode Dry-run on the Same Dump

The original rollout did not record text/span metadata, so the dry-run synthesized equivalent metadata in memory by decoding the old token ids and locating exact text-prefix assistant history spans in the next prompt. It then converted the same 96 selected trials through `trajectory_retokenized`.

| metric | value |
| --- | ---: |
| selected trials | 96 |
| samples after `trajectory_retokenized` | 98 |
| trajectory group ids | 96 |
| chunk histogram: 1 chunk | 94 trajectories |
| chunk histogram: 2 chunks | 2 trajectories |
| retokenized transitions merged | 546 |
| real rewrites/unmatched transitions left split | 2 |
| conversion errors | 0 |

Length distribution from the dry-run:

| length field | min | p50 | p90 | max | >64k | >128k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| total tokens | 4,088 | 58,428 | 99,469 | 127,263 | 38 | 0 |
| response tokens | 3,261 | 57,601 | 98,198 | 125,980 | 37 | 0 |

This confirms the intended behavior: the pathological `96 -> 644` split inflation collapses to `96 -> 98`, while the two content-level rewrites remain separate chunks. It also avoids the old `trajectory` mode failure where samples reached millions of tokens; the dry-run max stays below 128k.

## 2026-06-07 Live `trajectory_retokenized` Follow-up

Live run:

```text
/path/to/workspace/slime_harbor_qwen35_9b_tb2_full89_32gpu_actor1_rollout3_batch3_n32_fully_async_trajectory_retokenized_timeout3600_20260607_022356
Ray job: raysubmit_RgDyzCFP7KRXjWcu
```

The run did enter `trajectory_retokenized`; the issue was not a config/mode mismatch. Dump metadata showed:

| dump | samples | trajectories | retokenized prefix turns | normalized prefix turns | split-after turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1.pt` | 608 | 96 | 0 | 5905 | 512 |
| `2.pt` | 253 | 96 | 0 | 5624 | 157 |

So the retokenized fallback path was never used. It silently degraded to the older `trajectory_normalized` behavior and still split many trajectories.

### Live Root Cause

The live `result.json` files did record the text-level evidence:

- `completion_text`: present for every turn.
- `last_assistant_message_content`: present for the next prompt.
- For a 100-turn case, `completion_text[i] == last_assistant_message_content[i + 1]` for `99/99` transitions.
- The next prompt decoded text contained the previous completion text.

However, `last_assistant_message_span` was `None` for all turns. `_retokenized_transition_context()` requires that span, so it returned `None` even when the message text matched exactly.

The span recorder in `SlimeSGLangLLM._message_token_span()` used this assumption:

1. render `messages[:assistant_index]`;
2. render `messages[:assistant_index + 1]`;
3. require both token sequences to be prefixes of the actual full prompt.

That assumption fails for Qwen chat templates. When an assistant message is rendered as the final message, Qwen inserts the thinking prefill:

```text
<think>

</think>

```

The same assistant message does not include that prefill when it is in the middle of the full history and followed by a user message. Therefore the prefix check fails and the span is recorded as `None`.

### Fix

`slime_plugins/harbor/llm.py` now computes the previous assistant span from the actual prompt text first:

1. decode the exact `prompt_token_ids` used for `/generate`;
2. locate the previous assistant message content in that decoded prompt text;
3. tokenize the decoded prompt text with `return_offsets_mapping=True`;
4. require the re-tokenized ids to exactly match the original `prompt_token_ids`;
5. map the character span to a token span and verify that decoding the token span equals the assistant content;
6. fall back to the old chat-template-prefix method only if this text/offset path is unavailable.

This avoids the Qwen final-assistant thinking-prefill artifact while still requiring exact text and token-span evidence.

### Sanity Check

For the live trajectory `slime-r1-g2-s27-i91`, the broken dump produced `18` chunks and `0` retokenized transitions. Recomputing spans with the fixed helper and rerunning conversion gives:

| metric | value |
| --- | ---: |
| recorded spans | 99 / 100 |
| samples after conversion | 1 |
| retokenized transitions | 17 |
| remaining splits | 0 |
| max sample length | 50,309 |

The top 10 most-split trajectories from `1.pt` all collapsed to one sample after span recomputation:

| trial | before | after | max tokens after |
| --- | ---: | ---: | ---: |
| `slime-r1-g2-s27-i91` | 18 | 1 | 50,309 |
| `slime-r1-g0-s21-i21` | 15 | 1 | 117,931 |
| `slime-r1-g0-s3-i3` | 13 | 1 | 90,949 |
| `slime-r1-g0-s23-i23` | 13 | 1 | 122,080 |
| `slime-r1-g1-s31-i63` | 12 | 1 | 118,028 |
| `slime-r1-g1-s20-i52` | 12 | 1 | 115,110 |
| `slime-r1-g1-s26-i58` | 12 | 1 | 109,262 |
| `slime-r1-g0-s0-i0` | 12 | 1 | 76,513 |
| `slime-r1-g0-s25-i25` | 12 | 1 | 97,178 |
| `slime-r1-g2-s2-i66` | 11 | 1 | 31,975 |

The fix addresses the live `retokenized_prefix_turns=0` bug. It does not by itself solve long-sequence actor memory pressure: after successful stitching, many trajectories are again above 64k tokens. Those runs need context parallelism and/or a training cap that preserves valid masks and logprob lengths.
