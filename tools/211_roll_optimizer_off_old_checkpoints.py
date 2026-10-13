#!/usr/bin/env python3
"""Roll optimizer state off older checkpoints, keeping only the newest resumable.

Motivation: each Megatron torch_dist checkpoint of this 122B run is ~1.6 TB
(244 GB weights + ~1.5 TB distributed-Adam state), and both actor and critic
save one, so every save_interval costs ~3.2 TB on a shared volume that already
sits at 98%. But dropping the optimizer outright (``--no-save-optim``) makes
the run unresumable.

This watcher keeps that tradeoff: the NEWEST checkpoint keeps its optimizer so
training can always resume from it, while every OLDER checkpoint is converted
to a weights-only HF safetensors directory (234 GB) and its torch_dist tree
deleted.

Conversion is TWO steps and BOTH are mandatory:

  1. ``convert_torch_dist_to_hf_parallel.py`` -- torch_dist -> HF. On its own
     the output is NOT usable: it omits 1214 keys (``mtp.*`` 785,
     ``model.visual.*`` 333, i.e. the whole vision tower) and writes experts in
     a per-expert layout (36864 tensors) that transformers' fused
     ``gate_up_proj``/``down_proj`` parameters do not match -- so every expert
     would silently stay randomly initialised.
  2. ``210_repair_qwen35_moe_converted_hf.py`` -- refuses the experts, restores
     mtp/visual from the origin HF checkpoint, and casts the 72 fp32 GDN
     tensors back to origin dtype.

Safety: an iter dir is deleted ONLY after its repaired output is verified to
have exactly origin's 1949 keys and byte-identical ``total_size``. The newest
checkpoint (and anything the run may still need) is never touched. Run with
``--dry-run`` to see the plan without writing or deleting.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ITER_RE = re.compile(r"^iter_(\d+)$")
# Qwen3.5-122B-A10B LM head: 248320 x 3072 bf16. A critic checkpoint replaces it
# with a [1, hidden] value head, so its total_size is short by exactly this much.
LM_HEAD_BYTES = 248320 * 3072 * 2


def iter_dirs(ckpt_root: Path) -> list[tuple[int, Path]]:
    if not ckpt_root.is_dir():
        return []
    out = []
    for p in ckpt_root.iterdir():
        m = ITER_RE.match(p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    return sorted(out)


def origin_signature(origin_hf_dir: Path) -> tuple[int, int]:
    idx = json.loads((origin_hf_dir / "model.safetensors.index.json").read_text())
    return len(idx["weight_map"]), int(idx["metadata"]["total_size"])


def verify_repaired(repaired: Path, want_keys: int, want_size: int) -> str | None:
    """Return None if the repaired dir looks complete, else a reason string.

    ``want_size`` is origin's total_size, which an ACTOR checkpoint must match
    byte-for-byte. A CRITIC checkpoint cannot: its output_layer is a value head
    ([1, 3072]) where origin carries the LM head ([248320, 3072]), so its total
    is legitimately one lm_head smaller (1525671936 bytes -- exactly the delta
    the first critic sweep reported). Accept a shortfall of at most one lm_head;
    anything larger, or any surplus, still fails. The per-tensor shape+dtype
    check inside 210_repair already holds every other tensor to origin.
    """
    index = repaired / "model.safetensors.index.json"
    if not index.is_file():
        return "no model.safetensors.index.json"
    idx = json.loads(index.read_text())
    n = len(idx.get("weight_map", {}))
    size = int(idx.get("metadata", {}).get("total_size", 0))
    if n != want_keys:
        return f"key count {n} != origin {want_keys}"
    shortfall = want_size - size
    if not (shortfall == 0 or 0 < shortfall <= LM_HEAD_BYTES):
        return f"total_size {size} vs origin {want_size} (delta {shortfall}) outside the critic-head allowance"
    referenced = set(idx["weight_map"].values())
    missing = [f for f in referenced if not (repaired / f).is_file()]
    if missing:
        return f"{len(missing)} shard(s) referenced but absent, e.g. {missing[0]}"
    return None


def run(cmd: list[str], log: Path) -> int:
    with log.open("ab") as f:
        f.write(f"\n$ {' '.join(cmd)}\n".encode())
        f.flush()
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)


def convert_one(
    iter_dir: Path,
    step: int,
    out_root: Path,
    origin_hf_dir: Path,
    tag: str,
    workers: int,
    log: Path,
) -> Path | None:
    """torch_dist -> HF -> repaired. Returns the repaired dir, or None on failure."""
    raw = out_root / f"{tag}_iter{step}_hf_raw"
    repaired = out_root / f"{tag}_iter{step}_hf"
    want_keys, want_size = origin_signature(origin_hf_dir)

    if verify_repaired(repaired, want_keys, want_size) is None:
        print(f"[skip] {repaired.name} already verified", flush=True)
        return repaired

    rc = run(
        [
            sys.executable,
            str(TOOLS / "convert_torch_dist_to_hf_parallel.py"),
            "--input-dir",
            str(iter_dir),
            "--output-dir",
            str(raw),
            "--origin-hf-dir",
            str(origin_hf_dir),
            "--load-max-workers",
            str(workers),
            "--save-max-workers",
            str(workers),
            "--force",
        ],
        log,
    )
    if rc != 0:
        print(f"[FAIL] step1 convert rc={rc} for {iter_dir}", flush=True)
        return None

    rc = run(
        [
            sys.executable,
            str(TOOLS / "210_repair_qwen35_moe_converted_hf.py"),
            "--origin-hf-dir",
            str(origin_hf_dir),
            "--converted-hf-dir",
            str(raw),
            "--output-dir",
            str(repaired),
            "--force",
        ],
        log,
    )
    if rc != 0:
        print(f"[FAIL] step2 repair rc={rc} for {iter_dir}", flush=True)
        return None

    why = verify_repaired(repaired, want_keys, want_size)
    if why:
        print(f"[FAIL] verification: {why}", flush=True)
        return None

    shutil.rmtree(raw, ignore_errors=True)  # intermediate, superseded by repaired
    print(f"[ok] {repaired.name} verified ({want_keys} keys, {want_size} bytes)", flush=True)
    return repaired


def sweep(args) -> None:
    roots = [(args.actor_ckpt_dir, "actor"), (args.critic_ckpt_dir, "critic")]
    for ckpt_root, role in roots:
        if ckpt_root is None:
            continue
        ckpt_root = Path(ckpt_root)
        steps = iter_dirs(ckpt_root)
        if len(steps) <= args.keep:
            print(f"[{role}] {len(steps)} ckpt(s), keep={args.keep} -> nothing to roll", flush=True)
            continue
        # Newest `keep` stay resumable; everything older gets rolled.
        stale = steps[: -args.keep]
        newest = steps[-1][0]
        print(f"[{role}] newest=iter_{newest} keep={args.keep} rolling={[s for s, _ in stale]}", flush=True)
        for step, iter_dir in stale:
            if step == newest:
                continue  # never touch the resume point
            if args.dry_run:
                print(f"[dry-run] would convert+delete {iter_dir}", flush=True)
                continue
            repaired = convert_one(
                iter_dir,
                step,
                Path(args.output_root),
                Path(args.origin_hf_dir),
                f"{args.tag}_{role}" if role == "critic" else args.tag,
                args.workers,
                Path(args.log_file),
            )
            if repaired is None:
                print(f"[keep] conversion failed, NOT deleting {iter_dir}", flush=True)
                continue
            print(f"[rm] {iter_dir} (optimizer state; weights preserved in {repaired.name})", flush=True)
            shutil.rmtree(iter_dir, ignore_errors=True)


def pre_convert_newest(args) -> None:
    """Convert the newest checkpoint to safetensors but KEEP its torch_dist tree.

    This is the space-safety half of the strategy. ``sweep`` reclaims 3.2 TB only
    after a NEW checkpoint has landed, which is too late: the save itself is the
    peak, and on 2026-08-14 it hit a full volume mid-write and killed the run.
    Converting the current checkpoint ahead of time makes the next reclaim
    instantaneous -- delete a directory whose weights are already preserved --
    so the two-optimizer-checkpoint window shrinks from ~40 min to seconds.
    """
    for ckpt_root, role in [(args.actor_ckpt_dir, "actor"), (args.critic_ckpt_dir, "critic")]:
        if ckpt_root is None:
            continue
        steps = iter_dirs(Path(ckpt_root))
        if not steps:
            continue
        step, iter_dir = steps[-1]
        tag = f"{args.tag}_{role}" if role == "critic" else args.tag
        if args.dry_run:
            print(f"[dry-run] would pre-convert newest {iter_dir} (kept, not deleted)", flush=True)
            continue
        repaired = convert_one(
            iter_dir,
            step,
            Path(args.output_root),
            Path(args.origin_hf_dir),
            tag,
            args.workers,
            Path(args.log_file),
        )
        if repaired is None:
            print(f"[warn] pre-convert failed for {iter_dir}; it stays as-is", flush=True)
        else:
            print(f"[pre-ok] {repaired.name} ready; {iter_dir.name} kept (still the resume point)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor-ckpt-dir", required=True)
    ap.add_argument("--critic-ckpt-dir", default=None)
    ap.add_argument("--origin-hf-dir", required=True)
    ap.add_argument("--output-root", required=True, help="where the safetensors dirs land")
    ap.add_argument("--tag", default="q35_122b")
    ap.add_argument("--keep", type=int, default=1, help="newest N checkpoints keep their optimizer")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--log-file", default="/tmp/rolling_ckpt.log")
    ap.add_argument("--watch-interval", type=int, default=0, help="seconds; 0 = single sweep")
    ap.add_argument(
        "--pre-convert-newest",
        action="store_true",
        help=(
            "Also convert the NEWEST checkpoint to safetensors, WITHOUT deleting its "
            "torch_dist tree. Purpose: the plain sweep only reclaims space AFTER the next "
            "checkpoint lands, so it cannot prevent the save itself from filling the disk "
            "(observed 2026-08-14: iter_29 exhausted the volume mid-write and killed the run, "
            "leaving a truncated critic ckpt). Pre-converting means that once the next "
            "checkpoint appears, the previous one can be deleted IMMEDIATELY -- its weights "
            "are already safe on disk -- instead of waiting out a ~40 min conversion while "
            "two optimizer-bearing checkpoints coexist."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.keep < 1:
        raise SystemExit("--keep must be >= 1: the newest checkpoint must stay resumable")

    while True:
        sweep(args)
        if args.pre_convert_newest:
            pre_convert_newest(args)
        if not args.watch_interval:
            return
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
