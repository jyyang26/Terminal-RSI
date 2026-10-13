#!/usr/bin/env python3
"""Concurrent Daytona image-build probe — the gate before launching a Daytona run.

WHY THIS EXISTS
---------------
The tmax-15k PPO run was stopped TWICE (2026-07-01 02:24 and 06:27) by the same
Daytona `prod-daytona-build-context` (AWS us-east-1) S3 outage: every trial died
with `Failed to create sandbox: Generic S3 error: ... HEAD .../context.tar after
10 retries`. A *single* image-build probe passed in ~2s both times — but the
training fires ~280 concurrent `Image.from_dockerfile` uploads at rollout start,
and S3's HEAD-on-context.tar buckles under that concurrency. LESSON: a single
create is NOT evidence S3 recovered. You must probe UNDER CONCURRENCY.

This script mirrors Harbor's exact create path (`AsyncDaytona` +
`Image.from_dockerfile` + `CreateSandboxFromImageParams`, see
harbor/src/harbor/environments/daytona.py `_DaytonaDirect.start`) and fires N of
them concurrently against real training Dockerfiles, then reports success rate and
failure breakdown. Exit code 0 iff success-rate >= --min-success-rate, so it can
gate a launch:  `probe && launch`.

USAGE
-----
  # default: 64 concurrent builds from the tmax-15k pool, need >=90% success
  harbor/.venv/bin/python daytona_concurrent_build_probe.py

  # mimic the real training fan-out (280 concurrent), require 85%
  ... --concurrency 280 --min-success-rate 0.85

  # use a specific dataset's Dockerfiles
  ... --tasks-dir /tmp/tmax_harbor_15k_full --concurrency 100

Creds come from the same env file the training uses (KEY6 + star-proxy), or from
the ambient DAYTONA_API_KEY / proxy env vars.
"""
import argparse
import asyncio
import os
import random
import sys
import time
from pathlib import Path


def _load_env_file(path: str) -> None:
    """Source a shell-style env file (KEY=VALUE lines) into os.environ. Quiet on miss."""
    if not path or not os.path.isfile(path):
        return
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("export ") and "=" not in line:
            continue
        line = line[len("export "):] if line.startswith("export ") else line
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        # don't clobber anything already set in the ambient env
        os.environ.setdefault(k, v)


def _set_proxy(proxy: str) -> None:
    if not proxy:
        return
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.setdefault(v, proxy)
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,.woa.com,.oa.com,mirrors.tencent.com")
    os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])


def _classify(exc: Exception) -> str:
    s = f"{type(exc).__name__}: {exc}"
    low = s.lower()
    if "generic s3 error" in low or "context.tar" in low or "prod-daytona-build-context" in low:
        return "S3_BUILD_CONTEXT"
    if "timeout" in low or "timed out" in low:
        return "TIMEOUT"
    if "rate" in low and "limit" in low:
        return "RATE_LIMIT"
    if "quota" in low or "insufficient" in low or "capacity" in low:
        return "QUOTA_CAPACITY"
    return "OTHER"


def find_dockerfiles(tasks_dir: str, n: int, seed: int) -> list[str]:
    root = Path(tasks_dir)
    if not root.is_dir():
        raise SystemExit(f"tasks-dir not found: {tasks_dir}")
    dfs = [str(p) for p in root.glob("*/environment/Dockerfile")]
    if not dfs:
        # fall back to a deeper search (some layouts nest differently)
        dfs = [str(p) for p in root.rglob("Dockerfile")]
    if not dfs:
        raise SystemExit(f"no Dockerfiles under {tasks_dir}")
    rng = random.Random(seed)
    # sample WITH replacement if the pool is smaller than n (still exercises concurrency)
    if len(dfs) >= n:
        return rng.sample(dfs, n)
    return [rng.choice(dfs) for _ in range(n)]


class _TokenBucket:
    """Async token bucket — mirrors slime_plugins.harbor.rollout._AsyncTokenBucket so
    the probe validates the SAME throttle the training uses. rate<=0 disables it."""

    def __init__(self, rate: float, capacity: float):
        self.rate = float(rate)
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self.rate <= 0:
            return
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self._tokens) / self.rate)


async def _one_build(idx, dockerfile, sem, results, args, limiter):
    # Imports are deferred so --help works without the SDK installed.
    from daytona import AsyncDaytona, CreateSandboxFromImageParams, Resources, Image

    async with sem:
        if limiter is not None:
            await limiter.acquire()  # pace create rate, exactly like training
        t0 = time.time()
        rec = {"idx": idx, "dockerfile": dockerfile, "ok": False,
               "create_s": None, "kind": None, "err": None, "attempts": 0}
        image = Image.from_dockerfile(dockerfile)  # uploads build-context to S3
        # Exponential backoff over the create call: base * 2^(n-1) + jitter, capped.
        for attempt in range(1, args.max_retries + 2):  # 1 initial + max_retries
            rec["attempts"] = attempt
            client = AsyncDaytona()
            sbx = None
            try:
                params = CreateSandboxFromImageParams(
                    image=image,
                    labels={"slime_probe": "concurrent_build"},
                    resources=Resources(cpu=1, memory=2, disk=10),
                    auto_stop_interval=args.auto_stop,
                    auto_delete_interval=args.auto_delete,
                )
                sbx = await client.create(params=params, timeout=args.build_timeout)
                rec["create_s"] = round(time.time() - t0, 1)
                rec["ok"] = True
                rec["kind"] = None
                rec["err"] = None
                break
            except Exception as e:  # noqa: BLE001 - we want to bucket every failure
                rec["create_s"] = round(time.time() - t0, 1)
                rec["kind"] = _classify(e)
                rec["err"] = f"{type(e).__name__}: {str(e)[:160]}"
            finally:
                if sbx is not None:
                    try:
                        await client.delete(sbx)
                    except Exception:
                        pass  # auto_delete_interval is the backstop
                try:
                    await client.close()
                except Exception:
                    pass
            if rec["ok"] or attempt > args.max_retries:
                break
            # Backoff before the next create attempt. Deterministic jitter from idx
            # (Math.random unavailable / not needed) to avoid a retry thundering herd.
            delay = min(args.backoff_cap, args.backoff_base * (2 ** (attempt - 1)))
            delay += (idx % 7) * 0.13  # spread retries ~0-0.8s
            await asyncio.sleep(delay)
            if limiter is not None:
                await limiter.acquire()  # a retry is another .create() — pace it too

        results.append(rec)
        tag = "OK " if rec["ok"] else f"FAIL[{rec['kind']}]"
        print(f"  [{idx:>3}] {tag} {rec['create_s']}s a{rec['attempts']} "
              f"{Path(dockerfile).parent.parent.name}"
              + ("" if rec["ok"] else f"  {rec['err']}"), flush=True)


async def run_probe(args) -> int:
    dfs = find_dockerfiles(args.tasks_dir, args.concurrency, args.seed)
    if args.patch_proxy:
        # Apply the same S3-proxy monkey-patch the training uses, so obstore's
        # build-context uploads go through the egress proxy. Without this every
        # signed HEAD/PUT to prod-daytona-build-context times out (~58s).
        try:
            sys.path.insert(0, "/path/to/workspace/code/slime-v0.3-harbor")
            from slime_plugins.harbor.import_utils import ensure_daytona_s3_proxy
            ensure_daytona_s3_proxy()
            print("[probe] applied ensure_daytona_s3_proxy() (obstore proxy patch)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] WARNING: could not apply proxy patch: {exc!r}", flush=True)
    key = os.environ.get("DAYTONA_API_KEY", "")
    print(f"[probe] key ...{key[-8:] if key else '<none>'}  "
          f"concurrency={args.concurrency} timeout={args.build_timeout}s "
          f"min_success={args.min_success_rate:.0%}", flush=True)
    print(f"[probe] tasks-dir={args.tasks_dir} ({len(set(dfs))} unique Dockerfiles)", flush=True)
    if args.rate_per_sec > 0:
        print(f"[probe] rate limiter ON: {args.rate_per_sec:.1f} creates/sec "
              f"({args.rate_per_sec*60:.0f}/min), burst={args.rate_burst or args.rate_per_sec:.0f}",
              flush=True)
    if not key and not (os.environ.get("DAYTONA_JWT_TOKEN") and os.environ.get("DAYTONA_ORGANIZATION_ID")):
        print("[probe] ERROR: no DAYTONA_API_KEY (or JWT+ORG) in env. "
              "Pass --env-file or export creds.", file=sys.stderr, flush=True)
        return 2

    limiter = None
    if args.rate_per_sec > 0:
        cap = args.rate_burst if args.rate_burst > 0 else max(1.0, args.rate_per_sec)
        limiter = _TokenBucket(rate=args.rate_per_sec, capacity=cap)

    sem = asyncio.Semaphore(args.concurrency)  # all fire at once by default
    results: list[dict] = []
    t0 = time.time()
    await asyncio.gather(*(
        _one_build(i, df, sem, results, args, limiter) for i, df in enumerate(dfs)
    ))
    wall = time.time() - t0

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    rate = len(ok) / len(results) if results else 0.0
    by_kind: dict[str, int] = {}
    for r in fail:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    ok_times = sorted(r["create_s"] for r in ok)

    print("\n================ PROBE SUMMARY ================", flush=True)
    print(f"  total={len(results)}  ok={len(ok)}  fail={len(fail)}  "
          f"success_rate={rate:.1%}  wall={wall:.1f}s", flush=True)
    if ok_times:
        p50 = ok_times[len(ok_times)//2]
        p95 = ok_times[min(len(ok_times)-1, int(len(ok_times)*0.95))]
        print(f"  ok create time: min={ok_times[0]}s p50={p50}s "
              f"p95={p95}s max={ok_times[-1]}s", flush=True)
    if args.max_retries > 0:
        attempts = [r.get("attempts", 1) for r in results]
        ok_multi = sum(1 for r in ok if r.get("attempts", 1) > 1)
        print(f"  retries: max_attempts_seen={max(attempts) if attempts else 0} "
              f"succeeded_after_retry={ok_multi}", flush=True)
    if by_kind:
        print(f"  failure breakdown: {by_kind}", flush=True)
        if by_kind.get("S3_BUILD_CONTEXT"):
            print("  >>> S3 build-context failures present — the known outage signature. "
                  "Daytona S3 is NOT healthy under this load.", flush=True)
    verdict = rate >= args.min_success_rate
    print(f"  VERDICT: {'PASS — safe to launch' if verdict else 'FAIL — do NOT launch'} "
          f"(threshold {args.min_success_rate:.0%})", flush=True)
    print("==============================================", flush=True)
    return 0 if verdict else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Concurrent Daytona image-build probe (launch gate).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tasks-dir", default="/tmp/tmax_harbor_15k_full",
                    help="dir of <task>/environment/Dockerfile (the training pool)")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="number of image-builds to fire concurrently "
                         "(use 280 to mirror the real training fan-out)")
    ap.add_argument("--min-success-rate", type=float, default=0.90,
                    help="exit 0 iff success_rate >= this (the launch gate)")
    ap.add_argument("--rate-per-sec", type=float, default=0.0,
                    help="throttle creates to this many/sec (0=off). Mirrors the "
                         "training's harbor_create_rate_per_sec. Use 8.0 to validate "
                         "staying under Daytona's 600/min (=10/s) per-org limit.")
    ap.add_argument("--rate-burst", type=float, default=0.0,
                    help="token-bucket burst capacity (0 = ~1s worth = rate)")
    ap.add_argument("--max-retries", type=int, default=0,
                    help="retries per create on failure (0=off). Daytona suggested "
                         "exponential backoff on create failures.")
    ap.add_argument("--backoff-base", type=float, default=2.0,
                    help="exponential backoff base delay (sec): base*2^(n-1)")
    ap.add_argument("--backoff-cap", type=float, default=30.0,
                    help="max backoff delay per retry (sec)")
    ap.add_argument("--patch-proxy", action="store_true",
                    help="apply ensure_daytona_s3_proxy() so obstore build-context "
                         "uploads use the egress proxy (the real training fix)")
    ap.add_argument("--build-timeout", type=int, default=420,
                    help="per-sandbox create timeout (sec)")
    ap.add_argument("--auto-stop", type=int, default=5,
                    help="sandbox auto-stop interval (min) — short, we delete immediately")
    ap.add_argument("--auto-delete", type=int, default=5,
                    help="sandbox auto-delete interval (min) — backstop if delete() misses")
    ap.add_argument("--seed", type=int, default=12345, help="Dockerfile sampling seed")
    ap.add_argument("--env-file",
                    default="/path/to/workspace/.slime_run_env/daytona_tier4.env",
                    help="shell env file with DAYTONA_API_KEY etc. (KEY6 by default)")
    ap.add_argument("--proxy", default="http://star-proxy.oa.com:3128",
                    help="egress proxy for Daytona API (these nodes need star-proxy)")
    ap.add_argument("--no-proxy-setup", action="store_true",
                    help="do not touch *_PROXY env vars")
    args = ap.parse_args()

    _load_env_file(args.env_file)
    os.environ.setdefault("DAYTONA_API_URL", "https://app.daytona.io/api")
    if not args.no_proxy_setup:
        _set_proxy(args.proxy)

    try:
        return asyncio.run(run_probe(args))
    except KeyboardInterrupt:
        print("\n[probe] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
