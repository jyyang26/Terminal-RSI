#!/usr/bin/env python3
"""Convert the allenai/tmax (TMax-15k) dataset into harbor terminal-bench task dirs.

TMax-15k ships as ``tasks.zip`` containing 14601 ``task_<id>/`` dirs, each with:
    task.json              # name / description / domain / skill_type / truth ...
    container.def          # Singularity def: %files (fixtures) + %post (build shell)
    setup.sh               # runtime version of %post (unused here; %post -> Dockerfile)
    test_final_state.py    # pytest verifier logic
    test_initial_state.py  # build-time self-check (ignored)
    fixtures/ (optional)   # files COPYed into the image by %files

harbor expects terminal-bench layout (one dir per task, discovered by glob
``*/task.toml`` in slime_plugins/harbor/data_source.py):
    <task>/task.toml
    <task>/instruction.md
    <task>/environment/Dockerfile
    <task>/environment/files/...        # build context for COPY
    <task>/tests/test.sh                # writes /logs/verifier/reward.json {"reward": ..}
    <task>/tests/test_outputs.py
    <task>/tests/test_weights.json
    <task>/solution/solve.sh

The Daytona backend builds the Dockerfile on the fly when task.toml leaves
``docker_image`` empty (harbor-code-synthesis/.../environments/daytona.py:
``Image.from_dockerfile``), so tmax tasks need no prebuilt image.

The tests/test.sh driver + reward.json contract are reused verbatim from
slime_plugins/harbor/stable_tasks.py:_test_script (harbor verifier reads
/logs/verifier/reward.json per harbor/models/trial/paths.py).

Usage:
    python convert_tmax15k_to_terminal_bench.py \
        --src-zip /path/to/shared_storage/.../TMax-15k/tasks.zip \
        --out-dir /path/to/shared_storage/.../slime-terminal-RL_data/tmax15k_terminal_bench \
        [--limit N] [--workers K]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DEFAULT_APT_PROXY = "http://star-proxy.oa.com:3128"

# ---------------------------------------------------------------------------
# tests/test.sh — reused verbatim from stable_tasks.py:_test_script.
# Runs test_outputs.py under pytest, weights via test_weights.json, and writes
# /logs/verifier/reward.json {"reward": <0..1>} which harbor's verifier reads.
# ---------------------------------------------------------------------------
def _test_script(timeout_sec: int) -> str:
    return f"""#!/usr/bin/env bash
set +e

export http_proxy="${{http_proxy:-{DEFAULT_APT_PROXY}}}"
export https_proxy="${{https_proxy:-{DEFAULT_APT_PROXY}}}"
export HTTP_PROXY="${{HTTP_PROXY:-{DEFAULT_APT_PROXY}}}"
export HTTPS_PROXY="${{HTTPS_PROXY:-{DEFAULT_APT_PROXY}}}"
mkdir -p /logs/verifier

timeout {int(timeout_sec)} python3 - <<'PY'
import json
import subprocess
import sys

try:
    import pytest
except Exception:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--break-system-packages", "pytest"],
        check=False,
    )
    import pytest

weights_path = "/tests/test_weights.json"
try:
    with open(weights_path) as f:
        weights = json.load(f)
except Exception:
    weights = {{}}

results = {{}}

class WeightedPlugin:
    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            name = report.nodeid.split("::")[-1].split("[", 1)[0]
            results[name] = bool(report.passed)

exit_code = pytest.main(["-q", "/tests/test_outputs.py"], plugins=[WeightedPlugin()])

if weights:
    denom = sum(float(v) for v in weights.values()) or 1.0
    reward = sum(float(v) for k, v in weights.items() if results.get(k, False)) / denom
else:
    reward = 1.0 if exit_code == 0 else 0.0

payload = {{
    "reward": float(reward),
    "test_reward": float(reward),
    "pytest_exit_code": int(exit_code),
}}
for name, passed in sorted(results.items()):
    safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
    payload[f"test__{{safe_name}}"] = 1.0 if passed else 0.0
with open("/logs/verifier/reward.json", "w") as f:
    json.dump(payload, f)
PY
"""


def _task_toml(task_name: str, domain: str, skill_type: str, agent_timeout: int, verifier_timeout: int) -> str:
    # docker_image intentionally omitted -> Daytona builds environment/Dockerfile.
    kw = json.dumps(["tmax-15k", domain] if domain else ["tmax-15k"])
    return f"""version = "1.0"

[task]
name = "tmax/{task_name}"
authors = []
keywords = {kw}

[metadata]
source = "tmax-15k"
domain = {json.dumps(domain or "")}
skill_type = {json.dumps(skill_type or "")}

[verifier]
timeout_sec = {float(verifier_timeout)}

[agent]
timeout_sec = {float(agent_timeout)}

[environment]
build_timeout_sec = 900.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
gpus = 0
allow_internet = true
mcp_servers = []
workdir = "/app"

[verifier.env]

[solution.env]
"""


def _parse_container_def(text: str) -> tuple[list[tuple[str, str]], str]:
    """Return (files_copies, post_shell) from a Singularity def.

    files_copies: list of (src_abs_on_build_host, dst_in_container) from %files.
    post_shell:   the raw shell body of %post.
    """
    files: list[tuple[str, str]] = []
    post_lines: list[str] = []
    section = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("%"):
            section = stripped.split()[0].lower()  # %files, %post, %environment, ...
            continue
        if section == "%files":
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                files.append((parts[0], parts[1]))
        elif section == "%post":
            post_lines.append(raw)
    return files, "\n".join(post_lines).strip("\n")


def _fixture_dst_map(files_copies: list[tuple[str, str]], src_task_dir: Path) -> list[tuple[str, str]]:
    """Map %files entries to (relpath_in_build_context, dst_in_container).

    The %files src paths point at the ORIGINAL build host (e.g. /gpfs/.../fixtures/x);
    the actual bytes live under the task's local fixtures/. We locate each by basename
    tail after 'fixtures/' and copy from src_task_dir/fixtures accordingly.
    """
    mapped: list[tuple[str, str]] = []
    for src_abs, dst in files_copies:
        # take the path tail starting at 'fixtures/' if present, else basename
        m = re.search(r"fixtures/.*$", src_abs)
        rel = m.group(0) if m else ("fixtures/" + os.path.basename(src_abs))
        mapped.append((rel, dst))
    return mapped


def _build_dockerfile(post_shell: str, fixture_copies: list[tuple[str, str]]) -> str:
    """Emit a plain (non-BuildKit) Dockerfile.

    The %post body is NOT inlined as a BuildKit heredoc (``RUN <<EOF``): those tmax
    scripts themselves contain nested ``cat <<'EOF'`` heredocs, and BuildKit-heredoc
    support is not guaranteed on the Daytona build path. Instead we COPY the whole
    %post as ``files/tmax_post.sh`` and run it with a single classic ``RUN bash`` —
    the nested heredocs then execute inside a normal shell, exactly as Singularity did.
    """
    lines = ["FROM ubuntu:22.04", "", "ENV DEBIAN_FRONTEND=noninteractive", "WORKDIR /app", ""]
    # COPY fixtures (build context = environment/files/) into the image.
    for rel, dst in fixture_copies:
        lines.append(f"COPY files/{rel} {dst}")
    if fixture_copies:
        lines.append("")
    if post_shell.strip():
        lines.append("COPY files/tmax_post.sh /tmp/tmax_post.sh")
        lines.append("RUN bash /tmp/tmax_post.sh && rm -f /tmp/tmax_post.sh")
        lines.append("")
    return "\n".join(lines)


def convert_one(src_task_dir: Path, out_dir: Path, agent_timeout: int, verifier_timeout: int) -> tuple[str, bool, str]:
    task_name = src_task_dir.name
    dst = out_dir / task_name
    done_marker = dst / ".converted"
    try:
        if done_marker.exists():
            return (task_name, True, "skip")

        task_json = json.loads((src_task_dir / "task.json").read_text(encoding="utf-8", errors="replace"))
        description = task_json.get("description") or task_json.get("instruction") or ""
        domain = str(task_json.get("domain") or "")
        skill_type = str(task_json.get("skill_type") or "")
        truth = task_json.get("truth") or ""

        container_def = (src_task_dir / "container.def").read_text(encoding="utf-8", errors="replace")
        files_copies, post_shell = _parse_container_def(container_def)
        fixture_copies = _fixture_dst_map(files_copies, src_task_dir)

        test_src = ""
        tfs = src_task_dir / "test_final_state.py"
        if tfs.exists():
            test_src = tfs.read_text(encoding="utf-8", errors="replace")

        # (re)create destination
        if dst.exists():
            shutil.rmtree(dst)
        (dst / "environment" / "files").mkdir(parents=True, exist_ok=True)
        (dst / "tests").mkdir(parents=True, exist_ok=True)
        (dst / "solution").mkdir(parents=True, exist_ok=True)

        # instruction.md
        (dst / "instruction.md").write_text(description.rstrip() + "\n", encoding="utf-8")

        # environment/files/ <- source fixtures/
        src_fixtures = src_task_dir / "fixtures"
        if src_fixtures.exists():
            shutil.copytree(src_fixtures, dst / "environment" / "files" / "fixtures")

        # %post body -> files/tmax_post.sh (COPYed + run by the Dockerfile, no BuildKit).
        if post_shell.strip():
            body = "\n".join(
                ln for ln in post_shell.splitlines() if ln.strip() != "export DEBIAN_FRONTEND=noninteractive"
            )
            post_sh = dst / "environment" / "files" / "tmax_post.sh"
            post_sh.write_text("#!/usr/bin/env bash\nset -e\n" + body.rstrip() + "\n", encoding="utf-8")
            post_sh.chmod(0o755)

        # environment/Dockerfile
        (dst / "environment" / "Dockerfile").write_text(
            _build_dockerfile(post_shell, fixture_copies), encoding="utf-8"
        )

        # tests/
        (dst / "tests" / "test_outputs.py").write_text(
            (test_src.rstrip() + "\n") if test_src else "def test_placeholder():\n    assert True\n",
            encoding="utf-8",
        )
        (dst / "tests" / "test_weights.json").write_text("{}\n", encoding="utf-8")
        test_sh = dst / "tests" / "test.sh"
        test_sh.write_text(_test_script(verifier_timeout), encoding="utf-8")
        test_sh.chmod(0o755)

        # task.toml
        (dst / "task.toml").write_text(
            _task_toml(task_name, domain, skill_type, agent_timeout, verifier_timeout), encoding="utf-8"
        )

        # solution/solve.sh (placeholder; tmax has no oracle solve script; keep truth as reference)
        solve = dst / "solution" / "solve.sh"
        truth_comment = "\n".join("# " + ln for ln in str(truth).splitlines())
        solve.write_text(
            "#!/usr/bin/env bash\n# tmax-15k has no oracle solve script.\n"
            "# Verifier config (task.json 'truth') for reference:\n"
            + (truth_comment + "\n" if truth_comment else ""),
            encoding="utf-8",
        )
        solve.chmod(0o755)

        done_marker.write_text("ok\n", encoding="utf-8")
        return (task_name, True, "ok")
    except Exception as exc:  # noqa: BLE001
        return (task_name, False, f"{type(exc).__name__}: {exc}")


def _iter_src_task_dirs(src_dir: Path):
    for child in sorted(src_dir.iterdir()):
        if child.is_dir() and (child / "task.json").exists():
            yield child


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert TMax-15k -> harbor terminal-bench tasks.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--src-zip", type=str, help="Path to TMax-15k tasks.zip")
    src.add_argument("--src-dir", type=str, help="Path to an already-unzipped dir of task_*/ dirs")
    ap.add_argument("--out-dir", type=str, required=True, help="Output dir of terminal-bench task dirs")
    ap.add_argument("--limit", type=int, default=0, help="Convert only the first N tasks (0 = all)")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8)))
    ap.add_argument("--agent-timeout", type=int, default=3600)
    ap.add_argument("--verifier-timeout", type=int, default=900)
    ap.add_argument("--unzip-dir", type=str, default="", help="Where to unzip (default: temp dir)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_ctx = None
    if args.src_zip:
        unzip_root = Path(args.unzip_dir) if args.unzip_dir else Path(tempfile.mkdtemp(prefix="tmax15k_unzip_"))
        unzip_root.mkdir(parents=True, exist_ok=True)
        print(f"[unzip] {args.src_zip} -> {unzip_root}", flush=True)
        with zipfile.ZipFile(args.src_zip) as zf:
            zf.extractall(unzip_root)
        # tasks.zip may extract to unzip_root/ directly (task_*/) or nested one level.
        if any(p.name.startswith("task_") and p.is_dir() for p in unzip_root.iterdir()):
            src_dir = unzip_root
        else:
            subs = [p for p in unzip_root.iterdir() if p.is_dir()]
            src_dir = subs[0] if len(subs) == 1 else unzip_root
        tmp_ctx = unzip_root if not args.unzip_dir else None
    else:
        src_dir = Path(args.src_dir)

    task_dirs = list(_iter_src_task_dirs(src_dir))
    if args.limit and args.limit > 0:
        task_dirs = task_dirs[: args.limit]
    total = len(task_dirs)
    print(f"[convert] {total} tasks  src={src_dir}  out={out_dir}  workers={args.workers}", flush=True)

    ok = 0
    fail = 0
    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(convert_one, td, out_dir, args.agent_timeout, args.verifier_timeout): td.name
            for td in task_dirs
        }
        done = 0
        for fut in as_completed(futs):
            name, success, msg = fut.result()
            done += 1
            if success:
                ok += 1
            else:
                fail += 1
                failures.append((name, msg))
            if done % 500 == 0 or done == total:
                print(f"  [{done}/{total}] ok={ok} fail={fail}", flush=True)

    print(f"[done] ok={ok} fail={fail} total={total}", flush=True)
    if failures:
        print("[failures] first 20:", flush=True)
        for name, msg in failures[:20]:
            print(f"  {name}: {msg}", flush=True)

    produced = len(list(out_dir.glob("*/task.toml")))
    print(f"[verify] task.toml count under {out_dir}: {produced}", flush=True)

    if tmp_ctx is not None:
        shutil.rmtree(tmp_ctx, ignore_errors=True)

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
