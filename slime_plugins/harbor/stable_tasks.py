from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_SIF_IMAGE_MAP = {
    "ghcr.io/laude-institute/t-bench/python-3-13:20250620": "python-3-13_20250620.sif",
    "ghcr.io/laude-institute/t-bench/ubuntu-24-04:latest": "ubuntu-24-04_latest.sif",
}

DEFAULT_APT_PROXY = "http://star-proxy.oa.com:3128"


@dataclass
class DockerfileInstruction:
    instruction: str
    arguments: str


@dataclass
class DockerfilePlan:
    base_image: str = ""
    workdir: str = "/app"
    env_vars: dict[str, str] = field(default_factory=dict)
    setup_steps: list[tuple[str, Any]] = field(default_factory=list)


def load_stable_task_entries(
    path: str | Path,
    *,
    materialized_tasks_dir: str | Path,
    sif_cache_dir: str | Path | None = None,
    limit: int | None = None,
    offset: int = 0,
    apt_proxy: str = DEFAULT_APT_PROXY,
) -> list[dict[str, Any]]:
    """Load stable_rl_tb_rl terminal-bench rows and materialize Harbor tasks.

    Supported inputs:
    - parquet files with an ``extra_info`` column, such as stable_rl_tb_rl/data/train.parquet
    - JSON files shaped like ``tasks_info.json``

    ``tasks_info.json`` entries do not always include instructions; those rows are
    skipped because Harbor needs an ``instruction.md``.
    """

    source_path = Path(path)
    rows = _read_stable_rows(source_path)
    selected = rows[offset:]
    if limit is not None and limit > 0:
        selected = selected[:limit]

    output_root = Path(materialized_tasks_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for row in selected:
        entry = materialize_stable_task(
            row,
            output_root=output_root,
            sif_cache_dir=Path(sif_cache_dir) if sif_cache_dir else _default_sif_cache_dir(source_path),
            apt_proxy=apt_proxy,
        )
        if entry is not None:
            entries.append(entry)
    return entries


def materialize_stable_task(
    row: dict[str, Any],
    *,
    output_root: Path,
    sif_cache_dir: Path | None,
    apt_proxy: str = DEFAULT_APT_PROXY,
) -> dict[str, Any] | None:
    extra_info = _normalize_extra_info(row)
    instruction = str(extra_info.get("instruction") or "").strip()
    if not instruction:
        return None

    task_name = _sanitize_name(str(extra_info.get("task_name") or row.get("task_name") or "stable-task"))
    task_id = _stable_task_id(extra_info)
    task_dir = output_root / f"{task_name}-{task_id[:10]}"

    dockerfile_contents = str(extra_info.get("dockerfile_contents") or "FROM ubuntu:24.04\nWORKDIR /app\n")
    test_contents = str(extra_info.get("py_test_file_contents") or "")
    test_weights = _jsonish(extra_info.get("test_weights"), default={})
    additional_files = _jsonish(extra_info.get("additional_files"), default={}) or {}
    max_test_timeout_sec = int(float(extra_info.get("max_test_timeout_sec") or 300))
    source_task_path = str(extra_info.get("task_path") or "")

    plan = _parse_dockerfile_plan(dockerfile_contents)
    docker_image = _resolve_docker_image(plan.base_image, sif_cache_dir)

    _reset_dir(task_dir)
    (task_dir / "environment" / "files" / "build_context").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    (task_dir / "solution").mkdir(parents=True, exist_ok=True)

    _write_text(task_dir / "instruction.md", instruction + "\n")
    _write_text(task_dir / "environment" / "Dockerfile", dockerfile_contents.rstrip() + "\n")
    _write_text(
        task_dir / "task.toml",
        _task_toml(
            task_name=task_name,
            docker_image=docker_image,
            workdir=plan.workdir,
            timeout_sec=max_test_timeout_sec,
        ),
    )
    _write_text(task_dir / "tests" / "test_outputs.py", test_contents.rstrip() + "\n")
    _write_text(task_dir / "tests" / "test_weights.json", json.dumps(test_weights, indent=2, sort_keys=True) + "\n")
    _write_text(task_dir / "tests" / "test.sh", _test_script(max_test_timeout_sec))
    (task_dir / "tests" / "test.sh").chmod(0o755)
    _write_text(task_dir / "solution" / "solve.sh", "#!/usr/bin/env bash\n# Stable task has no oracle solution.\n")
    (task_dir / "solution" / "solve.sh").chmod(0o755)

    build_context = task_dir / "environment" / "files" / "build_context"
    if source_task_path and Path(source_task_path).exists():
        _copy_task_context(Path(source_task_path), build_context)
    for rel_path, content in additional_files.items():
        _write_text(build_context / str(rel_path).lstrip("/"), str(content))

    _write_text(
        task_dir / "environment" / "files" / "setup.sh",
        _setup_script(plan, apt_proxy=apt_proxy),
    )
    (task_dir / "environment" / "files" / "setup.sh").chmod(0o755)

    return {
        "prompt": instruction,
        "task": {"path": str(task_dir)},
        "metadata": {
            "stable_task_name": task_name,
            "stable_source": str(row.get("_source_path") or ""),
            "stable_base_image": plan.base_image,
            "stable_docker_image": docker_image,
        },
    }


def _read_stable_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Reading stable parquet tasks requires pandas/pyarrow in the runtime.") from exc
        df = pd.read_parquet(path)
        rows = []
        for record in df.to_dict(orient="records"):
            extra_info = record.get("extra_info") or {}
            if not isinstance(extra_info, dict):
                extra_info = dict(extra_info)
            record = dict(record)
            record["extra_info"] = extra_info
            record["_source_path"] = str(path)
            rows.append(record)
        return rows

    with path.open() as f:
        data = json.load(f)

    if isinstance(data, dict):
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("task_name", key)
                row["_source_path"] = str(path)
                rows.append(row)
        return rows
    if isinstance(data, list):
        return [dict(item, _source_path=str(path)) for item in data]
    raise ValueError(f"Unsupported stable task file shape: {path}")


def _normalize_extra_info(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        out = dict(extra_info)
    else:
        out = dict(row)

    if "prompt" in row and "instruction" not in out:
        prompt = row["prompt"]
        user_content = _extract_user_prompt(prompt)
        if user_content:
            out["instruction"] = user_content
    return out


def _extract_user_prompt(prompt: Any) -> str | None:
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return prompt
    try:
        messages = list(prompt)
    except TypeError:
        return None
    for message in reversed(messages):
        role = message.get("role") if isinstance(message, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if role == "user" and content:
            return str(content)
    return None


def _parse_dockerfile_plan(contents: str) -> DockerfilePlan:
    instructions = _parse_dockerfile(contents)
    plan = DockerfilePlan()
    current_workdir = plan.workdir

    for instr in instructions:
        name = instr.instruction
        args = instr.arguments
        if name == "FROM":
            plan.base_image = _strip_from_args(args)
        elif name == "WORKDIR":
            current_workdir = args.strip() or "/app"
            plan.workdir = current_workdir
            plan.setup_steps.append(("workdir", current_workdir))
        elif name == "ENV":
            env_vars = _parse_env(args)
            plan.env_vars.update(env_vars)
            plan.setup_steps.append(("env", env_vars))
        elif name == "RUN":
            plan.setup_steps.append(("run", args))
        elif name in {"COPY", "ADD"}:
            for src, dest in _parse_copy_args(args):
                plan.setup_steps.append(("copy", (src, dest)))
        elif name == "HEREDOC_COPY":
            dest, content = args.split("\n", 1)
            plan.setup_steps.append(("heredoc_copy", (dest, content)))

    if not plan.base_image:
        plan.base_image = "ghcr.io/laude-institute/t-bench/ubuntu-24-04:latest"
    return plan


def _parse_dockerfile(contents: str) -> list[DockerfileInstruction]:
    raw_lines = contents.splitlines()
    instructions: list[DockerfileInstruction] = []
    i = 0
    heredoc_re = re.compile(r"^(COPY|ADD)\s+<<-?\s*['\"]?(\w+)['\"]?\s+(.+)$", re.IGNORECASE)

    while i < len(raw_lines):
        line = raw_lines[i]
        while line.rstrip().endswith("\\") and i + 1 < len(raw_lines):
            line = line.rstrip()[:-1] + " " + raw_lines[i + 1]
            i += 1
        i += 1

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        heredoc_match = heredoc_re.match(stripped)
        if heredoc_match:
            marker = heredoc_match.group(2)
            dest = heredoc_match.group(3).strip()
            content_lines = []
            while i < len(raw_lines):
                hline = raw_lines[i]
                i += 1
                if hline.strip() == marker:
                    break
                content_lines.append(hline)
            instructions.append(DockerfileInstruction("HEREDOC_COPY", f"{dest}\n{'\n'.join(content_lines)}"))
            continue

        match = re.match(r"^(\w+)\s+(.*)", stripped, re.DOTALL)
        if match:
            instructions.append(DockerfileInstruction(match.group(1).upper(), match.group(2).strip()))
    return instructions


def _strip_from_args(args: str) -> str:
    parts = shlex.split(args)
    out = []
    for part in parts:
        if part.startswith("--platform"):
            continue
        if part.upper() == "AS":
            break
        out.append(part)
    return " ".join(out).strip()


def _parse_copy_args(args: str) -> list[tuple[str, str]]:
    parts = shlex.split(args)
    if len(parts) < 2:
        return []
    dest = parts[-1]
    sources = [part for part in parts[:-1] if not part.startswith("--")]
    return [(src, dest) for src in sources]


def _parse_env(args: str) -> dict[str, str]:
    args = args.strip()
    if not args:
        return {}
    if "=" not in args:
        parts = args.split(None, 1)
        return {parts[0]: parts[1] if len(parts) > 1 else ""}

    env_vars = {}
    for part in shlex.split(args):
        if "=" in part:
            key, value = part.split("=", 1)
            env_vars[key] = value
    return env_vars


def _setup_script(plan: DockerfilePlan, *, apt_proxy: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        f"export http_proxy={shlex.quote(apt_proxy)}",
        f"export https_proxy={shlex.quote(apt_proxy)}",
        f"export HTTP_PROXY={shlex.quote(apt_proxy)}",
        f"export HTTPS_PROXY={shlex.quote(apt_proxy)}",
        'export no_proxy="${no_proxy:-localhost,127.0.0.1,mirror.ccs.tencentyun.com}"',
        "if command -v apt-get >/dev/null 2>&1; then",
        "  mkdir -p /etc/apt/apt.conf.d",
        f"  printf '%s\\n' 'Acquire::http::Proxy \"{apt_proxy}\";' 'Acquire::https::Proxy \"{apt_proxy}\";' > /etc/apt/apt.conf.d/99slime-harbor-proxy || true",
        "fi",
        "run_step() {",
        "  local label=\"$1\"",
        "  local cmd=\"$2\"",
        "  echo \"[stable-task setup] ${label}: ${cmd}\" >&2",
        "  bash -lc \"${cmd}\" || echo \"[stable-task setup] WARNING: ${label} failed\" >&2",
        "}",
        "copy_step() {",
        "  local src=\"$1\"",
        "  local dest=\"$2\"",
        "  local base=\"/staging/env_files/build_context\"",
        "  local from=\"${base}/${src#/}\"",
        "  if [ ! -e \"$from\" ]; then",
        "    echo \"[stable-task setup] WARNING: COPY source not found: $src\" >&2",
        "    return 0",
        "  fi",
        "  if [[ \"$dest\" != /* ]]; then",
        "    dest=\"${CURRENT_WORKDIR%/}/$dest\"",
        "  fi",
        "  if [[ \"$dest\" == */ ]] || [ -d \"$dest\" ]; then",
        "    mkdir -p \"$dest\"",
        "    cp -a \"$from\" \"$dest/\" || true",
        "  else",
        "    mkdir -p \"$(dirname \"$dest\")\"",
        "    cp -a \"$from\" \"$dest\" || true",
        "  fi",
        "}",
        f"CURRENT_WORKDIR={shlex.quote(plan.workdir or '/app')}",
        "mkdir -p \"$CURRENT_WORKDIR\"",
        "cd \"$CURRENT_WORKDIR\" || true",
        "if command -v apt-get >/dev/null 2>&1; then",
        "  run_step 'apt update' 'apt-get update -qq'",
        "  run_step 'harbor terminal deps' 'apt-get install -y -qq tmux asciinema curl ca-certificates'",
        "fi",
    ]

    for key, value in plan.env_vars.items():
        lines.append(f"export {key}={shlex.quote(value)}")

    heredoc_index = 0
    for step_type, data in plan.setup_steps:
        if step_type == "workdir":
            lines.extend(
                [
                    f"CURRENT_WORKDIR={shlex.quote(str(data))}",
                    "mkdir -p \"$CURRENT_WORKDIR\"",
                    "cd \"$CURRENT_WORKDIR\" || true",
                ]
            )
        elif step_type == "env":
            for key, value in data.items():
                lines.append(f"export {key}={shlex.quote(value)}")
        elif step_type == "run":
            lines.append(f"run_step 'RUN' {shlex.quote(str(data))}")
        elif step_type == "copy":
            src, dest = data
            lines.append(f"copy_step {shlex.quote(src)} {shlex.quote(dest)}")
        elif step_type == "heredoc_copy":
            dest, content = data
            marker = f"SLIME_HARBOR_HEREDOC_{heredoc_index}"
            heredoc_index += 1
            lines.extend(
                [
                    f"mkdir -p {shlex.quote(str(Path(dest).parent))}",
                    f"cat > {shlex.quote(dest)} <<'{marker}'",
                    content,
                    marker,
                ]
            )

    return "\n".join(lines) + "\n"


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
import os
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
with open("/logs/verifier/details.json", "w") as f:
    json.dump(
        {{
            "passed_tests": sorted(k for k, v in results.items() if v),
            "failed_tests": sorted(k for k, v in results.items() if not v),
        }},
        f,
    )
PY

if [ ! -s /logs/verifier/reward.json ]; then
  printf '{{"reward": 0.0, "test_reward": 0.0, "verifier_error": 1.0}}\\n' > /logs/verifier/reward.json
fi
exit 0
"""


def _task_toml(*, task_name: str, docker_image: str, workdir: str, timeout_sec: int) -> str:
    return f"""version = "1.0"

[task]
name = "slime-harbor/{task_name}"
authors = []
keywords = ["terminal-bench", "stable-rl"]

[metadata]
source = "stable_rl_tb_rl"

[verifier]
timeout_sec = {float(timeout_sec)}

[agent]
timeout_sec = 900.0

[environment]
docker_image = {json.dumps(docker_image)}
workdir = {json.dumps(workdir or "/app")}
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
gpus = 0
allow_internet = true
mcp_servers = []

[verifier.env]

[solution.env]
"""


def _resolve_docker_image(base_image: str, sif_cache_dir: Path | None) -> str:
    if sif_cache_dir is not None:
        filename = DEFAULT_SIF_IMAGE_MAP.get(base_image)
        if filename:
            sif_path = sif_cache_dir / filename
            if sif_path.exists():
                return str(sif_path)
    return base_image


def _default_sif_cache_dir(source_path: Path) -> Path | None:
    parts = list(source_path.parents)
    for parent in parts:
        candidate = parent / "sif_cache"
        if candidate.exists():
            return candidate
    stable_default = Path("/path/to/workspace/code/stable_rl_tb_rl/data/sif_cache")
    if stable_default.exists():
        return stable_default
    local_default = Path("/path/to/workspace/code/stable_rl_tb_rl/data/sif_cache")
    if local_default.exists():
        return local_default
    return None


def _jsonish(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        if not value:
            return default
        return json.loads(value)
    return value


def _stable_task_id(extra_info: dict[str, Any]) -> str:
    basis = json.dumps(
        {
            "task_name": extra_info.get("task_name"),
            "instruction": extra_info.get("instruction"),
            "dockerfile_contents": extra_info.get("dockerfile_contents"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def _sanitize_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-._")
    return clean[:80] or "stable-task"


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_task_context(source: Path, dest: Path) -> None:
    for child in source.iterdir():
        target = dest / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
