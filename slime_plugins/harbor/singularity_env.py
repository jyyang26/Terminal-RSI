from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import tempfile
from pathlib import Path

import httpx

from slime_plugins.harbor.import_utils import ensure_harbor_distribution_metadata

ensure_harbor_distribution_metadata()

from harbor.environments.singularity.singularity import SingularityEnvironment
from harbor.models.trial.paths import EnvironmentPaths


class SlimeSingularityEnvironment(SingularityEnvironment):
    """Singularity environment using a slime-provided bootstrap script.

    Harbor's default bootstrap creates a venv and has a short get-pip timeout.
    In the current Jizhi Apptainer pods, that path is fragile for the prebuilt
    ubuntu SIF. This subclass leaves Harbor's runtime logic intact and only
    swaps the bootstrap script copied into the container.
    """

    def __init__(
        self,
        *args,
        singularity_fakeroot: bool = True,
        singularity_drop_caps: str | None = None,
        singularity_bind_host_proc: bool = False,
        singularity_disable_memory_watchdog: bool = False,
        singularity_memory_limit_mb: int | None = None,
        singularity_memory_limit_multiplier: float | None = None,
        singularity_registry_proxy: str | None = None,
        singularity_registry_no_proxy: str | None = None,
        singularity_registry_mirror: str | None = None,
        **kwargs,
    ):
        self._singularity_fakeroot = singularity_fakeroot
        self._singularity_drop_caps = singularity_drop_caps
        self._singularity_bind_host_proc = singularity_bind_host_proc
        self._singularity_disable_memory_watchdog = singularity_disable_memory_watchdog
        self._singularity_memory_limit_mb = singularity_memory_limit_mb
        self._singularity_memory_limit_multiplier = singularity_memory_limit_multiplier
        self._singularity_registry_proxy = singularity_registry_proxy
        self._singularity_registry_no_proxy = singularity_registry_no_proxy
        self._singularity_registry_mirror = singularity_registry_mirror.strip("/") if singularity_registry_mirror else None
        super().__init__(*args, **kwargs)
        if self._singularity_memory_limit_mb is not None:
            self._memory_limit_bytes = int(self._singularity_memory_limit_mb) * 1024 * 1024
        elif self._singularity_memory_limit_multiplier is not None:
            self._memory_limit_bytes = int(self._memory_limit_bytes * float(self._singularity_memory_limit_multiplier))

    async def _memory_watchdog(self) -> None:
        if self._singularity_disable_memory_watchdog:
            self.logger.info("Slime Singularity memory watchdog disabled")
            return
        await super()._memory_watchdog()

    async def _convert_docker_to_sif(self, docker_image: str, *, force_pull: bool = False) -> Path:
        keys = (
            "http_proxy",
            "https_proxy",
            "ftp_proxy",
            "no_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "FTP_PROXY",
            "NO_PROXY",
        )
        old_env = {key: os.environ.get(key) for key in keys}
        try:
            self._set_registry_proxy_env()
            if self._singularity_registry_mirror and not docker_image.endswith(".sif"):
                return await self._convert_docker_to_sif_via_mirror(docker_image, force_pull=force_pull)
            return await super()._convert_docker_to_sif(docker_image, force_pull=force_pull)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _set_registry_proxy_env(self) -> None:
        if not self._singularity_registry_proxy:
            return
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ[key] = self._singularity_registry_proxy
        os.environ["ftp_proxy"] = self._singularity_registry_proxy
        os.environ["FTP_PROXY"] = self._singularity_registry_proxy
        if self._singularity_registry_no_proxy is not None:
            os.environ["no_proxy"] = self._singularity_registry_no_proxy
            os.environ["NO_PROXY"] = self._singularity_registry_no_proxy

    async def _convert_docker_to_sif_via_mirror(self, docker_image: str, *, force_pull: bool = False) -> Path:
        if ":" not in docker_image:
            docker_image = f"{docker_image}:latest"

        safe_name = docker_image.replace("/", "_").replace(":", "_")
        sif_path = self._image_cache_dir / f"{safe_name}.sif"
        if not force_pull and sif_path.exists():
            self.logger.debug(f"Using cached Singularity image: {sif_path}")
            return sif_path

        mirrored_image = f"{self._singularity_registry_mirror}/{docker_image}"
        self.logger.info("Pulling Docker image via registry mirror: %s -> %s", docker_image, mirrored_image)
        mirrored_sif_path = await super()._convert_docker_to_sif(mirrored_image, force_pull=force_pull)

        self._image_cache_dir.mkdir(parents=True, exist_ok=True)
        if sif_path.exists():
            return sif_path
        shutil.copy2(mirrored_sif_path, sif_path)
        return sif_path

    async def _start_server(self) -> None:
        await self._cleanup_server_resources()

        self._staging_dir = Path(tempfile.mkdtemp(prefix="singularity_staging_"))
        self._staging_dir.chmod(0o755)

        server_module = importlib.import_module("harbor.environments.singularity.server")
        server_script = Path(server_module.__file__)
        staging_server = self._staging_dir / "_hbexec.py"
        shutil.copy(server_script, staging_server)
        server_text = staging_server.read_text()
        staging_server.write_text(server_text.replace('host="127.0.0.1"', 'host="0.0.0.0"'))

        bootstrap_src = Path(__file__).parent / "bootstrap_singularity.sh"
        bootstrap_script = self._staging_dir / "bootstrap.sh"
        shutil.copy(bootstrap_src, bootstrap_script)
        bootstrap_script.chmod(0o755)

        max_port_retries = 3
        last_error = None

        for port_attempt in range(max_port_retries):
            reserved_socket, port = self._reserve_port()
            self._server_port = port

            env_files_dir = self.environment_dir / "files"
            bind_mounts = [
                "-B",
                f"{self._staging_dir}:/staging",
                "-B",
                f"{self.trial_paths.verifier_dir}:{EnvironmentPaths.verifier_dir}",
                "-B",
                f"{self.trial_paths.agent_dir}:{EnvironmentPaths.agent_dir}",
                "-B",
                f"{self.trial_paths.artifacts_dir}:{EnvironmentPaths.artifacts_dir}",
            ]
            if Path("/etc/resolv.conf").exists():
                bind_mounts.extend(["-B", "/etc/resolv.conf:/etc/resolv.conf"])
            if self._singularity_bind_host_proc and Path("/proc").exists():
                bind_mounts.extend(["-B", "/proc:/proc"])
            if env_files_dir.exists():
                bind_mounts.extend(["-B", f"{env_files_dir}:/staging/env_files"])

            no_mount_args: list[str] = []
            singularity_no_mount = self._singularity_no_mount
            if singularity_no_mount is None:
                singularity_no_mount = "home,tmp,bind-paths"
            if singularity_no_mount:
                for part in singularity_no_mount.split(","):
                    part = part.strip()
                    if part:
                        no_mount_args.extend(["--no-mount", part])

            bootstrap_cmd = [
                "bash",
                "-c",
                'exec /staging/bootstrap.sh "$@"',
                "bash",
                self._workdir,
                "/staging/_hbexec.py",
                "--port",
                str(self._server_port),
                "--workdir",
                self._workdir,
            ]
            fakeroot_args = ["--fakeroot"] if self._singularity_fakeroot else []
            drop_caps_args = ["--drop-caps", self._singularity_drop_caps] if self._singularity_drop_caps else []
            cmd = [
                "singularity",
                "exec",
                *drop_caps_args,
                *no_mount_args,
                "--pwd",
                self._workdir,
                "--writable-tmpfs",
                *fakeroot_args,
                "--containall",
                "--pid",
                *bind_mounts,
                str(self._sif_path),
                *bootstrap_cmd,
            ]

            self.logger.info(
                "Starting Slime Singularity container with server on port "
                f"{self._server_port} (attempt {port_attempt + 1}/{max_port_retries})"
            )

            reserved_socket.close()

            self._server_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd="/tmp",
            )

            self._stream_task = asyncio.create_task(self._stream_server_output())
            self._http_client = httpx.AsyncClient(timeout=30.0)
            server_ready = False

            for _ in range(240):
                try:
                    response = await self._http_client.get(f"http://localhost:{self._server_port}/health")
                    if response.status_code == 200:
                        if self._server_process.returncode is not None:
                            await self._stream_task
                            last_error = RuntimeError(
                                f"Port collision on {self._server_port}: health check succeeded but server process died."
                            )
                            self.logger.warning("Health check succeeded but server process died.")
                            break
                        self.logger.info("Slime Singularity FastAPI server is ready")
                        self._memory_watchdog_task = asyncio.create_task(self._memory_watchdog())
                        server_ready = True
                        break
                except httpx.RequestError:
                    pass

                if self._server_process.returncode is not None:
                    await self._stream_task
                    last_error = RuntimeError(
                        f"Server process died on port {self._server_port}. Check trial.log for server output."
                    )
                    self.logger.warning(
                        f"Server failed to start on port {self._server_port}, will retry with new port"
                    )
                    break

                await asyncio.sleep(1)

            if server_ready:
                return

            if self._server_process and self._server_process.returncode is None:
                self._server_process.kill()
                await self._server_process.wait()
            if self._stream_task and not self._stream_task.done():
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    pass
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None

        raise last_error or RuntimeError(
            f"Failed to start Singularity FastAPI server after {max_port_retries} port attempts"
        )
