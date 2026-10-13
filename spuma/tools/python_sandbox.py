"""`python_execute`: model-generated Python, run only inside a locked-down Docker container.

The container has no network, a read-only root filesystem with a small noexec tmpfs, no
capabilities, an unprivileged user, CPU/memory/PID limits, no host mounts and no host environment
variables. Without a reachable Docker daemon and a locally present image the tool is unavailable
(the registry disables it and logs why); it never falls back to host execution.
"""

import logging
import os
import shutil
import subprocess
import uuid
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from spuma.config import get_settings
from spuma.tools.base import ToolUnavailable

logger = logging.getLogger(__name__)

_DOCKER_ENV_KEYS = ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_CERT_PATH")


class SandboxUnavailable(RuntimeError):
    pass


class DockerSandbox:
    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout_seconds: float = 10.0,
        memory_mb: int = 512,
        cpus: float = 1.0,
        pids_limit: int = 64,
        max_output_chars: int = 8000,
        network_access: bool = False,
    ) -> None:
        if network_access:
            raise SandboxUnavailable("python_execute.network_access must be false")
        self.image, self.timeout, self.memory_mb = image, timeout_seconds, memory_mb
        self.cpus, self.pids_limit, self.max_output_chars = cpus, pids_limit, max_output_chars

    @staticmethod
    def _env() -> dict[str, str]:
        """The docker CLI's own environment: only what it needs to reach the daemon (no secrets)."""
        return {k: os.environ[k] for k in _DOCKER_ENV_KEYS if k in os.environ}

    def check_available(self) -> None:
        if shutil.which("docker") is None:
            raise SandboxUnavailable("docker executable not found")
        info = subprocess.run(["docker", "info"], capture_output=True, env=self._env(), timeout=20)
        if info.returncode != 0:
            raise SandboxUnavailable("docker daemon is not reachable")
        image = subprocess.run(["docker", "image", "inspect", self.image], capture_output=True, env=self._env())
        if image.returncode != 0:
            raise SandboxUnavailable(f"image {self.image!r} not present (run: docker pull {self.image})")

    def command(self, name: str) -> list[str]:
        return [
            "docker", "run", "--rm", "-i", "--name", name,
            "--network", "none",
            "--memory", f"{self.memory_mb}m", "--memory-swap", f"{self.memory_mb}m",
            "--cpus", str(self.cpus), "--pids-limit", str(self.pids_limit),
            "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534", "--workdir", "/tmp",
            self.image, "python", "-I", "-",
        ]  # fmt: skip

    def execute(self, code: str) -> dict[str, Any]:
        name = f"spuma-sandbox-{uuid.uuid4().hex[:12]}"
        try:
            proc = subprocess.run(
                self.command(name), input=code, capture_output=True, text=True, timeout=self.timeout, env=self._env()
            )
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True, env=self._env())
            return {"status": "error", "error_type": "timeout", "error": f"execution exceeded {self.timeout}s"}
        out = {
            "stdout": self._cap(proc.stdout),
            "stderr": self._cap(proc.stderr),
            "exit_code": proc.returncode,
        }
        if proc.returncode == 0:
            return {"status": "success", **out}
        kind = "memory_limit" if proc.returncode == 137 else "execution_error"
        return {"status": "error", "error_type": kind, "error": f"exit code {proc.returncode}", **out}

    def _cap(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars] + f"\n…[truncated {len(text) - self.max_output_chars} chars]"


class PythonInput(BaseModel):
    code: str = Field(max_length=20000, description="Python 3 source; print() what you need to see")


class PythonSandboxProvider:
    """`provider: python_sandbox` registry entries -> the `python_execute` LangChain tool (Docker only)."""

    def build(self, name: str) -> BaseTool:
        config = get_settings().tools.registry[name]
        opts = config.options
        sandbox = DockerSandbox(
            image=opts.get("image", "python:3.12-slim"),
            timeout_seconds=config.timeout_seconds,
            memory_mb=opts.get("memory_mb", 512),
            cpus=opts.get("cpus", 1.0),
            pids_limit=opts.get("pids_limit", 64),
            network_access=opts.get("network_access", False),
        )
        try:
            sandbox.check_available()
        except (SandboxUnavailable, OSError, subprocess.SubprocessError) as e:
            raise ToolUnavailable(f"secure sandbox unavailable: {e}") from e

        @tool(name, args_schema=PythonInput)
        def python_execute(code: str) -> dict[str, Any]:
            """Run Python 3 code in an isolated sandbox (no network, no files). Print the results you need."""
            return sandbox.execute(code)

        return python_execute
