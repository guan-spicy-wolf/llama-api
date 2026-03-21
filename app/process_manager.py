"""Process manager for llama-server."""
import asyncio
import os
import socket
import time
from pathlib import Path
from typing import Optional

from .models import ServerStatus, ServerInfo
from .config import get_config


def find_free_port(start_port: int = 18080, max_tries: int = 100) -> int:
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError("No free port found")


class ProcessManager:
    """Manages llama-server process lifecycle."""

    def __init__(self):
        self._process: Optional[asyncio.subprocess.Process] = None
        self._model: Optional[str] = None
        self._port: Optional[int] = None
        self._started_at: Optional[float] = None
        self._error: Optional[str] = None
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None

    @property
    def status(self) -> ServerInfo:
        """Get current server status."""
        if self._error:
            return ServerInfo(
                status=ServerStatus.ERROR,
                model=self._model,
                error=self._error,
            )

        if self._process is None:
            return ServerInfo(status=ServerStatus.STOPPED)

        if self._process.returncode is not None:
            return ServerInfo(
                status=ServerStatus.STOPPED,
                model=self._model,
                error=f"Process exited with code {self._process.returncode}",
            )

        return ServerInfo(
            status=ServerStatus.RUNNING,
            model=self._model,
            pid=self._process.pid,
            port=self._port,
            started_at=self._started_at,
        )

    async def start(self, model_name: str, port: Optional[int] = None) -> ServerInfo:
        """Start llama-server with specified model."""
        config = get_config()
        model_config = config.get_model_config(model_name)

        if not model_config:
            raise ValueError(f"Model '{model_name}' not found")

        if self._process and self._process.returncode is None:
            raise RuntimeError(f"Server already running with model '{self._model}'")

        model_path = config.models_dir / model_config.file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Get effective args
        args = config.get_effective_args(model_name)
        # Use internal port (auto-assign if not specified)
        self._port = find_free_port()
        self._model = model_name
        self._error = None

        # Build command
        cmd = [
            str(config.llama_server),
            "-m", str(model_path),
            "--alias", model_name,
            "--port", str(self._port),
            "--host", "0.0.0.0",
        ]

        # Add model args
        args_dict = args.model_dump(exclude_unset=True, exclude_none=True)
        for key, value in args_dict.items():
            arg_name = key.replace("_", "-")
            cmd.extend([f"--{arg_name}", str(value)])

        # Prepare environment
        env = os.environ.copy()
        if config.ld_library_path:
            current_ld = env.get("LD_LIBRARY_PATH", "")
            if current_ld:
                env["LD_LIBRARY_PATH"] = f"{config.ld_library_path}:{current_ld}"
            else:
                env["LD_LIBRARY_PATH"] = config.ld_library_path

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            self._started_at = time.time()

            # Start output readers (non-blocking)
            self._stdout_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._read_stderr())

            # Wait a moment to check if process starts successfully
            await asyncio.sleep(0.5)

            if self._process.returncode is not None:
                raise RuntimeError(f"Process exited immediately with code {self._process.returncode}")

            return self.status

        except Exception as e:
            self._error = str(e)
            self._process = None
            raise

    async def stop(self) -> ServerInfo:
        """Stop llama-server."""
        if self._process is None or self._process.returncode is not None:
            return ServerInfo(status=ServerStatus.STOPPED)

        try:
            self._process.terminate()
            await asyncio.wait_for(self._process.wait(), timeout=10)
        except asyncio.TimeoutError:
            self._process.kill()
            await self._process.wait()
        finally:
            if self._stdout_task:
                self._stdout_task.cancel()
            if self._stderr_task:
                self._stderr_task.cancel()
            self._process = None
            self._model = None
            self._port = None
            self._started_at = None

        return ServerInfo(status=ServerStatus.STOPPED)

    async def _read_stdout(self) -> None:
        """Read and log stdout."""
        if not self._process or not self._process.stdout:
            return
        try:
            async for line in self._process.stdout:
                print(f"[llama-server stdout] {line.decode().strip()}")
        except asyncio.CancelledError:
            pass

    async def _read_stderr(self) -> None:
        """Read and log stderr."""
        if not self._process or not self._process.stderr:
            return
        try:
            async for line in self._process.stderr:
                print(f"[llama-server stderr] {line.decode().strip()}")
        except asyncio.CancelledError:
            pass


# Global process manager
_process_manager: Optional[ProcessManager] = None


def get_process_manager() -> ProcessManager:
    """Get global process manager instance."""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager