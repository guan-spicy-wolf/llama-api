"""Process manager for llama-server (via podman or direct process)."""
import asyncio
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

from .models import ProcessInfo
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


def get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    import psutil
    return psutil.virtual_memory().available / (1024 ** 3)


def estimate_model_memory_gb(model_path: Path) -> float:
    """Estimate model memory requirement from file size."""
    return model_path.stat().st_size / (1024 ** 3)


@dataclass
class RunningModel:
    """Runtime state for a loaded model."""
    process: asyncio.subprocess.Process
    container_name: str  # podman container name or 'proc-{model}' for direct mode
    port: int
    model: str
    started_at: float
    last_used_at: float
    idle_timeout: int


class ProcessManager:
    """Manages multiple llama-server instances (via podman containers or direct processes)."""

    def __init__(self):
        self._processes: Dict[str, RunningModel] = {}
        self._idle_task: Optional[asyncio.Task] = None
        self._stdout_tasks: Dict[str, asyncio.Task] = {}
        self._stderr_tasks: Dict[str, asyncio.Task] = {}

    @property
    def processes(self) -> Dict[str, RunningModel]:
        return self._processes

    def get_model(self, model_name: str) -> Optional[RunningModel]:
        return self._processes.get(model_name)

    def touch_model(self, model_name: str) -> None:
        if model_name in self._processes:
            self._processes[model_name].last_used_at = time.time()
            self._save_routing()

    async def start(self, model_name: str) -> ProcessInfo:
        """Start llama-server for the specified model."""
        config = get_config()
        model_config = config.get_model_config(model_name)

        if not model_config:
            raise ValueError(f"Model '{model_name}' not found")

        if model_name in self._processes:
            proc = self._processes[model_name]
            if proc.process.returncode is None:
                raise RuntimeError(f"Model '{model_name}' is already running")
            else:
                del self._processes[model_name]

        model_path = config.models_dir / model_config.file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Memory check
        available = get_available_memory_gb()
        required = estimate_model_memory_gb(model_path)
        if required > available * 0.8:
            raise MemoryError(
                f"Not enough memory to load '{model_name}': "
                f"need ~{required:.1f}GB, only {available:.1f}GB available"
            )

        idle_timeout = model_config.idle_timeout or config.default_idle_timeout

        # Find a free port not already used by running models
        used_ports = {p.port for p in self._processes.values()}
        port = find_free_port()
        while port in used_ports:
            port = find_free_port(port + 1)

        # Build llama-server arguments (same for both modes)
        args = config.get_effective_args(model_name)
        args_dict = args.model_dump(exclude_unset=True, exclude_none=True)

        if config.container_image:
            container_name = f"llama-{model_name}"
            # Remove any stale container from a previous crash
            stale = await asyncio.create_subprocess_exec(
                "podman", "rm", "-f", container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await stale.wait()

            cmd, container_name = self._build_podman_cmd(
                config, model_name, model_path, port, args_dict
            )
            env = None  # Container handles its own environment
        else:
            cmd, container_name = self._build_direct_cmd(
                config, model_name, model_path, port, args_dict
            )
            env = os.environ.copy()
            if config.ld_library_path:
                current_ld = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    f"{config.ld_library_path}:{current_ld}" if current_ld
                    else config.ld_library_path
                )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Wait briefly to catch immediate startup failures
            await asyncio.sleep(0.5)
            if process.returncode is not None:
                raise RuntimeError(
                    f"Process exited immediately with code {process.returncode}"
                )

            now = time.time()
            self._processes[model_name] = RunningModel(
                process=process,
                container_name=container_name,
                port=port,
                model=model_name,
                started_at=now,
                last_used_at=now,
                idle_timeout=idle_timeout,
            )

            self._stdout_tasks[model_name] = asyncio.create_task(
                self._read_stdout(model_name)
            )
            self._stderr_tasks[model_name] = asyncio.create_task(
                self._read_stderr(model_name)
            )

            if self._idle_task is None or self._idle_task.done():
                self._idle_task = asyncio.create_task(self._idle_checker())

            self._save_routing()

            return ProcessInfo(
                port=port,
                pid=process.pid,
                model=model_name,
                started_at=now,
                last_used_at=now,
                idle_timeout=idle_timeout,
            )

        except Exception:
            raise

    def _build_podman_cmd(
        self,
        config,
        model_name: str,
        model_path: Path,
        host_port: int,
        args_dict: dict,
    ) -> tuple[list[str], str]:
        """Build podman run command. Returns (cmd, container_name)."""
        container_name = f"llama-{model_name}"

        cmd = [
            "podman", "run",
            "--rm",
            "--name", container_name,
            "-p", f"{host_port}:8080",
            "-v", f"{config.models_dir}:{config.models_dir}:ro,z",
            "--device", "/dev/dri",
            "--device", "/dev/kfd",
            "--group-add", "video",
            "--security-opt", "seccomp=unconfined",
            "--ipc=host",
        ]

        if config.podman_extra_args:
            cmd.extend(config.podman_extra_args)

        cmd.append(config.container_image)

        # llama-server binary + args inside the container (port always 8080)
        cmd.append(config.container_llama_bin)
        cmd += [
            "-m", str(model_path),
            "--alias", model_name,
            "--port", "8080",
            "--host", "0.0.0.0",
        ]
        for key, value in args_dict.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        return cmd, container_name

    def _build_direct_cmd(
        self,
        config,
        model_name: str,
        model_path: Path,
        port: int,
        args_dict: dict,
    ) -> tuple[list[str], str]:
        """Build direct llama-server command. Returns (cmd, container_name)."""
        container_name = f"proc-{model_name}"

        cmd = [
            str(config.llama_server),
            "-m", str(model_path),
            "--alias", model_name,
            "--port", str(port),
            "--host", "0.0.0.0",
        ]
        for key, value in args_dict.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        return cmd, container_name

    async def unload(self, model_name: str) -> bool:
        """Unload a specific model."""
        if model_name not in self._processes:
            return False

        rm = self._processes[model_name]
        config = get_config()

        try:
            if rm.process.returncode is None:
                if config.container_image:
                    await self._stop_container(rm.container_name, rm.process)
                else:
                    rm.process.terminate()
                    try:
                        await asyncio.wait_for(rm.process.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        rm.process.kill()
                        await rm.process.wait()
        except ProcessLookupError:
            pass

        del self._processes[model_name]
        for task_dict in (self._stdout_tasks, self._stderr_tasks):
            if model_name in task_dict:
                task_dict[model_name].cancel()
                del task_dict[model_name]

        self._save_routing()
        return True

    async def _stop_container(
        self, container_name: str, process: asyncio.subprocess.Process
    ) -> None:
        """Stop a podman container and wait for its run process to exit."""
        stop_proc = await asyncio.create_subprocess_exec(
            "podman", "stop", container_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(stop_proc.wait(), process.wait()),
                timeout=30,
            )
        except asyncio.TimeoutError:
            # Force kill the container, then wait
            kill_proc = await asyncio.create_subprocess_exec(
                "podman", "kill", container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await kill_proc.wait()
            await process.wait()

    async def unload_all(self) -> None:
        """Unload all models."""
        for model_name in list(self._processes.keys()):
            await self.unload(model_name)

    def _total_loaded_memory_gb(self) -> float:
        """Sum of estimated memory for all loaded models."""
        config = get_config()
        total = 0.0
        for name in self._processes:
            mc = config.get_model_config(name)
            if mc:
                total += estimate_model_memory_gb(config.models_dir / mc.file)
        return total

    async def _idle_checker(self):
        """Background task to unload idle models using LRU when memory > 80 GB."""
        memory_limit_gb = 80.0
        idle_timeout = 1800  # 30 minutes

        while True:
            await asyncio.sleep(60)
            now = time.time()

            if self._total_loaded_memory_gb() <= memory_limit_gb:
                continue

            # Sort by least recently used
            lru_order = sorted(
                self._processes.items(),
                key=lambda kv: kv[1].last_used_at,
            )
            for model_name, rm in lru_order:
                if self._total_loaded_memory_gb() <= memory_limit_gb:
                    break
                if now - rm.last_used_at > idle_timeout:
                    print(f"[ProcessManager] Memory pressure: unloading idle model {model_name}")
                    await self.unload(model_name)

    async def _read_stdout(self, model_name: str):
        rm = self._processes.get(model_name)
        if not rm or not rm.process.stdout:
            return
        try:
            async for line in rm.process.stdout:
                print(f"[{model_name} stdout] {line.decode().strip()}")
        except asyncio.CancelledError:
            pass

    async def _read_stderr(self, model_name: str):
        rm = self._processes.get(model_name)
        if not rm or not rm.process.stderr:
            return
        try:
            async for line in rm.process.stderr:
                print(f"[{model_name} stderr] {line.decode().strip()}")
        except asyncio.CancelledError:
            pass

    def _save_routing(self):
        """Persist routing state to file."""
        config = get_config()
        data = {}
        for name, rm in self._processes.items():
            data[name] = {
                "port": rm.port,
                "pid": rm.process.pid,
                "container_name": rm.container_name,
                "model": rm.model,
                "started_at": rm.started_at,
                "last_used_at": rm.last_used_at,
                "idle_timeout": rm.idle_timeout,
            }
        try:
            with open(config.routing_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ProcessManager] Failed to save routing: {e}")

    def load_routing(self):
        """Load routing state from file (startup only — clears stale data)."""
        config = get_config()
        if not config.routing_file.exists():
            return
        try:
            with open(config.routing_file) as f:
                data = json.load(f)
            if data:
                print(
                    f"[ProcessManager] Found stale routing data for "
                    f"{list(data.keys())}, clearing"
                )
                self._save_routing()
        except Exception as e:
            print(f"[ProcessManager] Failed to load routing: {e}")


# Global process manager
_process_manager: Optional[ProcessManager] = None


def get_process_manager() -> ProcessManager:
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager
