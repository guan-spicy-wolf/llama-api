# Multi-Model Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor llama-api to support multiple concurrent models with idle timeout unloading and memory protection.

**Architecture:** Single gateway port (8002) routes requests to multiple llama-server instances on dynamic ports. ProcessManager tracks all running models, persists state to routing.json, and runs background idle checker.

**Tech Stack:** FastAPI, asyncio, Pydantic, httpx

---

## Task 1: Update Data Models

**Files:**
- Modify: `app/models.py`

**Step 1: Add ProcessInfo model**

Add after `ServerStatus` enum:

```python
from typing import Dict, Any

class ProcessInfo(BaseModel):
    """Runtime info for a loaded model."""
    port: int
    pid: int
    model: str
    started_at: float
    last_used_at: float
    idle_timeout: int  # seconds

class ServerInfo(BaseModel):
    """Current server status and info."""
    status: ServerStatus
    model: Optional[str] = None
    pid: Optional[int] = None
    port: Optional[int] = None
    started_at: Optional[float] = None
    error: Optional[str] = None

class MultiServerStatus(BaseModel):
    """Status of all loaded models."""
    models: Dict[str, ProcessInfo]
    total_memory_gb: float

class LoadRequest(BaseModel):
    """Request to load a model."""
    model: str = Field(..., description="Model name to load")

class UnloadRequest(BaseModel):
    """Request to unload a model."""
    model: str = Field(..., description="Model name to unload")

class LoadResponse(BaseModel):
    """Response to load request."""
    success: bool
    message: str
    pid: Optional[int] = None
    port: Optional[int] = None
```

**Step 2: Update ModelConfig to include idle_timeout**

```python
class ModelConfig(BaseModel):
    """Configuration for a single model."""
    name: str = Field(..., description="Model name identifier")
    file: str = Field(..., description="GGUF file name")
    args: Optional[ModelArgs] = Field(default=None, description="Model-specific arguments")
    idle_timeout: Optional[int] = Field(default=None, description="Idle timeout in seconds, overrides global default")

    model_config = {"extra": "allow"}
```

**Step 3: Commit**

```bash
git add app/models.py
git commit -m "feat: add ProcessInfo and MultiServerStatus models for multi-model support"
```

---

## Task 2: Update Config for Global Settings

**Files:**
- Modify: `app/config.py`

**Step 1: Add global config fields**

Add to `AppConfig` class:

```python
class AppConfig:
    """Application configuration."""

    def __init__(self):
        self.models_dir: Path = Path(os.getenv("LLAMA_MODELS_DIR", "/var/models"))
        self.models_config_dir: Path = Path(os.getenv("LLAMA_MODELS_CONFIG_DIR", "/var/containers/llama-api/models.d"))
        self.llama_server: Path = Path(os.getenv("LLAMA_SERVER", "/usr/local/bin/llama-server"))
        self.ld_library_path: Optional[str] = os.getenv("LLAMA_LD_LIBRARY_PATH")
        self.default_idle_timeout: int = int(os.getenv("LLAMA_DEFAULT_IDLE_TIMEOUT", "300"))  # 5 minutes
        self.data_dir: Path = Path(os.getenv("LLAMA_DATA_DIR", "/var/containers/llama-api"))
        self.routing_file: Path = self.data_dir / "routing.json"
```

**Step 2: Ensure data_dir exists on init**

Add to `init_config()`:

```python
def init_config():
    """Initialize configuration singleton."""
    global _config
    _config = AppConfig()
    # Ensure data directory exists
    _config.data_dir.mkdir(parents=True, exist_ok=True)
```

**Step 3: Commit**

```bash
git add app/config.py
git commit -m "feat: add global idle_timeout and data_dir config"
```

---

## Task 3: Refactor ProcessManager for Multi-Model

**Files:**
- Modify: `app/process_manager.py`

**Step 1: Rewrite ProcessManager class**

Replace the entire `ProcessManager` class with:

```python
"""Process manager for llama-server."""
import asyncio
import json
import os
import signal
import socket
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict

from .models import ProcessInfo, ServerStatus
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
    # Fallback
    import psutil
    return psutil.virtual_memory().available / (1024 ** 3)


def estimate_model_memory_gb(model_path: Path) -> float:
    """Estimate model memory requirement from file size."""
    # GGUF file size is roughly the memory needed
    return model_path.stat().st_size / (1024 ** 3)


@dataclass
class RunningModel:
    """Runtime state for a loaded model."""
    process: asyncio.subprocess.Process
    port: int
    model: str
    started_at: float
    last_used_at: float
    idle_timeout: int

    def to_persist(self) -> dict:
        """Get persistable state (without process object)."""
        return {
            "port": self.port,
            "pid": self.process.pid,
            "model": self.model,
            "started_at": self.started_at,
            "last_used_at": self.last_used_at,
            "idle_timeout": self.idle_timeout,
        }


class ProcessManager:
    """Manages multiple llama-server processes."""

    def __init__(self):
        self._processes: Dict[str, RunningModel] = {}
        self._idle_task: Optional[asyncio.Task] = None
        self._stdout_tasks: Dict[str, asyncio.Task] = {}
        self._stderr_tasks: Dict[str, asyncio.Task] = {}

    @property
    def processes(self) -> Dict[str, RunningModel]:
        """Get all running processes."""
        return self._processes

    def get_model(self, model_name: str) -> Optional[RunningModel]:
        """Get a specific model's process info."""
        return self._processes.get(model_name)

    def touch_model(self, model_name: str) -> None:
        """Update last_used_at for a model."""
        if model_name in self._processes:
            self._processes[model_name].last_used_at = time.time()
            self._save_routing()

    async def start(self, model_name: str) -> ProcessInfo:
        """Start llama-server with specified model."""
        config = get_config()
        model_config = config.get_model_config(model_name)

        if not model_config:
            raise ValueError(f"Model '{model_name}' not found")

        if model_name in self._processes:
            proc = self._processes[model_name]
            if proc.process.returncode is None:
                raise RuntimeError(f"Model '{model_name}' is already running")
            else:
                # Clean up dead process
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

        # Get idle timeout
        idle_timeout = model_config.idle_timeout or config.default_idle_timeout

        # Find port
        used_ports = {p.port for p in self._processes.values()}
        port = find_free_port()
        while port in used_ports:
            port = find_free_port(port + 1)

        # Build command
        args = config.get_effective_args(model_name)
        cmd = [
            str(config.llama_server),
            "-m", str(model_path),
            "--alias", model_name,
            "--port", str(port),
            "--host", "0.0.0.0",
        ]

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
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Wait briefly to check immediate failure
            await asyncio.sleep(0.5)
            if process.returncode is not None:
                raise RuntimeError(f"Process exited immediately with code {process.returncode}")

            now = time.time()
            self._processes[model_name] = RunningModel(
                process=process,
                port=port,
                model=model_name,
                started_at=now,
                last_used_at=now,
                idle_timeout=idle_timeout,
            )

            # Start output readers
            self._stdout_tasks[model_name] = asyncio.create_task(
                self._read_stdout(model_name)
            )
            self._stderr_tasks[model_name] = asyncio.create_task(
                self._read_stderr(model_name)
            )

            # Start idle checker if not running
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

        except Exception as e:
            raise

    async def unload(self, model_name: str) -> bool:
        """Unload a specific model."""
        if model_name not in self._processes:
            return False

        rm = self._processes[model_name]

        try:
            if rm.process.returncode is None:
                rm.process.terminate()
                try:
                    await asyncio.wait_for(rm.process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    rm.process.kill()
                    await rm.process.wait()
        except ProcessLookupError:
            pass

        # Clean up
        del self._processes[model_name]
        if model_name in self._stdout_tasks:
            self._stdout_tasks[model_name].cancel()
            del self._stdout_tasks[model_name]
        if model_name in self._stderr_tasks:
            self._stderr_tasks[model_name].cancel()
            del self._stderr_tasks[model_name]

        self._save_routing()
        return True

    async def unload_all(self) -> None:
        """Unload all models."""
        for model_name in list(self._processes.keys()):
            await self.unload(model_name)

    async def _idle_checker(self):
        """Background task to check for idle models."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            now = time.time()
            for model_name, rm in list(self._processes.items()):
                if now - rm.last_used_at > rm.idle_timeout:
                    print(f"[ProcessManager] Unloading idle model: {model_name}")
                    await self.unload(model_name)

    async def _read_stdout(self, model_name: str):
        """Read and log stdout for a model."""
        rm = self._processes.get(model_name)
        if not rm or not rm.process.stdout:
            return
        try:
            async for line in rm.process.stdout:
                print(f"[{model_name} stdout] {line.decode().strip()}")
        except asyncio.CancelledError:
            pass

    async def _read_stderr(self, model_name: str):
        """Read and log stderr for a model."""
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
        """Load routing state from file. Called on startup."""
        config = get_config()
        if not config.routing_file.exists():
            return
        try:
            with open(config.routing_file) as f:
                data = json.load(f)
            # Note: We can't restore actual processes, just clear stale data
            # The models will need to be reloaded
            if data:
                print(f"[ProcessManager] Found stale routing data for {list(data.keys())}, clearing")
                self._save_routing()  # Clear it
        except Exception as e:
            print(f"[ProcessManager] Failed to load routing: {e}")


# Global process manager
_process_manager: Optional[ProcessManager] = None


def get_process_manager() -> ProcessManager:
    """Get global process manager instance."""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager
```

**Step 2: Commit**

```bash
git add app/process_manager.py
git commit -m "feat: refactor ProcessManager for multi-model support with idle timeout"
```

---

## Task 4: Update Server Routes

**Files:**
- Modify: `app/routes/server.py`

**Step 1: Rewrite server routes**

Replace entire file with:

```python
"""Server control API routes."""
import time

import psutil
from fastapi import APIRouter, HTTPException

from ..models import LoadRequest, LoadResponse, UnloadRequest, MultiServerStatus, ProcessInfo
from ..process_manager import get_process_manager
from ..config import get_config

router = APIRouter(prefix="/api/server", tags=["server"])


@router.get("/status")
async def get_status() -> dict:
    """Get status of all loaded models."""
    pm = get_process_manager()

    models = {}
    for name, rm in pm.processes.items():
        # Check if process is still alive
        if rm.process.returncode is not None:
            continue

        idle_remaining = rm.idle_timeout - (time.time() - rm.last_used_at)
        models[name] = {
            "port": rm.port,
            "pid": rm.process.pid,
            "model": rm.model,
            "started_at": rm.started_at,
            "last_used_at": rm.last_used_at,
            "idle_timeout": rm.idle_timeout,
            "idle_remaining": max(0, idle_remaining),
        }

    # Get total process memory
    total_mem = 0.0
    for name, rm in pm.processes.items():
        try:
            proc = psutil.Process(rm.process.pid)
            total_mem += proc.memory_info().rss / (1024 ** 3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return {
        "models": models,
        "total_memory_gb": round(total_mem, 1),
    }


@router.get("/status/{model_name}")
async def get_model_status(model_name: str) -> dict:
    """Get status of a specific model."""
    pm = get_process_manager()
    rm = pm.get_model(model_name)

    if not rm or rm.process.returncode is not None:
        return {"status": "not_loaded", "model": model_name}

    idle_remaining = rm.idle_timeout - (time.time() - rm.last_used_at)
    return {
        "status": "running",
        "port": rm.port,
        "pid": rm.process.pid,
        "model": rm.model,
        "started_at": rm.started_at,
        "last_used_at": rm.last_used_at,
        "idle_timeout": rm.idle_timeout,
        "idle_remaining": max(0, idle_remaining),
    }


@router.get("/metrics")
async def get_metrics() -> dict:
    """Get system metrics (CPU, memory, GPU)."""
    cpu_percent = psutil.cpu_percent(interval=0.5)

    mem = psutil.virtual_memory()
    mem_total_gb = round(mem.total / (1024 ** 3), 1)
    mem_used_gb = round(mem.used / (1024 ** 3), 1)
    mem_percent = mem.percent

    # GPU utilization (ROCm)
    gpu_util_percent = None
    gpu_temp = None
    try:
        import subprocess
        result = subprocess.run(
            ["rocm-smi"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if line.strip() and line[0].isdigit() and "Device" not in line:
                    parts = line.split()
                    if len(parts) >= 16:
                        gpu_str = parts[15].rstrip('%')
                        if gpu_str.replace('.', '').isdigit():
                            gpu_util_percent = float(gpu_str)
                        temp_str = parts[4].rstrip('°C')
                        if temp_str.replace('.', '').isdigit():
                            gpu_temp = float(temp_str)
                        break
    except Exception:
        pass

    return {
        "cpu": {"percent": cpu_percent},
        "memory": {
            "total_gb": mem_total_gb,
            "used_gb": mem_used_gb,
            "percent": mem_percent,
        },
        "gpu": {
            "util_percent": gpu_util_percent,
            "temp": gpu_temp,
        } if gpu_util_percent is not None else None,
    }


@router.post("/load", response_model=LoadResponse)
async def load_model(request: LoadRequest) -> LoadResponse:
    """Load a model and start llama-server."""
    pm = get_process_manager()
    config = get_config()

    if not config.get_model_config(request.model):
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")

    try:
        info = await pm.start(request.model)
        return LoadResponse(
            success=True,
            message=f"Model '{request.model}' loaded successfully",
            pid=info.pid,
            port=info.port,
        )
    except MemoryError as e:
        raise HTTPException(status_code=507, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload")
async def unload_model(request: UnloadRequest) -> dict:
    """Unload a specific model."""
    pm = get_process_manager()
    success = await pm.unload(request.model)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not loaded")
    return {"message": f"Model '{request.model}' unloaded", "model": request.model}


@router.post("/unload/all")
async def unload_all_models() -> dict:
    """Unload all models."""
    pm = get_process_manager()
    await pm.unload_all()
    return {"message": "All models unloaded"}
```

**Step 2: Commit**

```bash
git add app/routes/server.py
git commit -m "feat: update server routes for multi-model support"
```

---

## Task 5: Update Main Proxy Routing

**Files:**
- Modify: `app/main.py`

**Step 1: Update proxy_v1 function**

Replace the `proxy_v1` function with:

```python
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy_v1(request: Request, path: str):
    """Proxy /v1/* requests to appropriate llama-server based on model."""
    pm = get_process_manager()

    # Get request body to extract model
    body = await request.body()
    model_name = None

    if body:
        try:
            import json
            body_json = json.loads(body)
            model_name = body_json.get("model")
        except Exception:
            pass

    if not model_name:
        return Response(
            content='{"error": "No model specified in request body"}',
            status_code=400,
            media_type="application/json"
        )

    rm = pm.get_model(model_name)
    if not rm or rm.process.returncode is not None:
        return Response(
            content=f'{{"error": "Model \'{model_name}\' not loaded. Load via /api/server/load"}}',
            status_code=503,
            media_type="application/json"
        )

    # Update last used time
    pm.touch_model(model_name)

    # Build target URL
    target_url = f"http://127.0.0.1:{rm.port}/v1/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    # Prepare headers (exclude host)
    headers = dict(request.headers)
    headers.pop("host", None)

    # Determine if streaming
    accept = request.headers.get("accept", "")
    is_streaming = "text/event-stream" in accept
    if not is_streaming and body:
        try:
            import json
            body_json = json.loads(body)
            if body_json.get("stream") is True:
                is_streaming = True
        except Exception:
            pass

    if is_streaming:
        client_timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
    else:
        client_timeout = httpx.Timeout(connect=30.0, read=3600.0, write=30.0, pool=30.0)

    async with httpx.AsyncClient(timeout=client_timeout) as client:
        try:
            proxy_request = client.build_request(
                method=request.method,
                url=target_url,
                content=body,
                headers=headers,
            )

            response = await client.send(proxy_request, stream=True)

            if is_streaming:
                async def stream_generator():
                    async for chunk in response.aiter_bytes():
                        yield chunk
                    await response.aclose()

                return StreamingResponse(
                    stream_generator(),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type="text/event-stream"
                )
            else:
                content = await response.aread()
                await response.aclose()
                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
        except httpx.ConnectError:
            return Response(
                content=f'{{"error": "Cannot connect to llama-server for model \'{model_name}\'"}}',
                status_code=503,
                media_type="application/json"
            )
```

**Step 2: Update lifespan to unload all on shutdown**

Replace lifespan function:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    init_config()
    yield
    # Shutdown - stop all llama-servers
    pm = get_process_manager()
    await pm.unload_all()
```

**Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat: route /v1/* requests to correct model based on request body"
```

---

## Task 6: Update WebUI

**Files:**
- Modify: `webui/index.html`
- Modify: `webui/app.js`

**Step 1: Update index.html model list section**

Replace the model-related HTML sections to support multiple models with individual load/unload buttons. Add idle countdown display.

**Step 2: Update app.js for multi-model**

- Fetch `/api/server/status` to get all loaded models
- Show each model as a card with load/unload button
- Display idle remaining time countdown
- Update memory display to show per-model breakdown

**Step 3: Commit**

```bash
git add webui/
git commit -m "feat: update WebUI for multi-model support"
```

---

## Task 7: Test Multi-Model Functionality

**Step 1: Restart llama-api service**

```bash
systemctl --user restart llama-api
```

**Step 2: Test loading multiple models**

```bash
# Load first model
curl -X POST http://localhost:8002/api/server/load \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-claude"}'

# Check status
curl http://localhost:8002/api/server/status

# Load second model (if you have one configured)
curl -X POST http://localhost:8002/api/server/load \
  -H "Content-Type: application/json" \
  -d '{"model": "minimax"}'

# Check status again
curl http://localhost:8002/api/server/status
```

**Step 3: Test routing**

```bash
# Request to qwen model
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-claude", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}'
```

**Step 4: Test unload**

```bash
curl -X POST http://localhost:8002/api/server/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-claude"}'
```

---

## Summary

After implementation:
- Multiple models can run concurrently on different internal ports
- Requests are routed by `model` field in request body
- Models auto-unload after idle timeout (5 min default)
- Memory protection prevents overloading
- State persisted to `/var/containers/llama-api/routing.json`
- WebUI shows all loaded models with individual controls