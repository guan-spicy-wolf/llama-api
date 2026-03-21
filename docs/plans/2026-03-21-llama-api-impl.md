# llama-api 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个 FastAPI 服务，通过 Web UI 和 API 管理 llama-server 进程，支持加载/卸载不同模型。

**Architecture:** FastAPI 后端提供 REST API，静态文件服务 WebUI，使用 asyncio subprocess 管理 llama-server 进程。YAML 配置文件存储模型参数，支持智能推断默认值。

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, PyYAML, Pydantic, aiofiles

---

## Task 1: 项目初始化

**Files:**
- Create: `~/llama-api/requirements.txt`
- Create: `~/llama-api/app/__init__.py`

**Step 1: 创建 requirements.txt**

```txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pyyaml>=6.0
pydantic>=2.5.0
aiofiles>=23.2.0
```

**Step 2: 创建 app 包目录结构**

```bash
mkdir -p ~/llama-api/app/routes
touch ~/llama-api/app/__init__.py
touch ~/llama-api/app/routes/__init__.py
```

**Step 3: 创建虚拟环境并安装依赖**

```bash
cd ~/llama-api && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Step 4: Commit**

```bash
cd ~/llama-api && git init && git add . && git commit -m "chore: initialize project structure"
```

---

## Task 2: Pydantic 数据模型

**Files:**
- Create: `~/llama-api/app/models.py`

**Step 1: 编写 Pydantic 模型**

```python
"""Pydantic models for llama-api."""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ServerStatus(str, Enum):
    """Server process status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class ModelArgs(BaseModel):
    """Arguments for llama-server."""
    ctx_size: Optional[int] = Field(default=None, description="Context size")
    n_gpu_layers: Optional[int] = Field(default=None, description="GPU layers")
    threads: Optional[int] = Field(default=None, description="Number of threads")
    batch_size: Optional[int] = Field(default=None, description="Batch size")

    model_config = {"extra": "allow"}


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    name: str = Field(..., description="Model name identifier")
    file: str = Field(..., description="GGUF file name")
    args: Optional[ModelArgs] = Field(default=None, description="Model-specific arguments")

    model_config = {"extra": "allow"}


class ServerInfo(BaseModel):
    """Current server status and info."""
    status: ServerStatus
    model: Optional[str] = None
    pid: Optional[int] = None
    port: Optional[int] = None
    started_at: Optional[float] = None
    error: Optional[str] = None


class LoadRequest(BaseModel):
    """Request to load a model."""
    model: str = Field(..., description="Model name to load")


class LoadResponse(BaseModel):
    """Response to load request."""
    success: bool
    message: str
    pid: Optional[int] = None
```

**Step 2: 验证模块可导入**

```bash
cd ~/llama-api && source .venv/bin/activate && python -c "from app.models import ModelConfig; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
cd ~/llama-api && git add app/models.py && git commit -m "feat: add pydantic data models"
```

---

## Task 3: 配置管理

**Files:**
- Create: `~/llama-api/app/config.py`

**Step 1: 编写配置管理模块**

```python
"""Configuration management for llama-api."""
import os
import yaml
import math
from pathlib import Path
from typing import Optional

from .models import ModelConfig, ModelArgs


# Default paths
DEFAULT_CONFIG_DIR = Path("/var/containers/llama-api")
DEFAULT_MODELS_DIR = Path("/var/models")


class Config:
    """Application configuration."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
        self.models_dir = DEFAULT_MODELS_DIR
        self.llama_server = Path.home() / "llama.cpp/rocm/bin/llama-server"
        self.default_port = 8080
        self._load_main_config()
        self._model_configs: dict[str, ModelConfig] = {}
        self._load_model_configs()

    def _load_main_config(self) -> None:
        """Load main configuration file."""
        config_file = self.config_dir / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                data = yaml.safe_load(f) or {}
            if "models_dir" in data:
                self.models_dir = Path(data["models_dir"])
            if "llama_server" in data:
                self.llama_server = Path(data["llama_server"])
            if "default_port" in data:
                self.default_port = data["default_port"]

    def _load_model_configs(self) -> None:
        """Load all model configuration files from models.d/."""
        models_d = self.config_dir / "models.d"
        if not models_d.exists():
            return

        for yaml_file in models_d.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if data:
                    config = ModelConfig(**data)
                    self._model_configs[config.name] = config
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")

    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self._model_configs.get(name)

    def list_model_configs(self) -> list[ModelConfig]:
        """List all model configurations."""
        return list(self._model_configs.values())

    def save_model_config(self, config: ModelConfig) -> None:
        """Save model configuration to file."""
        models_d = self.config_dir / "models.d"
        models_d.mkdir(parents=True, exist_ok=True)

        config_file = models_d / f"{config.name}.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config.model_dump(mode="python"), f)

        self._model_configs[config.name] = config

    def delete_model_config(self, name: str) -> bool:
        """Delete model configuration."""
        if name not in self._model_configs:
            return False

        config_file = self.config_dir / "models.d" / f"{name}.yaml"
        if config_file.exists():
            config_file.unlink()

        del self._model_configs[name]
        return True

    def infer_args(self, model_file: str) -> ModelArgs:
        """Infer model arguments from file size."""
        model_path = self.models_dir / model_file
        if not model_path.exists():
            return ModelArgs()

        # Get file size in bytes
        size_bytes = model_path.stat().st_size
        size_gb = size_bytes / (1024 ** 3)

        # Infer parameters based on size (rough estimates for GGUF Q4)
        # Q4 models: ~0.3GB per billion parameters
        estimated_params_b = size_gb / 0.3

        # Determine context size
        if estimated_params_b < 10:
            ctx_size = 8192
        elif estimated_params_b < 50:
            ctx_size = 4096
        else:
            ctx_size = 2048

        # Get CPU threads
        cpu_count = os.cpu_count() or 4
        threads = max(1, cpu_count // 2)

        return ModelArgs(
            ctx_size=ctx_size,
            n_gpu_layers=99,  # Full GPU offload for ROCm
            threads=threads,
        )

    def get_effective_args(self, model_name: str) -> ModelArgs:
        """Get effective args: configured args + inferred defaults."""
        config = self.get_model_config(model_name)
        if not config:
            return ModelArgs()

        # Start with inferred args
        inferred = self.infer_args(config.file)

        # Override with configured args
        configured = config.args or ModelArgs()
        effective = inferred.model_dump()

        for key, value in configured.model_dump(exclude_unset=True).items():
            if value is not None:
                effective[key] = value

        return ModelArgs(**effective)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def init_config(config_dir: Optional[Path] = None) -> Config:
    """Initialize global config instance."""
    global _config
    _config = Config(config_dir)
    return _config
```

**Step 2: 验证模块可导入**

```bash
cd ~/llama-api && source .venv/bin/activate && python -c "from app.config import Config; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
cd ~/llama-api && git add app/config.py && git commit -m "feat: add configuration management"
```

---

## Task 4: 进程管理器

**Files:**
- Create: `~/llama-api/app/process_manager.py`

**Step 1: 编写进程管理器**

```python
"""Process manager for llama-server."""
import asyncio
import time
from pathlib import Path
from typing import Optional

from .models import ServerStatus, ServerInfo
from .config import get_config


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
        self._port = port or config.default_port
        self._model = model_name
        self._error = None

        # Build command
        cmd = [
            str(config.llama_server),
            "-m", str(model_path),
            "--port", str(self._port),
            "--host", "0.0.0.0",
        ]

        # Add model args
        args_dict = args.model_dump(exclude_unset=True, exclude_none=True)
        for key, value in args_dict.items():
            arg_name = key.replace("_", "-")
            cmd.extend([f"--{arg_name}", str(value)])

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
```

**Step 2: 验证模块可导入**

```bash
cd ~/llama-api && source .venv/bin/activate && python -c "from app.process_manager import ProcessManager; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
cd ~/llama-api && git add app/process_manager.py && git commit -m "feat: add llama-server process manager"
```

---

## Task 5: API 路由 - 模型管理

**Files:**
- Create: `~/llama-api/app/routes/models.py`

**Step 1: 编写模型路由**

```python
"""Model management API routes."""
from fastapi import APIRouter, HTTPException

from ..models import ModelConfig
from ..config import get_config

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
async def list_models() -> list[dict]:
    """List all configured models."""
    config = get_config()
    models = config.list_model_configs()
    return [m.model_dump() for m in models]


@router.get("/{name}")
async def get_model(name: str) -> dict:
    """Get a specific model configuration."""
    config = get_config()
    model = config.get_model_config(name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return model.model_dump()


@router.put("/{name}")
async def create_or_update_model(name: str, config: ModelConfig) -> dict:
    """Create or update a model configuration."""
    if config.name != name:
        raise HTTPException(status_code=400, detail="Name in path must match name in body")

    app_config = get_config()
    app_config.save_model_config(config)
    return {"message": f"Model '{name}' saved", "model": config.model_dump()}


@router.delete("/{name}")
async def delete_model(name: str) -> dict:
    """Delete a model configuration."""
    config = get_config()
    if not config.delete_model_config(name):
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return {"message": f"Model '{name}' deleted"}
```

**Step 2: 验证模块可导入**

```bash
cd ~/llama-api && source .venv/bin/activate && python -c "from app.routes.models import router; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
cd ~/llama-api && git add app/routes/models.py && git commit -m "feat: add model management API routes"
```

---

## Task 6: API 路由 - 服务器控制

**Files:**
- Create: `~/llama-api/app/routes/server.py`

**Step 1: 编写服务器路由**

```python
"""Server control API routes."""
from fastapi import APIRouter, HTTPException

from ..models import LoadRequest, LoadResponse, ServerInfo
from ..process_manager import get_process_manager
from ..config import get_config

router = APIRouter(prefix="/api/server", tags=["server"])


@router.get("/status")
async def get_status() -> dict:
    """Get current server status."""
    pm = get_process_manager()
    return pm.status.model_dump()


@router.post("/load", response_model=LoadResponse)
async def load_model(request: LoadRequest) -> LoadResponse:
    """Load a model and start llama-server."""
    pm = get_process_manager()
    config = get_config()

    # Check model exists
    if not config.get_model_config(request.model):
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")

    try:
        info = await pm.start(request.model)
        return LoadResponse(
            success=True,
            message=f"Model '{request.model}' loaded successfully",
            pid=info.pid,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload")
async def unload_model() -> dict:
    """Unload current model and stop llama-server."""
    pm = get_process_manager()
    await pm.stop()
    return {"message": "Model unloaded", "status": "stopped"}
```

**Step 2: 验证模块可导入**

```bash
cd ~/llama-api && source .venv/bin/activate && python -c "from app.routes.server import router; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
cd ~/llama-api && git add app/routes/server.py && git commit -m "feat: add server control API routes"
```

---

## Task 7: FastAPI 主入口

**Files:**
- Create: `~/llama-api/app/main.py`

**Step 1: 编写 FastAPI 主入口**

```python
"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import init_config
from .routes import models, server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    init_config()
    yield
    # Shutdown - stop llama-server if running
    from .process_manager import get_process_manager
    pm = get_process_manager()
    if pm._process and pm._process.returncode is None:
        await pm.stop()


app = FastAPI(
    title="llama-api",
    description="API for managing llama.cpp server instances",
    version="0.1.0",
    lifespan=lifespan,
)

# Include API routes
app.include_router(models.router)
app.include_router(server.router)

# Serve static files
WEBUI_DIR = Path(__file__).parent.parent / "webui"
if WEBUI_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEBUI_DIR), name="static")


@app.get("/")
async def index():
    """Serve the WebUI index page."""
    index_path = WEBUI_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "llama-api is running", "docs": "/docs"}
```

**Step 2: 验证应用可启动**

```bash
cd ~/llama-api && source .venv/bin/activate && python -c "from app.main import app; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
cd ~/llama-api && git add app/main.py && git commit -m "feat: add FastAPI application entry point"
```

---

## Task 8: WebUI

**Files:**
- Create: `~/llama-api/webui/index.html`

**Step 1: 编写 WebUI**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>llama-api Control Panel</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 2rem;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { margin-bottom: 1rem; color: #4ecca3; }
        h2 { margin: 1.5rem 0 1rem; color: #4ecca3; font-size: 1.2rem; }
        .card {
            background: #16213e;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        .status-running { background: #059669; }
        .status-stopped { background: #6b7280; }
        .status-error { background: #dc2626; }
        .model-list { list-style: none; }
        .model-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            margin: 0.5rem 0;
            background: #1a1a2e;
            border-radius: 6px;
        }
        .model-item:hover { background: #0f3460; }
        button {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.875rem;
            transition: all 0.2s;
        }
        .btn-primary { background: #4ecca3; color: #1a1a2e; }
        .btn-primary:hover { background: #3db892; }
        .btn-danger { background: #dc2626; color: white; }
        .btn-danger:hover { background: #b91c1c; }
        .btn-secondary { background: #4b5563; color: white; }
        .btn-secondary:hover { background: #374151; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .info { color: #9ca3af; font-size: 0.875rem; margin-top: 0.5rem; }
        .error { color: #f87171; margin-top: 0.5rem; }
        .refresh-btn { float: right; }
    </style>
</head>
<body>
    <div class="container">
        <h1>llama-api Control Panel</h1>

        <div class="card">
            <h2>Server Status</h2>
            <div id="status-content">
                <span class="status-badge status-stopped">Loading...</span>
            </div>
            <div id="status-info" class="info"></div>
            <div style="margin-top: 1rem;">
                <button id="unload-btn" class="btn btn-danger" onclick="unloadModel()" disabled>Unload Model</button>
            </div>
        </div>

        <div class="card">
            <h2>Available Models
                <button class="btn btn-secondary refresh-btn" onclick="loadModels()">Refresh</button>
            </h2>
            <ul id="model-list" class="model-list">
                <li class="info">Loading models...</li>
            </ul>
        </div>
    </div>

    <script>
        let currentStatus = null;

        async function loadStatus() {
            try {
                const resp = await fetch('/api/server/status');
                currentStatus = await resp.json();
                updateStatusUI();
            } catch (e) {
                document.getElementById('status-content').innerHTML =
                    '<span class="status-badge status-error">Error</span>';
                document.getElementById('status-info').textContent = e.message;
            }
        }

        function updateStatusUI() {
            const status = currentStatus;
            const statusEl = document.getElementById('status-content');
            const infoEl = document.getElementById('status-info');
            const unloadBtn = document.getElementById('unload-btn');

            let badgeClass = 'status-stopped';
            if (status.status === 'running') badgeClass = 'status-running';
            else if (status.status === 'error') badgeClass = 'status-error';

            statusEl.innerHTML = `<span class="status-badge ${badgeClass}">${status.status}</span>`;

            if (status.status === 'running') {
                infoEl.textContent = `Model: ${status.model} | PID: ${status.pid} | Port: ${status.port}`;
                unloadBtn.disabled = false;
            } else if (status.error) {
                infoEl.textContent = status.error;
                unloadBtn.disabled = true;
            } else {
                infoEl.textContent = 'No model loaded';
                unloadBtn.disabled = true;
            }
        }

        async function loadModels() {
            try {
                const resp = await fetch('/api/models');
                const models = await resp.json();
                const listEl = document.getElementById('model-list');

                if (models.length === 0) {
                    listEl.innerHTML = '<li class="info">No models configured. Add config files to /var/containers/llama-api/models.d/</li>';
                    return;
                }

                listEl.innerHTML = models.map(m => `
                    <li class="model-item">
                        <div>
                            <strong>${m.name}</strong>
                            <div class="info">${m.file}</div>
                        </div>
                        <button class="btn btn-primary" onclick="loadModel('${m.name}')">Load</button>
                    </li>
                `).join('');
            } catch (e) {
                document.getElementById('model-list').innerHTML =
                    `<li class="error">Failed to load models: ${e.message}</li>`;
            }
        }

        async function loadModel(name) {
            try {
                const resp = await fetch('/api/server/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: name })
                });

                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || 'Failed to load model');
                }

                await loadStatus();
                alert(`Model '${name}' loaded successfully`);
            } catch (e) {
                alert(`Error: ${e.message}`);
            }
        }

        async function unloadModel() {
            try {
                const resp = await fetch('/api/server/unload', { method: 'POST' });
                if (!resp.ok) throw new Error('Failed to unload');
                await loadStatus();
            } catch (e) {
                alert(`Error: ${e.message}`);
            }
        }

        // Initial load
        loadStatus();
        loadModels();

        // Auto-refresh status
        setInterval(loadStatus, 5000);
    </script>
</body>
</html>
```

**Step 2: Commit**

```bash
cd ~/llama-api && git add webui/index.html && git commit -m "feat: add WebUI control panel"
```

---

## Task 9: 示例配置文件

**Files:**
- Create: `~/llama-api/config.example.yaml`

**Step 1: 创建示例配置**

```yaml
# llama-api 配置文件示例
# 复制到 /var/containers/llama-api/config.yaml

# llama-server 可执行文件路径
llama_server: /home/holo/llama.cpp/rocm/bin/llama-server

# 模型文件目录
models_dir: /var/models

# 默认端口
default_port: 8080
```

**Step 2: 创建示例模型配置**

```yaml
# 示例模型配置
# 放置在 /var/containers/llama-api/models.d/minimax.yaml

name: minimax
file: MiniMax-139B-Merged.gguf
args:
  ctx_size: 4096
  n_gpu_layers: 99
  threads: 8
```

保存为 `~/llama-api/models.d.example/minimax.yaml`

**Step 3: Commit**

```bash
cd ~/llama-api && mkdir -p models.d.example && git add . && git commit -m "docs: add example configuration files"
```

---

## Task 10: 初始化运行目录并测试

**Step 1: 创建运行目录**

```bash
sudo mkdir -p /var/containers/llama-api/models.d
sudo chown -R holo:holo /var/containers/llama-api
```

**Step 2: 创建配置文件**

```bash
cp ~/llama-api/config.example.yaml /var/containers/llama-api/config.yaml
```

**Step 3: 创建示例模型配置**

```bash
cat > /var/containers/llama-api/models.d/z-image-turbo.yaml << 'EOF'
name: z-image-turbo
file: z-image-turbo-Q4_K_M.gguf
args:
  ctx_size: 8192
  n_gpu_layers: 99
  threads: 8
EOF
```

**Step 4: 启动服务**

```bash
cd ~/llama-api && source .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Step 5: 验证 API**

```bash
# 在另一个终端
curl http://localhost:8000/api/models
curl http://localhost:8000/api/server/status
```

**Step 6: 访问 WebUI**

打开浏览器访问 `http://localhost:8000/`

**Step 7: Final commit**

```bash
cd ~/llama-api && git add . && git commit -m "chore: add deployment instructions"
```

---

## 完成检查清单

- [ ] API 可启动：`uvicorn app.main:app`
- [ ] `/api/models` 返回模型列表
- [ ] `/api/server/status` 返回状态
- [ ] `/api/server/load` 可加载模型
- [ ] `/api/server/unload` 可卸载模型
- [ ] WebUI 可访问并正常工作
- [ ] 配置文件正确读取