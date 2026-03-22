# Multi-Model Support Design

## Overview

Refactor llama-api to support multiple models running concurrently, with automatic idle unloading and memory protection.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         llama-api (port 8002)        │
                    │    ┌─────────────────────────┐       │
   Request ────────►│    │   ProcessManager        │       │
   {model: "qwen"}  │    │   processes: {          │       │
                    │    │     "qwen": ProcessInfo │       │
                    │    │     "minimax": ...      │       │
                    │    │   }                     │       │
                    │    └──────────┬──────────────┘       │
                    └───────────────┼─────────────────────┘
                                    │ route by model name
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              llama-server    llama-server    llama-server
              (port 18080)    (port 18081)    (port 18082)
              model: qwen     model: minimax  model: ...
```

## Core Components

### 1. ProcessInfo Dataclass

```python
@dataclass
class ProcessInfo:
    process: asyncio.subprocess.Process
    port: int
    model: str
    started_at: float
    last_used_at: float
    idle_timeout: int  # seconds, from config or global default
```

### 2. ProcessManager Refactor

- `_processes: Dict[str, ProcessInfo]` - model_name -> ProcessInfo
- `start(model_name)` - Load model, check memory first
- `unload(model_name)` - Unload specific model
- `unload_all()` - Unload all models
- `_idle_checker()` - Background task, check every 60s
- `_save_routing()` / `_load_routing()` - Persist to routing.json

### 3. Routing Logic (main.py)

```python
# proxy_v1()
model = request_body.get("model")
if model not in pm._processes:
    # Auto-load? Or return 503?
    return 503
target_url = f"http://127.0.0.1:{pm._processes[model].port}/v1/{path}"
pm._processes[model].last_used_at = time.time()
```

### 4. Configuration

**Global config (config.yaml):**
```yaml
default_idle_timeout: 300  # 5 minutes
```

**Model config (models.d/*.yaml):**
```yaml
name: qwen3-claude
file: Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.i1-Q4_K_M.gguf
idle_timeout: 600  # Optional, overrides global default
args:
  ctx_size: 65536
  n_gpu_layers: 99
```

### 5. Memory Protection

Before loading a model:
1. Estimate required memory from model file size
2. Get available system memory
3. If `required > available * 0.8`, reject with error
4. This prevents crashing existing models

### 6. Data Persistence

**Location:** `/var/containers/llama-api/routing.json`

```json
{
  "qwen3-claude": {
    "port": 18080,
    "pid": 12345,
    "started_at": 1774187531.65,
    "last_used_at": 1774187800.12
  }
}
```

- Saved after each load/unload
- Loaded on startup to restore state (optional: verify processes still alive)

## API Changes

| Endpoint | Current | New |
|----------|---------|-----|
| `GET /api/server/status` | Single model status | `{models: [{name, status, pid, port, idle_remaining}]}` |
| `POST /api/server/load` | Replace model | Load model, keep others running |
| `POST /api/server/unload` | Unload only model | `{"model": "name"}` - unload specific |
| `GET /v1/models` | Configured models | Loaded models with status |

### New Endpoints

- `GET /api/server/status/{model}` - Single model status
- `POST /api/server/unload/all` - Unload all models

## WebUI Adaptations

| Feature | Change |
|---------|--------|
| Status display | Single card → Multiple model cards |
| Load model | Button per model config, not just dropdown |
| Unload model | Button on each loaded model card |
| Idle timeout | Show countdown timer on each card |
| Memory indicator | Total usage + per-model breakdown |

## Implementation Order

1. Refactor ProcessManager for multi-model support
2. Update API routes (status, load, unload)
3. Add routing.json persistence
4. Add idle checker background task
5. Add memory protection
6. Update proxy routing logic
7. Update WebUI

## Configuration Changes

- Add `default_idle_timeout` to config
- Add optional `idle_timeout` field to ModelConfig
- Add `data_dir` config pointing to `/var/containers/llama-api`