"""Server control API routes."""
import time

import psutil
from fastapi import APIRouter, HTTPException

from ..models import LoadRequest, LoadResponse, UnloadRequest
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