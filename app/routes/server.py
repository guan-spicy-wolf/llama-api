"""Server control API routes."""
import os

import psutil
from fastapi import APIRouter, HTTPException

from ..models import LoadRequest, LoadResponse
from ..process_manager import get_process_manager
from ..config import get_config

router = APIRouter(prefix="/api/server", tags=["server"])


@router.get("/status")
async def get_status() -> dict:
    """Get current server status."""
    pm = get_process_manager()
    return pm.status.model_dump()


@router.get("/metrics")
async def get_metrics() -> dict:
    """Get system metrics (CPU, memory, GPU)."""
    # CPU
    cpu_percent = psutil.cpu_percent(interval=0.5)

    # Memory
    mem = psutil.virtual_memory()
    mem_total_gb = round(mem.total / (1024 ** 3), 1)
    mem_used_gb = round(mem.used / (1024 ** 3), 1)
    mem_percent = mem.percent

    # Process memory (llama-server)
    pm = get_process_manager()
    process_mem_gb = 0.0
    if pm._process and pm._process.pid:
        try:
            proc = psutil.Process(pm._process.pid)
            process_mem_gb = round(proc.memory_info().rss / (1024 ** 3), 1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # GPU memory (ROCm)
    gpu_mem_percent = None
    gpu_mem_used_gb = None
    gpu_mem_total_gb = 96.0  # Default for Radeon 8060S
    try:
        import subprocess
        result = subprocess.run(
            ["rocm-smi"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                # Look for data line starting with device index
                if line.strip() and line[0].isdigit() and "Device" not in line:
                    parts = line.split()
                    # VRAM% is at index 14, GPU% at index 15
                    if len(parts) >= 15:
                        vram_str = parts[14].rstrip('%')
                        if vram_str.replace('.', '').isdigit():
                            gpu_mem_percent = float(vram_str)
                            gpu_mem_used_gb = round(gpu_mem_total_gb * gpu_mem_percent / 100, 1)
                        break
    except Exception:
        pass

    return {
        "cpu": {
            "percent": cpu_percent,
        },
        "memory": {
            "total_gb": mem_total_gb,
            "used_gb": mem_used_gb,
            "percent": mem_percent,
            "process_gb": process_mem_gb,
        },
        "gpu": {
            "total_gb": gpu_mem_total_gb,
            "used_gb": gpu_mem_used_gb,
            "percent": gpu_mem_percent,
        } if gpu_mem_percent is not None else None,
    }


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