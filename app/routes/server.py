"""Server control API routes."""
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