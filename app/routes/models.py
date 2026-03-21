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