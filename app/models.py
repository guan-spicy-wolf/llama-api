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