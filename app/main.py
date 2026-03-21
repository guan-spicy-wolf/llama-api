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