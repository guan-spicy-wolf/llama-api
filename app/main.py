"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from .config import init_config
from .routes import models, server
from .process_manager import get_process_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    init_config()
    yield
    # Shutdown - stop llama-server if running
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


# Reverse proxy for /v1/* to llama-server
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy_v1(request: Request, path: str):
    """Proxy /v1/* requests to llama-server."""
    pm = get_process_manager()

    if not pm._port or pm._process is None or pm._process.returncode is not None:
        return Response(
            content='{"error": "No model loaded. Load a model first via /api/server/load"}',
            status_code=503,
            media_type="application/json"
        )

    # Build target URL
    target_url = f"http://127.0.0.1:{pm._port}/v1/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    # Get request body
    body = await request.body()

    # Prepare headers (exclude host)
    headers = dict(request.headers)
    headers.pop("host", None)

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Check if streaming is requested
            accept = request.headers.get("accept", "")
            is_streaming = "text/event-stream" in accept

            proxy_request = client.build_request(
                method=request.method,
                url=target_url,
                content=body,
                headers=headers,
            )

            response = await client.send(proxy_request, stream=is_streaming)

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
                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
        except httpx.ConnectError:
            return Response(
                content='{"error": "Cannot connect to llama-server"}',
                status_code=503,
                media_type="application/json"
            )