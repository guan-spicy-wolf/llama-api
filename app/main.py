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
    # Shutdown - stop all llama-servers
    pm = get_process_manager()
    await pm.unload_all()


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
    """Proxy /v1/* requests to appropriate llama-server based on model."""
    import json

    pm = get_process_manager()

    # Get request body to extract model
    body = await request.body()
    model_name = None
    body_json = None

    if body:
        try:
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
    if not is_streaming and body_json:
        if body_json.get("stream") is True:
            is_streaming = True

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