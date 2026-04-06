"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from pathlib import Path

import time

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from .config import init_config, get_config
from .routes import models, server
from .process_manager import get_process_manager
from .anthropic_adapter import (
    convert_anthropic_to_openai,
    convert_openai_to_anthropic,
    convert_stream_openai_to_anthropic,
    create_error_response,
)

# Global HTTP client for streaming
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Get or create the global HTTP client."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0))
    return _http_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _http_client
    # Startup
    init_config()
    yield
    # Shutdown - stop all llama-servers and close HTTP client
    pm = get_process_manager()
    await pm.unload_all()
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


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


# Anthropic Messages API compatibility endpoint
@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic Messages API compatibility endpoint.

    Converts Anthropic format to OpenAI format, forwards to llama-server,
    and converts the response back to Anthropic format.
    """
    import json

    pm = get_process_manager()

    # Parse Anthropic request
    try:
        anthropic_body = await request.json()
    except json.JSONDecodeError:
        return Response(
            content=json.dumps(create_error_response("invalid_request_error", "Invalid JSON body")),
            status_code=400,
            media_type="application/json"
        )

    # Get model
    model_name = anthropic_body.get("model")
    if not model_name:
        return Response(
            content=json.dumps(create_error_response("invalid_request_error", "Missing required field: model")),
            status_code=400,
            media_type="application/json"
        )

    rm = pm.get_model(model_name)
    if not rm or rm.process.returncode is not None:
        return Response(
            content=json.dumps(create_error_response("not_found_error", f"Model '{model_name}' not loaded. Load via /api/server/load")),
            status_code=503,
            media_type="application/json"
        )

    # Update last used time
    pm.touch_model(model_name)

    # Convert to OpenAI format
    openai_request = convert_anthropic_to_openai(anthropic_body)

    # Build target URL
    target_url = f"http://127.0.0.1:{rm.port}/v1/chat/completions"

    # Prepare headers (exclude host and content-length, we'll set new body)
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers["content-type"] = "application/json"

    # Determine streaming
    is_streaming = anthropic_body.get("stream", False)

    if is_streaming:
        client_timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
    else:
        client_timeout = httpx.Timeout(connect=30.0, read=3600.0, write=30.0, pool=30.0)

    client = get_http_client()
    try:
        proxy_request = client.build_request(
            method="POST",
            url=target_url,
            content=json.dumps(openai_request),
            headers=headers,
        )

        response = await client.send(proxy_request, stream=True)

        if is_streaming:
            async def stream_generator():
                async for chunk in convert_stream_openai_to_anthropic(
                    response.aiter_bytes(),
                    model_name,
                    anthropic_body.get("max_tokens", 4096)
                ):
                    yield chunk
                await response.aclose()

            return StreamingResponse(
                stream_generator(),
                status_code=200,
                media_type="text/event-stream"
            )
        else:
            content = await response.aread()
            await response.aclose()

            try:
                openai_response = json.loads(content)
                anthropic_response = convert_openai_to_anthropic(openai_response, model_name)
                return Response(
                    content=json.dumps(anthropic_response),
                    status_code=response.status_code,
                    media_type="application/json"
                )
            except json.JSONDecodeError:
                return Response(
                    content=content,
                    status_code=response.status_code,
                    media_type="application/json"
                )

    except httpx.ConnectError:
        return Response(
            content=json.dumps(create_error_response("api_error", f"Cannot connect to llama-server for model '{model_name}'")),
            status_code=503,
            media_type="application/json"
        )


# OpenAI-compatible GET /v1/models — list all configured models with
# context-length metadata so clients (e.g. Hermes, vLLM-style probes) can
# discover capabilities without a POST body.
@app.get("/v1/models")
async def list_v1_models() -> dict:
    config = get_config()
    now = int(time.time())
    data = []
    for mc in config.list_model_configs():
        ctx = None
        try:
            ctx = config.get_effective_args(mc.name).ctx_size
        except Exception:
            pass
        entry = {
            "id": mc.name,
            "object": "model",
            "created": now,
            "owned_by": "llama-api",
        }
        if ctx:
            # Expose under both common keys so vLLM- and LM-Studio-style
            # clients find it.
            entry["max_model_len"] = ctx
            entry["context_length"] = ctx
        data.append(entry)
    return {"object": "list", "data": data}


@app.get("/v1/models/{model_name}")
async def get_v1_model(model_name: str):
    config = get_config()
    mc = config.get_model_config(model_name)
    if not mc:
        return Response(
            content=f'{{"error": "Model \'{model_name}\' not found"}}',
            status_code=404,
            media_type="application/json",
        )
    ctx = None
    try:
        ctx = config.get_effective_args(mc.name).ctx_size
    except Exception:
        pass
    entry = {
        "id": mc.name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "llama-api",
    }
    if ctx:
        entry["max_model_len"] = ctx
        entry["context_length"] = ctx
    return entry


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

    # Prepare headers (exclude problematic headers)
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("transfer-encoding", None)
    headers.pop("connection", None)
    headers.pop("accept-encoding", None)

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

    client = get_http_client()
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