"""Anthropic Messages API compatibility adapter.

Converts between Anthropic Messages API and OpenAI Chat Completions API formats.
"""
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional


def convert_anthropic_to_openai(anthropic_request: dict) -> dict:
    """Convert Anthropic Messages request to OpenAI Chat Completions request."""
    openai_request = {}

    # Model
    openai_request["model"] = anthropic_request.get("model", "")

    # Max tokens
    if "max_tokens" in anthropic_request:
        openai_request["max_tokens"] = anthropic_request["max_tokens"]

    # Temperature
    if "temperature" in anthropic_request:
        openai_request["temperature"] = anthropic_request["temperature"]

    # Stream
    if "stream" in anthropic_request:
        openai_request["stream"] = anthropic_request["stream"]

    # Messages
    messages = []

    # System prompt (Anthropic has it at top level)
    if "system" in anthropic_request:
        system = anthropic_request["system"]
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # System as content blocks
            system_text = ""
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "")
            if system_text:
                messages.append({"role": "system", "content": system_text})

    # Convert messages
    for msg in anthropic_request.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # Content blocks
                user_content = []
                for block in content:
                    if block.get("type") == "text":
                        user_content.append({"type": "text", "text": block.get("text", "")})
                    elif block.get("type") == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            # Convert to OpenAI image format
                            media_type = source.get("media_type", "image/png")
                            data = source.get("data", "")
                            user_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{data}"
                                }
                            })
                if user_content:
                    messages.append({"role": "user", "content": user_content})

        elif role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                # Content blocks - extract text and tool_use
                text_parts = []
                tool_calls = []

                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "thinking":
                        # Include thinking as text (llama.cpp handles reasoning_content)
                        text_parts.append(block.get("thinking", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", str(uuid.uuid4())),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {}))
                            }
                        })

                assistant_msg = {"role": "assistant"}
                if text_parts:
                    assistant_msg["content"] = "\n".join(text_parts)
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)

    openai_request["messages"] = messages

    # Tools
    if "tools" in anthropic_request:
        openai_tools = []
        for tool in anthropic_request["tools"]:
            if tool.get("type") == "function" or "name" in tool:
                # Already OpenAI format or Anthropic tool
                schema = tool.get("input_schema", tool.get("function", {}).get("parameters", {}))
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": schema
                    }
                })
        if openai_tools:
            openai_request["tools"] = openai_tools

    # Tool choice
    if "tool_choice" in anthropic_request:
        choice = anthropic_request["tool_choice"]
        if isinstance(choice, dict):
            if choice.get("type") == "tool":
                openai_request["tool_choice"] = {
                    "type": "function",
                    "function": {"name": choice.get("name", "")}
                }
            else:
                openai_request["tool_choice"] = choice.get("type", "auto")
        else:
            openai_request["tool_choice"] = choice

    return openai_request


def convert_openai_to_anthropic(openai_response: dict, model: str) -> dict:
    """Convert OpenAI Chat Completions response to Anthropic Messages response."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Extract content
    content = []
    choices = openai_response.get("choices", [])

    if choices:
        choice = choices[0]
        msg = choice.get("message", {})

        # Text content
        text = msg.get("content", "")
        if text:
            content.append({"type": "text", "text": text})

        # Reasoning content (from thinking models)
        reasoning = msg.get("reasoning_content", "")
        if reasoning:
            content.insert(0, {"type": "thinking", "thinking": reasoning})

        # Tool calls
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": func.get("name", ""),
                "input": args
            })

    # Stop reason
    finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(finish_reason, "end_turn")

    # Usage
    usage = openai_response.get("usage", {})

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
    }


class StreamState:
    """Track state for streaming conversion."""
    def __init__(self):
        self.content_index = 0
        self.current_block_type: Optional[str] = None
        self.current_block_index: Optional[int] = None
        self.tool_calls: Dict[int, dict] = {}  # index -> {id, name, arguments}
        self.next_tool_index = 0


async def convert_stream_openai_to_anthropic(
    openai_stream: AsyncGenerator[bytes, None],
    model: str,
    request_max_tokens: int = 4096
) -> AsyncGenerator[bytes, None]:
    """Convert OpenAI streaming response to Anthropic streaming format."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    state = StreamState()

    def make_event(event_type: str, data: dict) -> bytes:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()

    # Message start event
    message_start = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "created_at": int(time.time()),
        }
    }
    yield make_event("message_start", message_start)

    buffer = ""

    try:
        async for chunk in openai_stream:
            chunk_str = chunk.decode("utf-8", errors="replace")
            # print(f"[anthropic_adapter] Received chunk: {chunk_str[:100]!r}...")
            buffer += chunk_str

            # Process complete lines (SSE uses double newlines)
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line or not line.startswith("data: "):
                    continue

                data = line[6:]  # Remove "data: "
                if data == "[DONE]":
                    # Close any open block
                    if state.current_block_type:
                        yield make_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": state.current_block_index
                        })
                        state.current_block_type = None

                    # Message stop
                    yield make_event("message_stop", {"type": "message_stop"})
                    continue

                try:
                    openai_chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = openai_chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # Process reasoning content (thinking)
                reasoning = delta.get("reasoning_content", "")
                if reasoning:
                    if state.current_block_type != "thinking":
                        # Close previous block if any
                        if state.current_block_type:
                            yield make_event("content_block_stop", {
                                "type": "content_block_stop",
                                "index": state.current_block_index
                            })
                        # Start new thinking block
                        state.current_block_type = "thinking"
                        state.current_block_index = state.content_index
                        yield make_event("content_block_start", {
                            "type": "content_block_start",
                            "index": state.content_index,
                            "content_block": {"type": "thinking", "thinking": ""}
                        })
                        state.content_index += 1

                    yield make_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": state.current_block_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning}
                    })

                # Process text content
                text = delta.get("content", "")
                if text:
                    if state.current_block_type != "text":
                        # Close previous block if any
                        if state.current_block_type:
                            yield make_event("content_block_stop", {
                                "type": "content_block_stop",
                                "index": state.current_block_index
                            })
                        # Start new text block
                        state.current_block_type = "text"
                        state.current_block_index = state.content_index
                        yield make_event("content_block_start", {
                            "type": "content_block_start",
                            "index": state.content_index,
                            "content_block": {"type": "text", "text": ""}
                        })
                        state.content_index += 1

                    yield make_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": state.current_block_index,
                        "delta": {"type": "text_delta", "text": text}
                    })

                # Process tool calls
                tool_calls = delta.get("tool_calls", [])
                for tc in tool_calls:
                    tc_id = tc.get("id")
                    tc_func = tc.get("function", {})

                    if tc_id:
                        # New tool call - close previous block if any
                        if state.current_block_type:
                            yield make_event("content_block_stop", {
                                "type": "content_block_stop",
                                "index": state.current_block_index
                            })
                            state.current_block_type = None

                        # Start new tool_use block
                        tool_index = state.next_tool_index
                        state.tool_calls[tool_index] = {
                            "id": tc_id,
                            "name": tc_func.get("name", ""),
                            "arguments": ""
                        }
                        state.next_tool_index += 1

                        state.current_block_type = "tool_use"
                        state.current_block_index = state.content_index

                        yield make_event("content_block_start", {
                            "type": "content_block_start",
                            "index": state.content_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": tc_func.get("name", ""),
                                "input": {}
                            }
                        })
                        state.content_index += 1

                        # Send initial arguments if any
                        if tc_func.get("arguments"):
                            state.tool_calls[tool_index]["arguments"] += tc_func.get("arguments", "")
                            yield make_event("content_block_delta", {
                                "type": "content_block_delta",
                                "index": state.current_block_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": tc_func.get("arguments", "")
                                }
                            })

                    elif tc_func.get("arguments") and state.current_block_type == "tool_use":
                        # Continuation of tool call arguments
                        yield make_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": state.current_block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": tc_func.get("arguments", "")
                            }
                        })

                # Handle finish reason
                if finish_reason:
                    # Close any open block
                    if state.current_block_type:
                        yield make_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": state.current_block_index
                        })
                        state.current_block_type = None

                    stop_reason = {
                        "stop": "end_turn",
                        "length": "max_tokens",
                        "tool_calls": "tool_use",
                    }.get(finish_reason, "end_turn")

                    yield make_event("message_delta", {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason},
                        "usage": {"output_tokens": 0}
                    })

                    yield make_event("message_stop", {"type": "message_stop"})

    except Exception as e:
        # Close any open block
        if state.current_block_type:
            yield make_event("content_block_stop", {
                "type": "content_block_stop",
                "index": state.current_block_index
            })

        error_msg = str(e) if str(e) else type(e).__name__
        error_event = {
            "type": "error",
            "error": {"type": "internal_error", "message": error_msg}
        }
        yield make_event("error", error_event)


def create_error_response(error_type: str, message: str, status_code: int = 400) -> dict:
    """Create an Anthropic-style error response."""
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message
        }
    }