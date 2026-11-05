"""Opt-in, export-only GenAI content. Never persist full prompts in request_traces."""

from __future__ import annotations

import json
from typing import Any


def enabled() -> bool:
    """Use the existing OTel content-capture policy, not the preview policy."""
    from docsgpt.tracing.otel import content_capture_enabled

    return content_capture_enabled()


def safe_value(value: Any) -> Any:
    """Snapshot provider data without embedding binary/file payloads or opaque reprs."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"[binary, {len(value)} bytes]"
    if isinstance(value, str):
        return "[inline file omitted]" if value.startswith("data:") else value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [safe_value(v) for v in value]
    if isinstance(value, dict):
        return {
            str(k): "[file content omitted]"
            if k in {"data", "file_data", "image_url", "input_audio"}
            else safe_value(v)
            for k, v in value.items()
        }
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return safe_value(dump())
    return (
        {k: safe_value(v) for k, v in vars(value).items() if not k.startswith("_")}
        if hasattr(value, "__dict__")
        else type(value).__name__
    )


def text_message(value: Any, role: str = "assistant") -> list[dict]:
    """Encode a text or structured deliverable in the GenAI message format."""
    value = safe_value(value)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return [{"role": role, "parts": [{"type": "text", "content": text}]}]


def messages(value: Any, role: str = "assistant", *, stream: bool = False) -> list[dict]:
    """Normalize OpenAI-compatible messages/choices and streamed tool deltas."""
    values = value if isinstance(value, (list, tuple)) else [value]
    result = []
    text = []
    calls: dict[Any, dict] = {}
    for item in values:
        if not stream:
            calls = {}
        if isinstance(item, str):
            text.append(item)
            continue
        item = safe_value(item)
        if not isinstance(item, dict):
            result.extend(text_message(item, role))
            continue
        item = item.get("message") or item.get("delta") or item
        item_role = item.get("role") or role
        if item_role in {"system", "developer"}:
            continue
        parts = []
        content = item.get("content")
        if isinstance(content, str):
            parts.append({"type": "text", "content": content})
        elif content:
            parts.append({"type": "text", "content": json.dumps(safe_value(content), ensure_ascii=False)})
        if item_role == "tool":
            parts = [{"type": "tool_call_response", "id": item.get("tool_call_id"), "response": content}]
        for call in item.get("tool_calls") or []:
            function = call.get("function") or call
            key = call.get("index", call.get("id", len(calls)))
            current = calls.setdefault(
                key, {"type": "tool_call", "id": call.get("id"), "name": function.get("name"), "arguments": ""}
            )
            current["id"] = call.get("id") or current["id"]
            current["name"] = function.get("name") or current["name"]
            args = function.get("arguments")
            if isinstance(args, str):
                current["arguments"] += args
            elif args is not None:
                current["arguments"] = args
        if calls and not stream:
            parts.extend(_decoded_calls(calls))
        if parts:
            result.append({"role": item_role, "parts": parts})
    if text:
        result.extend(text_message("".join(text), role))
    if calls and stream:
        result.append({"role": "assistant", "parts": _decoded_calls(calls)})
    return result


def _decoded_calls(calls: dict) -> list[dict]:
    """Decode completed JSON arguments, retaining malformed provider output verbatim."""
    for call in calls.values():
        try:
            call["arguments"] = json.loads(call["arguments"])
        except (ValueError, TypeError):
            pass
    return list(calls.values())


def capture_response(span: Any, response: Any, *, stream: bool = False) -> None:
    """Normalize only when capturing; telemetry must not evaluate opaque outputs otherwise."""
    if span and enabled():
        try:
            capture(span, **{"gen_ai.output.messages": messages(response, stream=stream)})
        except (TypeError, ValueError, AttributeError, RecursionError):
            return


def capture(span: Any, **attributes: Any) -> None:
    """Keep JSON snapshots only for an opted-in, unblocked active trace."""
    if not span or not enabled() or span.trace.content_blocked:
        return
    try:
        for key, value in attributes.items():
            span.otel_content[key] = json.dumps(safe_value(value), ensure_ascii=False)
    except (TypeError, ValueError, RecursionError):
        # A provider's non-serializable extension must not break the agent.
        return


def capture_request(span: Any, request: Any, tools: Any = None) -> None:
    """Record instructions separately from request messages and offered tools."""
    if not span or not enabled():
        return
    try:
        instructions = [
            m.get("content") for m in request or [] if isinstance(m, dict) and m.get("role") in {"system", "developer"}
        ]
        capture(span, **{"gen_ai.input.messages": messages(request, "user")})
        if instructions:
            capture(span, **{"gen_ai.system_instructions": instructions})
        if tools:
            definitions = [
                {"type": tool.get("type", "function"), **(tool.get("function") or tool)}
                if isinstance(tool, dict)
                else tool
                for tool in tools
            ]
            capture(span, **{"gen_ai.tool.definitions": definitions})
    except (TypeError, ValueError, AttributeError, RecursionError):
        return
