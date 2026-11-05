"""Regression coverage for the Convergent destination and export-only content."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from docsgpt import tracing
from docsgpt.core.settings import settings
from docsgpt.llm.deadlines import timed_stream
from docsgpt.tracing import content
from docsgpt.tracing.convergent import configure
from docsgpt.tracing.otel import export_trace


@pytest.fixture(autouse=True)
def tracing_settings(monkeypatch):
    monkeypatch.setattr(settings, "TRACES_ENABLED", True)
    monkeypatch.setattr(settings, "TRACES_OTEL_EXPORT", True)
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "span_only")
    for key in ("CONVERGENT_API_KEY", "CONVERGENT_SPANS_DIR", "CONVERGENT_TRACES_EXPORTER", "CONVERGENT_RELEASE"):
        monkeypatch.delenv(key, raising=False)


def test_configure_is_opt_in_and_uses_package_release(monkeypatch):
    import convergent

    init = Mock()
    monkeypatch.setattr(convergent, "init", init)
    configure()
    init.assert_not_called()
    monkeypatch.setenv("CONVERGENT_SPANS_DIR", "test-traces")
    configure()
    init.assert_called_once_with(release="1.1.0")
    monkeypatch.setenv("CONVERGENT_RELEASE", "build-42")
    configure()
    init.assert_called_with(release="build-42")


def test_content_is_opt_in_snapshot_not_a_persisted_prompt(monkeypatch):
    trace = tracing.start_trace(source="test")
    span = trace.start_span(tracing.KIND_AGENT, "agent")
    request = [{"role": "user", "content": "original"}]
    content.capture_request(span, request)
    request[0]["content"] = "mutated"
    assert "original" in span.otel_content["gen_ai.input.messages"]
    assert "original" not in json.dumps(trace.to_record())
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")
    second = trace.start_span(tracing.KIND_LLM, "chat")
    content.capture_request(second, request)
    assert not second.otel_content


def test_request_preserves_tool_round_order_and_separates_system_prompt():
    request = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "tool_calls": [{"id": "call-1", "function": {"name": "read", "arguments": '{"q":"x"}'}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "found"},
    ]
    normalized = content.messages(request, "user")
    assert [m["role"] for m in normalized] == ["user", "assistant", "tool"]
    assert normalized[1]["parts"][0]["arguments"] == {"q": "x"}
    assert normalized[2]["parts"][0]["response"] == "found"


def test_stream_joins_fragmented_tool_arguments_and_text():
    chunks = [
        SimpleNamespace(
            delta=SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(name="read", arguments='{"q":'))
                ]
            )
        ),
        SimpleNamespace(
            delta=SimpleNamespace(
                tool_calls=[SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments='"x"}'))]
            )
        ),
    ]
    message = content.messages(chunks, stream=True)[0]
    assert message["parts"] == [{"type": "tool_call", "id": "call-1", "name": "read", "arguments": {"q": "x"}}]
    assert content.messages(["hello", " world"], stream=True) == content.text_message("hello world")


def test_binary_content_not_embedded():
    serialized = json.dumps(
        content.safe_value(
            {"file_data": "secret-base64", "blob": b"binary", "image_url": "data:image/png;base64,secret"}
        )
    )
    assert "secret" not in serialized
    assert "binary, 6 bytes" in serialized


def test_timed_stream_keeps_agent_parent_without_context_leak():
    trace = tracing.start_trace(source="test")

    def provider():
        with tracing.span(tracing.KIND_LLM, "chat"):
            yield "answer"

    with tracing.activate(trace):
        with tracing.span(tracing.KIND_AGENT, "agent") as agent:
            assert list(timed_stream(provider, first_token=5, stall=5)) == ["answer"]
            assert trace.current_parent() is agent
    assert trace.spans[1].parent_id == agent.id
    assert tracing.current_trace() is None


@pytest.mark.parametrize("blocked", [False, True])
def test_export_respects_content_block_and_preserves_provider(blocked):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace = tracing.start_trace(source="test", conversation_id="conversation-1")
    span = trace.start_span(
        tracing.KIND_AGENT,
        "agent",
        attributes={"gen_ai.agent.name": "TestAgent", "gen_ai.operation.name": "invoke_agent"},
    )
    content.capture(span, **{"gen_ai.output.messages": content.text_message("deliverable")})
    span.end()
    trace.content_blocked = blocked
    trace.finish()
    export_trace(trace, tracer_provider=provider)
    exported = [s for s in exporter.get_finished_spans() if s.name == "agent"]
    assert len(exported) == 1
    assert ("gen_ai.output.messages" in exported[0].attributes) is not blocked
    assert exported[0].attributes["gen_ai.conversation.id"] == "conversation-1"
    provider.shutdown()
