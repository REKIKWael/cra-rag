"""Tests for the OpenAI-compatible client (LiteLLM proxy, vLLM...), with a simulated server."""

import json

import httpx
import pytest

from cra_rag import llm


def sse(*chunks):
    """A streamed reply as an OpenAI-compatible server sends it (Server-Sent Events)."""
    lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
    return httpx.Response(200, content=("\n\n".join(lines) + "\n\n").encode())


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_chat_streams_the_answer_and_reads_the_usage():
    seen = {}

    def server(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return sse(
            {"choices": [{"delta": {"content": "Au plus "}}]},
            {"choices": [{"delta": {"content": "tard 24 heures."}}]},
            {"choices": [], "usage": {"prompt_tokens": 1800, "completion_tokens": 9}},
        )

    model = llm.OpenAICompatibleLLM("ministral-3:3b", "http://proxy:4000/v1/", "secret",
                                    client=client(server))  # fmt: skip
    pieces = []
    completion = model.chat("system", "user", on_token=pieces.append)

    assert seen["url"] == "http://proxy:4000/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["temperature"] == 0 and seen["body"]["stream"] is True
    assert pieces == ["Au plus ", "tard 24 heures."]
    assert completion.text == "Au plus tard 24 heures."
    assert (completion.prompt_tokens, completion.output_tokens) == (1800, 9)


def test_a_schema_becomes_a_response_format():
    seen = {}

    def server(request):
        seen.update(json.loads(request.content))
        return sse({"choices": [{"delta": {"content": '{"verdict": "correct"}'}}]})

    llm.OpenAICompatibleLLM("judge", "http://x/v1", client=client(server)).chat(
        "s", "u", schema={"type": "object"}
    )
    assert seen["response_format"]["type"] == "json_schema"
    assert seen["response_format"]["json_schema"]["schema"] == {"type": "object"}


def test_tool_calls_streamed_in_pieces_are_put_back_together():
    def server(request):
        return sse(
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a",
                "function": {"name": "search", "arguments": '{"que'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0,
                "function": {"arguments": 'ry": "délai"}'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "b",
                "function": {"name": "read_section", "arguments": '{"id": "cra:art_14"}'}}]}}]},
        )  # fmt: skip

    model = llm.OpenAICompatibleLLM("m", "http://x/v1", client=client(server))
    completion = model.converse([{"role": "user", "content": "?"}], tools=[{"type": "function"}])
    assert completion.tool_calls == (
        llm.ToolCall("search", {"query": "délai"}),
        llm.ToolCall("read_section", {"id": "cra:art_14"}),
    )


def test_the_agent_conversation_is_translated_to_the_openai_format():
    ollama_style = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "search", "arguments": {"query": "délai"}}},
            {"function": {"name": "read_section", "arguments": {"id": "cra:art_14"}}}]},
        {"role": "tool", "tool_name": "search", "content": "passages"},
        {"role": "tool", "tool_name": "read_section", "content": "article"},
    ]  # fmt: skip
    openai_style = llm.to_openai_messages(ollama_style)

    calls = openai_style[2]["tool_calls"]
    assert [c["id"] for c in calls] == ["call_0", "call_1"]
    assert calls[0]["function"]["arguments"] == '{"query": "d\\u00e9lai"}'  # JSON text
    assert openai_style[3] == {"role": "tool", "tool_call_id": "call_0", "content": "passages"}
    assert openai_style[4]["tool_call_id"] == "call_1"


def test_a_refused_key_is_explained():
    model = llm.OpenAICompatibleLLM(
        "m", "http://x/v1", client=client(lambda r: httpx.Response(401))
    )
    with pytest.raises(llm.LLMError, match="CRA_RAG_LLM_API_KEY"):
        model.chat("s", "u")


def test_connect_chooses_the_server_from_the_environment(monkeypatch):
    monkeypatch.delenv("CRA_RAG_LLM_URL", raising=False)
    assert isinstance(llm.connect("ministral-3:3b"), llm.OllamaLLM)

    monkeypatch.setenv("CRA_RAG_LLM_URL", "http://localhost:4000/v1")
    chosen = llm.connect("ministral-3:3b")
    assert isinstance(chosen, llm.OpenAICompatibleLLM)
    assert chosen.base_url == "http://localhost:4000/v1"


def test_an_error_without_a_key_suggests_setting_one():
    failing = client(lambda r: httpx.Response(500, text="No api key passed in."))
    model = llm.OpenAICompatibleLLM("m", "http://x/v1", client=failing)
    with pytest.raises(llm.LLMError, match="no key was sent"):
        model.chat("s", "u")
