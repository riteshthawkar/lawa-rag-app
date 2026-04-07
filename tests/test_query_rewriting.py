import asyncio
import importlib
import json
from types import SimpleNamespace

from tests.fakes import FakeOpenAIClient, make_completion


def test_query_rewriting_agent_falls_back_on_llm_error(monkeypatch):
    query_module = importlib.import_module("modules.query_rewriting")

    class ExplodingCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("OpenAI unavailable")

    monkeypatch.setattr(
        query_module,
        "openai_client",
        SimpleNamespace(chat=SimpleNamespace(completions=ExplodingCompletions())),
    )

    result = asyncio.run(
        query_module.query_rewriting_agent(
            question="What is the UAE Golden Visa?",
            language="English",
            message_history=[],
        )
    )

    assert result["action"] == "rewrite"
    assert result["rewritten_query"] == "What is the UAE Golden Visa?"
    assert result["query_type"] == "Error"


def test_expand_query_uses_max_completion_tokens(monkeypatch):
    query_module = importlib.import_module("modules.query_rewriting")
    fake_client = FakeOpenAIClient(
        [make_completion(json.dumps({"queries": ["query one", "query two", "query three"]}))]
    )
    monkeypatch.setattr(query_module, "openai_client", fake_client)

    result = asyncio.run(query_module.expand_query_with_domain_knowledge("golden visa"))

    assert result == ["query one", "query two", "query three"]
    call = fake_client.chat.completions.calls[0]
    assert call["max_completion_tokens"] == 500
    assert "max_tokens" not in call


def test_combine_queries_uses_max_completion_tokens(monkeypatch):
    query_module = importlib.import_module("modules.query_rewriting")
    fake_client = FakeOpenAIClient([make_completion("Combined query")])
    monkeypatch.setattr(query_module, "openai_client", fake_client)

    result = asyncio.run(
        query_module.combine_queries(
            original_query="visa",
            clarification_response="for investors",
            language="English",
        )
    )

    assert result == "Combined query"
    call = fake_client.chat.completions.calls[0]
    assert call["max_completion_tokens"] == 300
    assert "max_tokens" not in call
