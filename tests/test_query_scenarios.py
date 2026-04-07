import pytest

from tests.fakes import FakeOpenAIClient, FakeRetriever, FakeStream


def _base_payload(question, response_detail_level="concise", previous_chats=None):
    return {
        "question": question,
        "language": "English",
        "previous_chats": previous_chats or [],
        "response_detail_level": response_detail_level,
    }


@pytest.mark.parametrize(
    ("question", "agent_result", "expected_response"),
    [
        (
            "Who are you?",
            {
                "action": "respond",
                "response": "I am an AI assistant developed by Lawa.ai.",
                "query_type": "Identity",
            },
            "I am an AI assistant developed by Lawa.ai.",
        ),
        (
            "visas",
            {
                "action": "clarify",
                "response": "Could you clarify which visa type you mean?",
                "query_type": "Clarification",
            },
            "Could you clarify which visa type you mean?",
        ),
        (
            "How do I fix my broken iPhone screen?",
            {
                "action": "respond",
                "response": "I can only answer questions related to UAE government topics.",
                "query_type": "Other",
            },
            "I can only answer questions related to UAE government topics.",
        ),
    ],
)
def test_query_scenarios_for_direct_responses(
    client, app_module, monkeypatch, question, agent_result, expected_response
):
    async def fake_query_rewriting_agent(*args, **kwargs):
        return agent_result

    monkeypatch.setattr(app_module, "query_rewriting_agent", fake_query_rewriting_agent)

    response = client.post("/telegram-chat", json=_base_payload(question))

    assert response.status_code == 200
    assert response.json() == {"response": expected_response, "sources": []}


def test_query_scenario_for_government_question_uses_detail_level_token_budget(
    client, app_module, monkeypatch, sample_docs
):
    async def fake_query_rewriting_agent(*args, **kwargs):
        return {
            "action": "rewrite",
            "rewritten_query": "UAE Golden Visa overview",
            "query_type": "General Information",
            "relevant_history_indices": [],
        }

    fake_client = FakeOpenAIClient(
        [FakeStream(["The UAE Golden Visa supports long-term residence [1]."])]
    )

    monkeypatch.setattr(app_module, "query_rewriting_agent", fake_query_rewriting_agent)
    monkeypatch.setattr(client.app.state, "retriever", FakeRetriever(sample_docs))
    monkeypatch.setattr(
        app_module,
        "rerank_docs",
        lambda *args, **kwargs: [
            {
                "page_source": "https://example.com/golden-visa",
                "chunk": sample_docs[0].page_content,
                "summary": sample_docs[0].metadata["summary"],
            }
        ],
    )
    monkeypatch.setattr(app_module, "openai_client", fake_client)

    response = client.post(
        "/telegram-chat",
        json=_base_payload(
            "What is the UAE Golden Visa?",
            response_detail_level="detailed",
        ),
    )

    assert response.status_code == 200
    data = response.json()
    assert "Golden Visa" in data["response"]
    assert data["sources"] == [
        {"url": "https://example.com/golden-visa", "cite_num": "1"}
    ]

    final_generation_call = fake_client.chat.completions.calls[0]
    assert (
        final_generation_call["max_completion_tokens"]
        == app_module.get_max_tokens_for_detail_level("detailed")
    )
