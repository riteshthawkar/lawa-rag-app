from tests.fakes import FakeOpenAIClient, FakeRetriever, FakeStream, make_completion


def _chat_payload():
    return {
        "question": "What is the UAE Golden Visa?",
        "language": "English",
        "previous_chats": [],
        "response_detail_level": "concise",
    }


def test_health_endpoints_work(client):
    assert client.get("/").json() == {"status": "working"}
    assert client.get("/api").json() == {"message": "API is working"}
    assert client.get("/health").json() == {"message": "working"}


def test_generation_health_probe_reports_success(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "openai_client", FakeOpenAIClient([make_completion("OK")]))

    response = client.get("/health/generation")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "message": "Generation path available",
        "model": app_module.MAIN_MODEL,
        "reply": "OK",
    }


def test_generation_health_probe_reports_failure(client, app_module, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "openai_client",
        FakeOpenAIClient([RuntimeError("OpenAI unavailable")]),
    )

    response = client.get("/health/generation")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Generation probe failed"
    assert "OpenAI unavailable" in data["detail"]


def test_telegram_chat_returns_grounded_response(client, app_module, monkeypatch, sample_docs):
    async def fake_query_rewriting_agent(*args, **kwargs):
        return {
            "action": "rewrite",
            "rewritten_query": "UAE Golden Visa overview",
            "query_type": "General Information",
            "relevant_history_indices": [],
        }

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
    monkeypatch.setattr(
        app_module,
        "openai_client",
        FakeOpenAIClient([FakeStream(["The UAE Golden Visa is a long-term residence visa [1]."])]),
    )

    response = client.post("/telegram-chat", json=_chat_payload())

    assert response.status_code == 200
    data = response.json()
    assert "Golden Visa" in data["response"]
    assert data["sources"] == [
        {"url": "https://example.com/golden-visa", "cite_num": "1"}
    ]


def test_telegram_chat_handles_direct_responses(client, app_module, monkeypatch):
    async def fake_query_rewriting_agent(*args, **kwargs):
        return {
            "action": "respond",
            "response": "I can only answer UAE government questions.",
            "query_type": "Other",
        }

    monkeypatch.setattr(app_module, "query_rewriting_agent", fake_query_rewriting_agent)

    response = client.post("/telegram-chat", json=_chat_payload())

    assert response.status_code == 200
    assert response.json() == {
        "response": "I can only answer UAE government questions.",
        "sources": [],
    }


def test_probe_requests_fail_on_direct_response(client, app_module, monkeypatch):
    async def fake_query_rewriting_agent(*args, **kwargs):
        return {
            "action": "respond",
            "response": "I can only answer UAE government questions.",
            "query_type": "Other",
        }

    monkeypatch.setattr(app_module, "query_rewriting_agent", fake_query_rewriting_agent)

    response = client.post(
        "/telegram-chat",
        json=_chat_payload(),
        headers={"x-health-probe": "true"},
    )

    assert response.status_code == 503
    assert response.json()["message"] == "Synthetic probe was handled as a non-answer path"


def test_telegram_chat_handles_retrieval_failure(client, app_module, monkeypatch):
    async def fake_query_rewriting_agent(*args, **kwargs):
        return {
            "action": "rewrite",
            "rewritten_query": "UAE Golden Visa overview",
            "query_type": "General Information",
            "relevant_history_indices": [],
        }

    class BrokenRetriever:
        def invoke(self, query):
            raise RuntimeError("Pinecone unavailable")

    monkeypatch.setattr(app_module, "query_rewriting_agent", fake_query_rewriting_agent)
    monkeypatch.setattr(client.app.state, "retriever", BrokenRetriever())

    response = client.post("/telegram-chat", json=_chat_payload())

    assert response.status_code == 200
    assert response.json() == {
        "response": "This question is out of my scope. Please try again with another question.",
        "sources": [],
    }


def test_telegram_chat_handles_generation_failure(client, app_module, monkeypatch, sample_docs):
    async def fake_query_rewriting_agent(*args, **kwargs):
        return {
            "action": "rewrite",
            "rewritten_query": "UAE Golden Visa overview",
            "query_type": "General Information",
            "relevant_history_indices": [],
        }

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
    monkeypatch.setattr(
        app_module,
        "openai_client",
        FakeOpenAIClient([RuntimeError("OpenAI unavailable")]),
    )

    response = client.post("/telegram-chat", json=_chat_payload())

    assert response.status_code == 200
    assert response.json() == {
        "response": "Response generation failed. Please try again later.",
        "sources": [],
    }


def test_telegram_chat_falls_back_when_rerank_times_out(
    client, app_module, monkeypatch, sample_docs
):
    async def fake_query_rewriting_agent(*args, **kwargs):
        return {
            "action": "rewrite",
            "rewritten_query": "UAE Golden Visa overview",
            "query_type": "General Information",
            "relevant_history_indices": [],
        }

    monkeypatch.setattr(app_module, "query_rewriting_agent", fake_query_rewriting_agent)
    monkeypatch.setattr(client.app.state, "retriever", FakeRetriever(sample_docs))

    def exploding_rerank(*args, **kwargs):
        raise TimeoutError("Pinecone rerank timeout")

    fake_client = FakeOpenAIClient(
        [FakeStream(["The UAE Golden Visa is a long-term residence visa [1]."])]
    )

    monkeypatch.setattr(app_module, "rerank_docs", exploding_rerank)
    monkeypatch.setattr(app_module, "openai_client", fake_client)

    response = client.post("/telegram-chat", json=_chat_payload())

    assert response.status_code == 200
    data = response.json()
    assert "Golden Visa" in data["response"]
    assert data["sources"] == [
        {"url": "https://example.com/golden-visa", "cite_num": "1"}
    ]


def test_telegram_chat_rejects_invalid_payload(client):
    response = client.post("/telegram-chat", json={})
    assert response.status_code == 422
