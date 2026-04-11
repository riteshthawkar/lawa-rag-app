from tests.fakes import (
    FakeOpenAIClient,
    FakeRetriever,
    FakeStream,
    make_completion,
)


def _chat_payload():
    return {
        "question": "What is the UAE Golden Visa?",
        "language": "English",
        "previous_chats": [],
        "response_detail_level": "concise",
    }


def test_health_endpoints_work(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "openai_client", FakeOpenAIClient([]))

    health_response = client.get("/health")
    detailed_response = client.get("/health/detailed")

    assert client.get("/").json() == {"status": "working"}
    assert client.get("/api").json() == {"message": "API is working"}
    assert health_response.status_code == 200
    assert detailed_response.status_code == 200

    health_data = health_response.json()
    assert health_data["status"] == "healthy"
    assert health_data["service"] == "lawa-rag"
    assert "timestamp" in health_data
    assert "checks" in health_data
    assert health_data["checks"]["retriever"]["status"] == "healthy"
    assert health_data["checks"]["vector_store_client"]["status"] == "healthy"

    detailed_data = detailed_response.json()
    assert detailed_data["status"] == "healthy"
    assert detailed_data["checks"]["embedding_model"]["status"] == "healthy"
    assert detailed_data["checks"]["vector_store"]["status"] == "healthy"
    assert detailed_data["checks"]["generation_model"]["status"] == "healthy"
    assert detailed_data["checks"]["query_rewriting_model"]["status"] == "healthy"
    assert detailed_data["checks"]["reranker"]["status"] == "healthy"


def test_growth_dashboard_page_renders(client):
    response = client.get("/demo/growth-dashboard")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "Users and Queries Over Time" in body
    assert "Download PNG" in body
    assert "Daily Users" in body
    assert "Daily Queries" in body
    assert "30,000" in body
    assert "2,200" in body


def test_generation_health_probe_reports_success(client, app_module, monkeypatch):
    async def fake_query_rewriting_agent(*args, **kwargs):
        return {
            "action": "rewrite",
            "rewritten_query": "UAE Golden Visa overview",
            "query_type": "General Information",
            "relevant_history_indices": [],
        }

    monkeypatch.setattr(app_module, "query_rewriting_agent", fake_query_rewriting_agent)
    monkeypatch.setattr(
        app_module,
        "rerank_docs",
        lambda *args, **kwargs: [
            {
                "page_source": "https://example.com/golden-visa",
                "chunk": "The UAE Golden Visa is a long-term residence visa.",
                "summary": "Golden Visa overview",
            }
        ],
    )
    monkeypatch.setattr(app_module, "openai_client", FakeOpenAIClient([make_completion("OK")]))

    response = client.get("/health/generation")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "lawa-rag"
    assert data["checks"]["query_rewriting"]["status"] == "healthy"
    assert data["checks"]["retrieval"]["status"] == "healthy"
    assert data["checks"]["reranking"]["status"] == "healthy"
    assert data["checks"]["generation"]["status"] == "healthy"
    assert data["checks"]["generation"]["reply"] == "OK"


def test_generation_health_probe_reports_failure(client, app_module, monkeypatch):
    async def fake_query_rewriting_agent(*args, **kwargs):
        return {
            "action": "rewrite",
            "rewritten_query": "UAE Golden Visa overview",
            "query_type": "General Information",
            "relevant_history_indices": [],
        }

    monkeypatch.setattr(app_module, "query_rewriting_agent", fake_query_rewriting_agent)
    monkeypatch.setattr(
        app_module,
        "rerank_docs",
        lambda *args, **kwargs: [
            {
                "page_source": "https://example.com/golden-visa",
                "chunk": "The UAE Golden Visa is a long-term residence visa.",
                "summary": "Golden Visa overview",
            }
        ],
    )
    monkeypatch.setattr(
        app_module,
        "openai_client",
        FakeOpenAIClient([RuntimeError("OpenAI unavailable")]),
    )

    response = client.get("/health/generation")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["checks"]["generation"]["status"] == "unhealthy"
    assert "OpenAI unavailable" in data["checks"]["generation"]["detail"]


def test_generation_health_probe_reports_retrieval_failure(client, app_module, monkeypatch):
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
    monkeypatch.setattr(app_module, "openai_client", FakeOpenAIClient([]))

    response = client.get("/health/generation")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["checks"]["query_rewriting"]["status"] == "healthy"
    assert data["checks"]["retrieval"]["status"] == "unhealthy"
    assert "Pinecone unavailable" in data["checks"]["retrieval"]["detail"]
    assert data["checks"]["reranking"]["detail"] == "Skipped because retrieval failed"
    assert data["checks"]["generation"]["detail"] == "Skipped because retrieval failed"


def test_generation_health_probe_reports_reranking_failure(client, app_module, monkeypatch):
    async def fake_query_rewriting_agent(*args, **kwargs):
        return {
            "action": "rewrite",
            "rewritten_query": "UAE Golden Visa overview",
            "query_type": "General Information",
            "relevant_history_indices": [],
        }

    monkeypatch.setattr(app_module, "query_rewriting_agent", fake_query_rewriting_agent)
    monkeypatch.setattr(
        app_module,
        "rerank_docs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Rerank unavailable")),
    )
    monkeypatch.setattr(app_module, "openai_client", FakeOpenAIClient([]))

    response = client.get("/health/generation")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["checks"]["retrieval"]["status"] == "healthy"
    assert data["checks"]["reranking"]["status"] == "unhealthy"
    assert "Rerank unavailable" in data["checks"]["reranking"]["detail"]
    assert data["checks"]["generation"]["detail"] == "Skipped because reranking failed"


def test_health_detailed_reports_unhealthy_model_configuration(client, app_module, monkeypatch):
    model_failures = {
        app_module.settings.MAIN_MODEL: RuntimeError("model not available"),
    }
    monkeypatch.setattr(
        app_module,
        "openai_client",
        FakeOpenAIClient([], model_failures=model_failures),
    )

    response = client.get("/health/detailed")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["checks"]["generation_model"]["status"] == "unhealthy"
    assert "model not available" in data["checks"]["generation_model"]["detail"]


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


def test_telegram_chat_preserves_page_source_metadata(client, app_module, monkeypatch):
    async def fake_query_rewriting_agent(*args, **kwargs):
        return {
            "action": "rewrite",
            "rewritten_query": "UAE Golden Visa overview",
            "query_type": "General Information",
            "relevant_history_indices": [],
        }

    docs = [
        type(
            "Doc",
            (),
            {
                "page_content": "The UAE Golden Visa is a long-term residence visa.",
                "metadata": {
                    "page_source": "https://example.com/page-source",
                    "summary": "Golden Visa overview",
                },
            },
        )()
    ]

    monkeypatch.setattr(app_module, "query_rewriting_agent", fake_query_rewriting_agent)
    monkeypatch.setattr(client.app.state, "retriever", FakeRetriever(docs))
    monkeypatch.setattr(
        app_module,
        "rerank_docs",
        lambda *args, **kwargs: [
            {
                "page_source": "https://example.com/page-source",
                "chunk": docs[0].page_content,
                "summary": docs[0].metadata["summary"],
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
    assert response.json()["sources"] == [
        {"url": "https://example.com/page-source", "cite_num": "1"}
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
