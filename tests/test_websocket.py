import pytest
from starlette.websockets import WebSocketDisconnect

from tests.fakes import FakeOpenAIClient, FakeRetriever, FakeStream


def _ws_payload():
    return {
        "question": "What is the UAE Golden Visa?",
        "language": "English",
        "previous_chats": [],
        "response_detail_level": "concise",
    }


def test_websocket_rejects_missing_token(client):
    with client.websocket_connect("/chat") as websocket:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            websocket.receive_json()

    assert exc_info.value.code == 4001


def test_websocket_rejects_failed_session_validation(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "verify_token", lambda token: {"user_id": 1})

    async def fake_validate(*args, **kwargs):
        return None

    monkeypatch.setattr(app_module, "validate_websocket_session", fake_validate)

    with client.websocket_connect("/chat?token=test-token") as websocket:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            websocket.receive_json()

    assert exc_info.value.code == 4003


def test_websocket_streams_successful_response(
    client, app_module, monkeypatch, async_noop, sample_docs
):
    monkeypatch.setattr(app_module, "verify_token", lambda token: {"user_id": 1})

    async def fake_validate(*args, **kwargs):
        return {"valid": True}

    async def fake_fetch_history(*args, **kwargs):
        return []

    async def fake_query_rewriting_agent(*args, **kwargs):
        return {
            "action": "rewrite",
            "rewritten_queries": ["UAE Golden Visa overview"],
            "rewritten_query": "UAE Golden Visa overview",
            "query_type": "General Information",
            "relevant_history_indices": [],
        }

    async def fake_emit_processing_step(*args, **kwargs):
        return None

    monkeypatch.setattr(app_module, "validate_websocket_session", fake_validate)
    monkeypatch.setattr(app_module, "fetch_chat_history", fake_fetch_history)
    monkeypatch.setattr(app_module, "save_interaction", async_noop)
    monkeypatch.setattr(app_module, "update_interaction_status", async_noop)
    monkeypatch.setattr(app_module, "emit_processing_step", fake_emit_processing_step)
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
        FakeOpenAIClient([FakeStream(["Golden Visa answer [1]."])]),
    )

    messages = []
    with client.websocket_connect(
        "/chat?token=test-token&chat_id=chat-1&interaction_id=interaction-1"
    ) as websocket:
        websocket.send_json(_ws_payload())

        for _ in range(10):
            message = websocket.receive_json()
            messages.append(message)
            if message.get("status") == "completed":
                break

    assert any(message.get("status") == "processing" for message in messages)
    final_message = next(message for message in messages if message.get("replace") is True)
    assert "Golden Visa answer" in final_message["response"]
    assert final_message["sources"] == [
        {"url": "https://example.com/golden-visa", "cite_num": "1"}
    ]
    assert messages[-1]["status"] == "completed"
