import asyncio
import json
import os

import pytest

from tests.live_utils import (
    collect_websocket_messages,
    live_app_base_url,
    require_env_values,
    websocket_url_from_http_base,
)


def _require_websocket_e2e():
    if os.getenv("RUN_LIVE_PROVIDER_TESTS") != "1":
        pytest.skip("Live websocket tests are disabled")

    require_env_values(
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "WEBSOCKET_TEST_TOKEN",
        "BACKEND_URL",
    )


@pytest.mark.live
def test_live_websocket_end_to_end_flow():
    _require_websocket_e2e()

    payload = {
        "question": "What is the UAE Golden Visa?",
        "language": "English",
        "previous_chats": [],
        "response_detail_level": "concise",
    }

    messages = asyncio.run(
        collect_websocket_messages(
            websocket_url_from_http_base(live_app_base_url()),
            os.environ["WEBSOCKET_TEST_TOKEN"],
            payload,
        )
    )

    decoded_messages = [json.loads(message) for message in messages]

    assert any("processing_step" in message for message in decoded_messages)
    assert any("chat_id" in message for message in decoded_messages)
    assert any("interaction_id" in message for message in decoded_messages)
    assert any(message.get("status") == "processing" for message in decoded_messages)
    assert any(message.get("status") == "streaming" for message in decoded_messages)
    assert any(message.get("status") == "completed" for message in decoded_messages)

    final_message = next(
        message for message in decoded_messages if message.get("replace") is True
    )
    assert "golden visa" in final_message["response"].lower()
    assert len(final_message.get("sources", [])) >= 1
