import os
import time

import pytest

from tests.live_utils import live_app_base_url, post_chat_request, require_env_values


TRANSIENT_RESPONSES = {
    "this question is out of my scope. please try again with another question.",
    "response generation failed. please try again later.",
}


def _require_live_app():
    if os.getenv("RUN_LIVE_PROVIDER_TESTS") != "1":
        pytest.skip("Live semantic regression tests are disabled")

    require_env_values("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME")


@pytest.mark.live
@pytest.mark.parametrize(
    ("payload", "expected_terms", "expect_sources"),
    [
        (
            {
                "question": "What is the UAE Golden Visa?",
                "language": "English",
                "previous_chats": [],
                "response_detail_level": "concise",
            },
            ("golden visa", "residence"),
            True,
        ),
        (
            {
                "question": "Who are you?",
                "language": "English",
                "previous_chats": [],
                "response_detail_level": "concise",
            },
            ("lawa.ai", "ai assistant"),
            False,
        ),
        (
            {
                "question": "How do I fix my broken iPhone screen?",
                "language": "English",
                "previous_chats": [],
                "response_detail_level": "concise",
            },
            ("scope", "uae"),
            False,
        ),
    ],
)
def test_live_semantic_regression_for_key_query_types(
    payload, expected_terms, expect_sources
):
    _require_live_app()

    response = None
    data = None
    answer = ""

    for attempt in range(3):
        response = post_chat_request(live_app_base_url(), payload)
        assert response.status_code == 200

        data = response.json()
        answer = data.get("response", "").lower()
        if answer not in TRANSIENT_RESPONSES:
            break
        if attempt < 2:
            time.sleep(2)

    assert answer

    for term in expected_terms:
        assert term in answer

    if expect_sources:
        assert len(data.get("sources", [])) >= 1
    else:
        assert data.get("sources", []) == []
