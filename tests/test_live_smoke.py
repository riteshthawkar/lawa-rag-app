import os

import pytest
from openai import OpenAI
from pinecone import Pinecone


def _require_live_configuration():
    if os.getenv("RUN_LIVE_PROVIDER_TESTS") != "1":
        pytest.skip("Live provider smoke tests are disabled")

    missing = [
        name
        for name in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME")
        if not os.getenv(name)
    ]
    if missing:
        pytest.skip(f"Missing live test configuration: {', '.join(missing)}")


@pytest.mark.live
@pytest.mark.parametrize(
    ("env_name", "default_model"),
    [
        ("QUERY_REWRITING_MODEL", "gpt-5.4-mini"),
        ("MAIN_MODEL", "gpt-5.4"),
    ],
)
def test_openai_models_are_reachable_live(env_name, default_model):
    _require_live_configuration()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=os.getenv(env_name, default_model),
        messages=[{"role": "user", "content": "Reply with OK"}],
        max_completion_tokens=8,
    )

    assert response.choices
    assert response.choices[0].message is not None


@pytest.mark.live
def test_pinecone_index_is_reachable_live():
    _require_live_configuration()

    pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]

    assert pinecone.has_index(index_name) is True
