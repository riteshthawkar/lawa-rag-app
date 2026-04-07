import os

import pytest
from openai import OpenAI
from pinecone import Pinecone

from tests.live_utils import require_clean_env_value, require_env_values


def _require_live_configuration():
    if os.getenv("RUN_LIVE_PROVIDER_TESTS") != "1":
        pytest.skip("Live provider smoke tests are disabled")

    require_env_values("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME")


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

    api_key = _require_clean_env_value("OPENAI_API_KEY")
    if not api_key.startswith("sk-"):
        pytest.fail(
            "OPENAI_API_KEY does not look like a valid OpenAI API key. "
            "It should usually start with 'sk-'."
        )

    client = OpenAI(api_key=api_key)
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

    pinecone = Pinecone(api_key=_require_clean_env_value("PINECONE_API_KEY"))
    index_name = _require_clean_env_value("PINECONE_INDEX_NAME")

    assert pinecone.has_index(index_name) is True
