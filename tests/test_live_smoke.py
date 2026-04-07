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


def _require_clean_env_value(name: str) -> str:
    value = os.environ[name]

    if value in {"***", "your-openai-api-key", "your-pinecone-api-key"}:
        pytest.fail(
            f"{name} appears to still be a placeholder value. "
            "Update the GitHub Actions secret or variable with the real credential."
        )

    if value != value.strip():
        pytest.fail(
            f"{name} contains leading or trailing whitespace. "
            "Re-save it in GitHub without quotes or extra line breaks."
        )

    if any(ch in value for ch in ("\r", "\n", "\t", "\x00")):
        pytest.fail(
            f"{name} contains control characters. "
            "Re-save it in GitHub as a single plain-text line."
        )

    return value


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
