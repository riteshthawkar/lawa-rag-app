import importlib

import pytest


def test_default_models_use_gpt_5_family_without_pro():
    config = importlib.import_module("modules.config")

    assert config.MAIN_MODEL == "gpt-5.4"
    assert config.QUERY_REWRITING_MODEL == "gpt-5.4-mini"
    assert config.FALLBACK_MODEL == "gpt-5.4-mini"

    for model_name in (
        config.MAIN_MODEL,
        config.QUERY_REWRITING_MODEL,
        config.FALLBACK_MODEL,
    ):
        assert "pro" not in model_name.lower()


def test_validate_env_vars_requires_required_keys(monkeypatch):
    config = importlib.import_module("modules.config")

    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing required environment variables"):
        config.validate_env_vars()
