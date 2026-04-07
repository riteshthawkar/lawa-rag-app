import importlib
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeDocument, FakeRetriever


os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("SECRET_KEY", "x" * 40)
os.environ.setdefault("MAIN_MODEL", "gpt-5.4")
os.environ.setdefault("QUERY_REWRITING_MODEL", "gpt-5.4-mini")
os.environ.setdefault("FALLBACK_MODEL", "gpt-5.4-mini")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")


@pytest.fixture
def sample_docs():
    return [
        FakeDocument(
            "The UAE Golden Visa is a long-term residence visa.",
            {
                "source": "https://example.com/golden-visa",
                "summary": "Golden Visa overview",
            },
        )
    ]


@pytest.fixture
def app_module(monkeypatch, sample_docs):
    app_module = importlib.import_module("app")

    monkeypatch.setattr(app_module.nltk, "download", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        app_module,
        "initialize_pinecone",
        lambda: (FakeRetriever(sample_docs), SimpleNamespace()),
    )

    return app_module


@pytest.fixture
def client(app_module):
    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def async_noop():
    async def _noop(*args, **kwargs):
        return None

    return _noop
