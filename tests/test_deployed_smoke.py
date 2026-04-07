import os

import pytest
import requests

from tests.live_utils import deployed_base_url, post_chat_request, require_env_values


def _require_deployed_smoke():
    require_env_values("DEPLOYED_BASE_URL")


@pytest.mark.deployed
def test_deployed_health_endpoints():
    _require_deployed_smoke()

    base_url = deployed_base_url()
    assert requests.get(f"{base_url}/", timeout=20).status_code == 200
    assert requests.get(f"{base_url}/api", timeout=20).status_code == 200
    health_response = requests.get(f"{base_url}/health", timeout=20)
    assert health_response.status_code == 200
    assert health_response.json() == {"message": "working"}


@pytest.mark.deployed
def test_deployed_grounded_chat_smoke():
    _require_deployed_smoke()

    response = post_chat_request(
        deployed_base_url(),
        {
            "question": "What is the UAE Golden Visa?",
            "language": "English",
            "previous_chats": [],
            "response_detail_level": "concise",
        },
        timeout=180,
    )

    assert response.status_code == 200
    data = response.json()
    assert "golden visa" in data.get("response", "").lower()
    assert len(data.get("sources", [])) >= 1
