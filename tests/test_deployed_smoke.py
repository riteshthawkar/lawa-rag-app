import os

import pytest
import requests

from tests.live_utils import deployed_base_url, require_env_values


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
    health_data = health_response.json()
    assert health_data["status"] in {"healthy", "degraded"}
    assert "checks" in health_data

    detailed_response = requests.get(f"{base_url}/health/detailed", timeout=60)
    assert detailed_response.status_code == 200
    detailed_data = detailed_response.json()
    assert detailed_data["status"] in {"healthy", "degraded"}
    assert "checks" in detailed_data

    generation_response = requests.get(f"{base_url}/health/generation", timeout=120)
    assert generation_response.status_code == 200
    generation_data = generation_response.json()
    assert generation_data["status"] == "healthy"
    assert generation_data["checks"]["generation"]["status"] == "healthy"


@pytest.mark.deployed
def test_deployed_grounded_chat_smoke():
    _require_deployed_smoke()

    response = requests.post(
        f"{deployed_base_url()}/telegram-chat",
        json={
            "question": "What is the UAE Golden Visa?",
            "language": "English",
            "previous_chats": [],
            "response_detail_level": "concise",
        },
        headers={"x-health-probe": "true"},
        timeout=180,
    )

    assert response.status_code == 200
    data = response.json()
    assert "golden visa" in data.get("response", "").lower()
    assert len(data.get("sources", [])) >= 1
    assert "response generation failed" not in data.get("response", "").lower()
