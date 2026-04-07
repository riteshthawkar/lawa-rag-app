import asyncio
import json
import os
import time
from urllib.parse import urlparse

import pytest
import requests
import websockets


def require_env_values(*names: str) -> dict[str, str]:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        pytest.skip(f"Missing live test configuration: {', '.join(missing)}")
    return {name: os.environ[name] for name in names}


def require_clean_env_value(name: str) -> str:
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


def live_app_base_url() -> str:
    return os.getenv("LIVE_APP_BASE_URL", "http://127.0.0.1:8080").rstrip("/")


def deployed_base_url() -> str:
    return os.environ["DEPLOYED_BASE_URL"].rstrip("/")


def websocket_url_from_http_base(http_base_url: str) -> str:
    parsed = urlparse(http_base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/chat"


def wait_for_http_ready(base_url: str, timeout_seconds: int = 120) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None

    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return
            last_error = f"unexpected status {response.status_code}"
        except Exception as exc:  # pragma: no cover - best effort wait loop
            last_error = str(exc)
        time.sleep(2)

    raise AssertionError(f"App did not become ready at {base_url}: {last_error}")


def post_chat_request(base_url: str, payload: dict, timeout: int = 180) -> requests.Response:
    return requests.post(f"{base_url}/telegram-chat", json=payload, timeout=timeout)


async def collect_websocket_messages(ws_url: str, token: str, payload: dict, timeout: int = 120):
    messages = []
    async with websockets.connect(f"{ws_url}?token={token}") as websocket:
        await websocket.send(json.dumps(payload))
        deadline = time.time() + timeout

        while time.time() < deadline:
            raw_message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            messages.append(raw_message)
            if '"status":"completed"' in raw_message or '"status": "completed"' in raw_message:
                return messages

    raise AssertionError("WebSocket flow did not complete before timeout")
