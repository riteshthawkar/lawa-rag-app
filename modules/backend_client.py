import aiohttp
import asyncio
import os
from typing import Optional
import json
from .config import logger

# Base URL for the Django backend - configurable via environment
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

async def validate_websocket_session(token: str, chat_id: Optional[str] = None):
    """
    Ensures the supplied token/chat are valid by calling the backend.
    """
    url = f"{BACKEND_URL}/api/internal/ws-auth/"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {}
    if chat_id:
        payload["chat_id"] = chat_id

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=5) as response:
                if response.status == 200:
                    return await response.json()
                logger.error("Websocket auth validation failed with status %s", response.status)
                return None
    except Exception as exc:
        logger.error("Error validating websocket session: %s", exc)
        return None

async def fetch_chat_history(chat_id: str, token: str):
    """
    Fetches chat history from the Django backend.
    """
    url = f"{BACKEND_URL}/api/internal/chats/{chat_id}/history/"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("history", [])
                else:
                    logger.error(f"Failed to fetch history: {response.status}")
                    return []
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return []

async def save_interaction(chat_id: str, user_message: str, assistant_message: str, sources: list, token: str, query_type: Optional[str] = None, interaction_id: Optional[str] = None):
    """
    Saves the interaction to the Django backend.
    """
    url = f"{BACKEND_URL}/api/internal/save-interaction/"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "chat_id": chat_id,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "sources": sources,
        "query_type": query_type,
        "interaction_id": interaction_id
    }
    
    try:
        logger.info(f"Sending persistence request to {url}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=10) as response:
                logger.info(f"Persistence response status: {response.status}")
                if response.status == 201:
                    logger.info("Interaction saved successfully")
                else:
                    text = await response.text()
                    logger.error(f"Failed to save interaction: {response.status} - {text}")
    except Exception as e:
        logger.error(f"Error saving interaction: {e}")
        import traceback
        traceback.print_exc()


async def emit_processing_step(websocket, step_id: str, step_label: str):
    """
    Emit a processing step status to the frontend.
    Shows users real-time progress through the RAG pipeline.
    """
    from datetime import datetime
    try:
        await websocket.send_json({
            "processing_step": {
                "id": step_id,
                "label": step_label,
                "timestamp": datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Error emitting processing step: {e}")


async def update_interaction_status(interaction_id: str, status: str, token: str, error_message: Optional[str] = None):
    """
    Sends status updates back to the backend so the UI can reflect progress.
    """
    if not interaction_id:
        return

    url = f"{BACKEND_URL}/api/internal/interactions/status/"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"interaction_id": interaction_id, "status": status}
    if error_message:
        payload["error_message"] = error_message

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=5) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error("Failed to update interaction status: %s - %s", response.status, text)
    except Exception as exc:
        logger.error("Error updating interaction status: %s", exc)


async def create_chat(token: str, title: str = "New Chat", language: str = "English") -> Optional[str]:
    """
    Creates a new chat session in the backend.
    """
    url = f"{BACKEND_URL}/api/chats/"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "title": title,
        "language": language,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=5) as response:
                if response.status == 201:
                    data = await response.json()
                    return data.get("id")
                else:
                    text = await response.text()
                    logger.error(f"Failed to create chat: {response.status} - {text}")
                    return None
    except Exception as e:
        logger.error(f"Error creating chat: {e}")
        return None


async def create_interaction(token: str, chat_id: str, user_message: str) -> Optional[str]:
    """
    Creates a new interaction in the backend.
    """
    url = f"{BACKEND_URL}/api/interactions/"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "chat_id": chat_id,
        "user_message": user_message,
        "llm_response": "",
        "sources": [],
        "status": "processing"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=5) as response:
                if response.status == 201:
                    data = await response.json()
                    return data.get("id")
                else:
                    text = await response.text()
                    logger.error(f"Failed to create interaction: {response.status} - {text}")
                    return None
    except Exception as e:
        logger.error(f"Error creating interaction: {e}")
        return None
