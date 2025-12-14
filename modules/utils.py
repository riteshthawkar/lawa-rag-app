import logging
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict

from modules.config import logger

async def safe_send(websocket, message: dict):
    """Send messages safely over the websocket with proper error handling"""
    try:
        # Check if websocket application state is connected
        from starlette.websockets import WebSocketState
        if websocket.client_state == WebSocketState.DISCONNECTED:
             logger.warning("Attempted to send message to closed websocket.")
             return

        await websocket.send_json(message)
    except RuntimeError as e:
        if "Cannot call \"send\" once a close message has been sent" in str(e):
            logger.warning("WebSocket already closed when attempting to send message.")
        else:
            logger.error(f"RuntimeError in safe_send: {e}")
    except WebSocketDisconnect:
        logger.info("Client disconnected during send")
        raise
    except Exception as e:
        logger.exception("Error sending message:")
        raise

def format_docs(docs: List[dict]) -> str:
    """Format documents for inclusion in prompt"""
    context = ""
    for index, ele in enumerate(docs):
        context += (
            f"\n{'=' * 150}\n"
            f"**DOCUMENT:** {index + 1}\n"
            f"**SOURCE:** {ele['page_source']}\n\n"
            f"**CONTENT:** {ele['chunk']}\n\n"
        )
    return context

def format_query(query: str, language: str, docs: List[dict]) -> str:
    """Format the query with language and document context"""
    formatted_docs = format_docs(docs)
    return f"**USER QUERY:** {query}\n**LANGUAGE:** {language}\n**CONTEXT:**\n{formatted_docs}" 