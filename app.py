import asyncio
import os
import re
import sys
from typing import Optional
from contextlib import asynccontextmanager

import nltk
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

# Import modules
from modules import config as settings
from modules.config import (
    logger, validate_env_vars, system_prompt, get_system_prompt,
    ENABLE_CLARIFICATION, MAX_CLARIFICATION_ATTEMPTS,
    MAIN_MODEL, FALLBACK_MODEL, MAX_COMPLETION_TOKENS,
    MAIN_MODEL_TEMPERATURE, FALLBACK_MODEL_TEMPERATURE,
    WEBSOCKET_TIMEOUT, CHUNK_BUFFER_SIZE, HOST, PORT, CORS_ORIGINS,
    get_max_tokens_for_detail_level
)
from modules.monitoring import (
    DEGRADED,
    HEALTHY,
    UNHEALTHY,
    build_health_payload,
    component,
    health_status_code,
    utc_timestamp,
)
from modules.schemas import ChatRequest, CitationSource
from modules.utils import safe_send, format_query
from modules.citations import process_citations
from modules.retrieval import initialize_pinecone, rerank_docs
from modules.query_rewriting import query_rewriting_agent, openai_client, handle_clarification_response
from modules.auth import verify_token
from modules.backend_client import (
    fetch_chat_history,
    save_interaction,
    validate_websocket_session,
    update_interaction_status,
    create_chat,
    create_interaction,
    emit_processing_step,
)
from modules.demo_dashboard import build_growth_dashboard_page


# ------------------------------------------------------------------------------
# Initialize application and validate environment
# ------------------------------------------------------------------------------
validate_env_vars()

# ------------------------------------------------------------------------------
# Lifespan context manager for startup/shutdown events
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Download NLTK data and initialize Pinecone
    nltk.download("punkt", quiet=True)
    app.state.retriever, app.state.pc = initialize_pinecone()
    logger.info("Application started successfully")
    yield
    # Shutdown cleanup (if needed)
    logger.info("Application shutting down")

# ------------------------------------------------------------------------------
# Initialize FastAPI app with CORS middleware (restrict origins in production)
# ------------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def emit_status_update(
    websocket: WebSocket,
    interaction_id: str,
    token: str,
    status_value: str,
    error_message: Optional[str] = None,
):
    """
    Send status updates to both the frontend client and backend analytics.
    """
    status_payload = {
        "status": status_value,
        "interaction_id": interaction_id,
    }
    if error_message:
        status_payload["error"] = error_message

    try:
        await safe_send(websocket, status_payload)
    except Exception:
        logger.exception("Failed to send status payload to client.")

    if interaction_id and token:
        await update_interaction_status(interaction_id, status_value, token, error_message)


def is_health_probe(request: Request) -> bool:
    return request.headers.get("x-health-probe", "").lower() in {"1", "true", "yes"}


def build_release_metadata() -> dict:
    return {
        "version": settings.RELEASE_VERSION,
        "commit_sha": settings.COMMIT_SHA,
        "deployed_at": settings.DEPLOYED_AT,
        "deployed_by": settings.DEPLOYED_BY,
        "notes": settings.RELEASE_NOTES_SUMMARY,
    }


def build_operations_metadata() -> dict:
    return {
        "owner": settings.SERVICE_OWNER,
        "runbook_url": settings.RUNBOOK_URL,
        "dashboard_service_id": settings.DASHBOARD_SERVICE_ID,
        "repository_url": settings.PRIMARY_REPOSITORY_URL,
        "public_base_url": settings.PUBLIC_BASE_URL,
    }


def metadata_check(values: dict, *, required_keys: list[str]) -> dict:
    missing_fields = []
    for key in required_keys:
        value = values.get(key)
        if value in {None, "", "unknown"}:
            missing_fields.append(key)

    if missing_fields and settings.SERVICE_ENVIRONMENT.lower() not in {
        "development",
        "dev",
        "test",
        "local",
    }:
        return component(
            DEGRADED,
            missing_fields=missing_fields,
            detail="Monitoring metadata is incomplete for this deployment.",
        )

    return component(HEALTHY)


def backend_integration_check() -> dict:
    backend_url = os.getenv("BACKEND_URL", "").strip()
    if backend_url:
        return component(HEALTHY, url=backend_url)
    if settings.SERVICE_ENVIRONMENT.lower() in {"development", "dev", "test", "local"}:
        return component(
            HEALTHY,
            detail="BACKEND_URL is not configured in the local/test environment.",
        )
    return component(
        DEGRADED,
        detail="BACKEND_URL is not configured. WebSocket session validation will use the local fallback.",
    )


def model_config_check(model_name: str, *, label: str) -> dict:
    if not model_name:
        return component(UNHEALTHY, detail=f"{label} is not configured")
    return component(HEALTHY, model=model_name)


def basic_health_checks(request: Request) -> dict[str, dict]:
    retriever = getattr(request.app.state, "retriever", None)
    pinecone_client = getattr(request.app.state, "pc", None)
    release = build_release_metadata()
    operations = build_operations_metadata()

    return {
        "application": component(HEALTHY),
        "retriever": component(HEALTHY) if retriever else component(
            UNHEALTHY,
            detail="Retriever is not initialized",
        ),
        "vector_store_client": component(
            HEALTHY,
            index=settings.PINECONE_INDEX_NAME,
        ) if pinecone_client else component(
            UNHEALTHY,
            detail="Vector store client is not initialized",
            index=settings.PINECONE_INDEX_NAME,
        ),
        "generation_model_config": model_config_check(
            settings.MAIN_MODEL,
            label="MAIN_MODEL",
        ),
        "query_rewriting_model_config": model_config_check(
            settings.QUERY_REWRITING_MODEL,
            label="QUERY_REWRITING_MODEL",
        ),
        "backend_integration": backend_integration_check(),
        "release_metadata": metadata_check(
            release,
            required_keys=["version", "commit_sha", "deployed_at"],
        ),
        "operations_metadata": metadata_check(
            operations,
            required_keys=["owner", "repository_url", "public_base_url"],
        ),
    }


def namespace_count(index_stats) -> int:
    namespaces = {}
    if isinstance(index_stats, dict):
        namespaces = index_stats.get("namespaces", {})
    else:
        namespaces = getattr(index_stats, "namespaces", {}) or {}
    return len(namespaces)


async def detailed_health_checks(request: Request) -> dict[str, dict]:
    checks = basic_health_checks(request)
    retriever = getattr(request.app.state, "retriever", None)
    pinecone_client = getattr(request.app.state, "pc", None)

    embeddings = getattr(retriever, "embeddings", None) if retriever else None
    if embeddings:
        checks["embedding_model"] = component(
            HEALTHY,
            model=getattr(embeddings, "model_name", settings.EMBEDDING_MODEL),
        )
    else:
        checks["embedding_model"] = component(
            UNHEALTHY,
            model=settings.EMBEDDING_MODEL,
            detail="Embedding model is not attached to the retriever",
        )

    if pinecone_client:
        try:
            index_stats = await asyncio.to_thread(
                lambda: pinecone_client.Index(settings.PINECONE_INDEX_NAME).describe_index_stats()
            )
            checks["vector_store"] = component(
                HEALTHY,
                index=settings.PINECONE_INDEX_NAME,
                namespace_count=namespace_count(index_stats),
            )
        except Exception as exc:
            checks["vector_store"] = component(
                UNHEALTHY,
                index=settings.PINECONE_INDEX_NAME,
                detail=str(exc)[:200],
            )
    else:
        checks["vector_store"] = component(
            UNHEALTHY,
            index=settings.PINECONE_INDEX_NAME,
            detail="Vector store client is not initialized",
        )

    rerank_callable = getattr(getattr(pinecone_client, "inference", None), "rerank", None)
    if callable(rerank_callable):
        checks["reranker"] = component(HEALTHY, model=settings.RERANK_MODEL)
    else:
        checks["reranker"] = component(
            UNHEALTHY,
            model=settings.RERANK_MODEL,
            detail="Pinecone rerank inference client is unavailable",
        )

    try:
        await openai_client.models.retrieve(settings.MAIN_MODEL)
        checks["generation_model"] = component(
            HEALTHY,
            model=settings.MAIN_MODEL,
        )
    except Exception as exc:
        checks["generation_model"] = component(
            UNHEALTHY,
            model=settings.MAIN_MODEL,
            detail=str(exc)[:200],
        )

    try:
        await openai_client.models.retrieve(settings.QUERY_REWRITING_MODEL)
        checks["query_rewriting_model"] = component(
            HEALTHY,
            model=settings.QUERY_REWRITING_MODEL,
        )
    except Exception as exc:
        checks["query_rewriting_model"] = component(
            UNHEALTHY,
            model=settings.QUERY_REWRITING_MODEL,
            detail=str(exc)[:200],
        )

    return checks


def probe_failure_response(
    message: str,
    detail: Optional[str] = None,
    status_code: int = 503,
) -> JSONResponse:
    content = {
        "status": UNHEALTHY,
        "service": settings.SERVICE_NAME,
        "environment": settings.SERVICE_ENVIRONMENT,
        "timestamp": utc_timestamp(),
        "message": message,
    }
    if detail:
        content["detail"] = detail[:200]
    return JSONResponse(status_code=status_code, content=content)


async def run_generation_probe(request: Request) -> JSONResponse:
    checks = {
        "query_rewriting": component(UNHEALTHY, detail="Probe not started"),
        "retrieval": component(UNHEALTHY, detail="Probe not started"),
        "reranking": component(UNHEALTHY, detail="Probe not started"),
        "generation": component(UNHEALTHY, detail="Probe not started"),
    }

    def finalize_probe() -> JSONResponse:
        payload = build_health_payload(
            service=settings.SERVICE_NAME,
            environment=settings.SERVICE_ENVIRONMENT,
            checks=checks,
            release=build_release_metadata(),
            operations=build_operations_metadata(),
        )
        return JSONResponse(
            status_code=health_status_code(payload["status"]),
            content=payload,
        )

    try:
        agent_result = await query_rewriting_agent(
            settings.GENERATION_PROBE_QUESTION,
            "English",
            [],
        )
    except Exception as exc:
        logger.exception("Generation probe query rewriting failed:")
        checks["query_rewriting"] = component(
            UNHEALTHY,
            model=settings.QUERY_REWRITING_MODEL,
            detail=str(exc)[:200],
        )
        checks["retrieval"] = component(
            UNHEALTHY,
            detail="Skipped because query rewriting failed",
        )
        checks["reranking"] = component(
            UNHEALTHY,
            detail="Skipped because query rewriting failed",
        )
        checks["generation"] = component(
            UNHEALTHY,
            model=settings.MAIN_MODEL,
            detail="Skipped because query rewriting failed",
        )
        return finalize_probe()

    if agent_result.get("action") != "rewrite":
        checks["query_rewriting"] = component(
            UNHEALTHY,
            model=settings.QUERY_REWRITING_MODEL,
            action=agent_result.get("action"),
            detail="Probe query did not use the retrieval path",
        )
        checks["retrieval"] = component(
            UNHEALTHY,
            detail="Skipped because query rewriting did not return a retrieval query",
        )
        checks["reranking"] = component(
            UNHEALTHY,
            detail="Skipped because retrieval was not attempted",
        )
        checks["generation"] = component(
            UNHEALTHY,
            model=settings.MAIN_MODEL,
            detail="Skipped because retrieval was not attempted",
        )
        return finalize_probe()

    rewritten_queries = agent_result.get(
        "rewritten_queries",
        [agent_result.get("rewritten_query", settings.GENERATION_PROBE_QUESTION)],
    )
    rewritten_queries = [query for query in rewritten_queries if query]
    if not rewritten_queries:
        checks["query_rewriting"] = component(
            UNHEALTHY,
            model=settings.QUERY_REWRITING_MODEL,
            detail="Query rewriting returned no rewritten query",
        )
        checks["retrieval"] = component(
            UNHEALTHY,
            detail="Skipped because query rewriting returned no rewritten query",
        )
        checks["reranking"] = component(
            UNHEALTHY,
            detail="Skipped because retrieval was not attempted",
        )
        checks["generation"] = component(
            UNHEALTHY,
            model=settings.MAIN_MODEL,
            detail="Skipped because retrieval was not attempted",
        )
        return finalize_probe()

    checks["query_rewriting"] = component(
        HEALTHY,
        model=settings.QUERY_REWRITING_MODEL,
        query_count=len(rewritten_queries),
    )

    def retrieve_for_query(query: str):
        return request.app.state.retriever.invoke(query)

    try:
        results_list = await asyncio.gather(
            *[asyncio.to_thread(retrieve_for_query, query) for query in rewritten_queries]
        )
    except Exception as exc:
        logger.exception("Generation probe retrieval failed:")
        checks["retrieval"] = component(
            UNHEALTHY,
            detail=str(exc)[:200],
        )
        checks["reranking"] = component(
            UNHEALTHY,
            model=settings.RERANK_MODEL,
            detail="Skipped because retrieval failed",
        )
        checks["generation"] = component(
            UNHEALTHY,
            model=settings.MAIN_MODEL,
            detail="Skipped because retrieval failed",
        )
        return finalize_probe()

    unique_docs = {}
    for docs in results_list:
        for doc in docs:
            unique_docs[doc.page_content] = doc

    retrieved_docs = list(unique_docs.values())
    docs = [
        {
            "summary": doc.metadata.get("summary", ""),
            "chunk": doc.page_content,
            "page_source": doc.metadata.get("page_source", doc.metadata.get("source", "")),
        }
        for doc in retrieved_docs
    ]

    if not docs:
        checks["retrieval"] = component(
            UNHEALTHY,
            detail="Probe query returned no documents",
        )
        checks["reranking"] = component(
            UNHEALTHY,
            model=settings.RERANK_MODEL,
            detail="Skipped because retrieval returned no documents",
        )
        checks["generation"] = component(
            UNHEALTHY,
            model=settings.MAIN_MODEL,
            detail="Skipped because retrieval returned no documents",
        )
        return finalize_probe()

    checks["retrieval"] = component(
        HEALTHY,
        document_count=len(docs),
        embedding_model=settings.EMBEDDING_MODEL,
    )

    primary_query = rewritten_queries[0]
    try:
        ranked_docs = await asyncio.to_thread(
            rerank_docs,
            primary_query,
            docs,
            request.app.state.pc,
        )
    except Exception as exc:
        logger.exception("Generation probe reranking failed:")
        checks["reranking"] = component(
            UNHEALTHY,
            model=settings.RERANK_MODEL,
            detail=str(exc)[:200],
        )
        checks["generation"] = component(
            UNHEALTHY,
            model=settings.MAIN_MODEL,
            detail="Skipped because reranking failed",
        )
        return finalize_probe()

    if not ranked_docs:
        checks["reranking"] = component(
            UNHEALTHY,
            model=settings.RERANK_MODEL,
            detail="Reranker returned no ranked documents",
        )
        checks["generation"] = component(
            UNHEALTHY,
            model=settings.MAIN_MODEL,
            detail="Skipped because reranking returned no documents",
        )
        return finalize_probe()

    checks["reranking"] = component(
        HEALTHY,
        model=settings.RERANK_MODEL,
        document_count=len(ranked_docs),
    )

    try:
        completion = await openai_client.chat.completions.create(
            model=settings.MAIN_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a synthetic health probe. Reply with exactly OK.",
                },
                {
                    "role": "user",
                    "content": format_query(
                        settings.GENERATION_PROBE_QUESTION,
                        "English",
                        ranked_docs,
                    ),
                },
            ],
            temperature=0,
            max_completion_tokens=8,
        )
        reply = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.exception("Generation probe model invocation failed:")
        checks["generation"] = component(
            UNHEALTHY,
            model=settings.MAIN_MODEL,
            detail=str(exc)[:200],
        )
        return finalize_probe()

    if not reply:
        checks["generation"] = component(
            UNHEALTHY,
            model=settings.MAIN_MODEL,
            detail="Probe generation returned an empty response",
        )
        return finalize_probe()

    lowered_reply = reply.lower()
    if any(
        failure_phrase in lowered_reply
        for failure_phrase in (
            "response generation failed",
            "out of my scope",
            "i apologize",
        )
    ):
        checks["generation"] = component(
            UNHEALTHY,
            model=settings.MAIN_MODEL,
            detail=f"Probe generation returned a fallback response: {reply[:120]}",
        )
        return finalize_probe()

    checks["generation"] = component(
        HEALTHY,
        model=settings.MAIN_MODEL,
        reply=reply,
    )
    return finalize_probe()

# ------------------------------------------------------------------------------
# WebSocket endpoint for chat functionality with improved error handling
# ------------------------------------------------------------------------------
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket, token: str = None, chat_id: str = None, interaction_id: str = None):
    await websocket.accept()
    logger.info(f"WebSocket connected. chat_id: {chat_id}, token: {token[:10] if token else 'None'}")
    
    # 1. Authenticate User
    try:
        if not token:
            await websocket.close(code=4001, reason="Missing authentication token")
            return
        verify_token(token)
        validation_payload = await validate_websocket_session(token, chat_id)
        if validation_payload is None:
            await websocket.close(code=4003, reason="Unable to validate session")
            return
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        await websocket.close(code=4001, reason="Authentication failed")
        return

    # 2. Fetch History (Context Hydration)
    history_context = []
    if chat_id:
        history_context = await fetch_chat_history(chat_id, token)

    active_interaction_id = interaction_id
    question = None

    try:
        # Receive and validate the request
        try:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=WEBSOCKET_TIMEOUT)
            chat_request = ChatRequest(**data)
        except ValidationError as ve:
            logger.exception("Validation error:")
            await safe_send(websocket, {"response": "Invalid request format.", "sources": []})
            return
        except Exception as e:
            logger.exception("Error receiving data:")
            await safe_send(websocket, {"response": "Error receiving request data.", "sources": []})
            return


        question = chat_request.question
        language = chat_request.language
        response_detail_level = chat_request.response_detail_level
        active_interaction_id = chat_request.interaction_id or interaction_id
        # previous_chats = chat_request.previous_chats # Deprecated: use history_context
        previous_chats = history_context + (chat_request.previous_chats or [])

        # 3. Create/Validate Chat and Interaction
        if not chat_id:
            chat_id = await create_chat(token, title=question[:50], language=language)
            if not chat_id:
                await safe_send(websocket, {"response": "Failed to create chat session.", "sources": []})
                return
            # Send the new chat_id back to the client immediately
            await safe_send(websocket, {"chat_id": chat_id})

        if not active_interaction_id:
            active_interaction_id = await create_interaction(token, chat_id, question)
            if not active_interaction_id:
                await safe_send(websocket, {"response": "Failed to create interaction.", "sources": []})
                return
            # Send the new interaction_id back to the client
            await safe_send(websocket, {"interaction_id": active_interaction_id})

        await emit_status_update(websocket, active_interaction_id, token, "processing")
        
        # Emit first processing step
        await emit_processing_step(websocket, "analyzing", "Analyzing your question")

        # Apply query rewriting agent to analyze and possibly rewrite the query
        agent_result = await query_rewriting_agent(question, language, previous_chats)
        
        # Handle direct responses (out of scope or clarification requests)
        # Handle direct responses (out of scope or clarification requests)
        if agent_result["action"] in ["respond", "clarify"]:
            response_text = agent_result["response"]
            query_type = agent_result.get("query_type", "General Information")
            
            await safe_send(websocket, {"response": response_text, "sources": []})
            
            if chat_id:
                logger.info(f"Saving interaction for chat_id: {chat_id}")
                await save_interaction(chat_id, question, response_text, [], token, query_type=query_type)
                
            if active_interaction_id:
                await emit_status_update(websocket, active_interaction_id, token, "completed")
            return
            
        # Use the rewritten queries for retrieval
        queries_for_retrieval = agent_result.get("rewritten_queries", [agent_result.get("rewritten_query", question)])
        logger.info(f"Retrieving documents for {len(queries_for_retrieval)} queries: {queries_for_retrieval}")
        
        # Filter previous chat messages logic remains same...
        relevant_history = []
        if "relevant_history_indices" in agent_result and previous_chats:
             # ... (keep filtering logic - lines 176-198)
            indices = agent_result["relevant_history_indices"]
            indices_to_include = set()
            for idx in indices:
                if 0 <= idx < len(previous_chats):
                    indices_to_include.add(idx)
                    if idx + 1 < len(previous_chats) and previous_chats[idx]["role"] == "user" and previous_chats[idx + 1]["role"] == "assistant":
                        indices_to_include.add(idx + 1)
            sorted_indices = sorted(indices_to_include)
            relevant_history = [previous_chats[i] for i in sorted_indices]
            if len(relevant_history) < len(previous_chats):
                logger.info(f"Filtered message history from {len(previous_chats)} to {len(relevant_history)} relevant messages")
        else:
            relevant_history = []
        
        # Emit searching step
        await emit_processing_step(websocket, "searching", "Searching knowledge base")
        
        # Retrieve documents for ALL queries in parallel
        try:
            # Helper function for retrieval
            def retrieve_for_query(q):
                return websocket.app.state.retriever.invoke(q)

            # Run retrieval for each query
            results_list = await asyncio.gather(*[
                asyncio.to_thread(retrieve_for_query, q) for q in queries_for_retrieval
            ])
            
            # Combine and deduplicate documents by page_content
            unique_docs_map = {}
            for docs in results_list:
                for doc in docs:
                    # Use page_content as unique key
                    if doc.page_content not in unique_docs_map:
                        unique_docs_map[doc.page_content] = doc
            
            retrieved_docs = list(unique_docs_map.values())
            logger.info(f"Total unique documents retrieved: {len(retrieved_docs)}")

        except Exception as e:
            logger.exception("Document retrieval error:")
            await safe_send(websocket, {"response": "This question is out of my scope. Please try again with another question.", "sources": []})
            if active_interaction_id:
                await emit_status_update(
                    websocket,
                    active_interaction_id,
                    token,
                    "error",
                    error_message="Document retrieval failed.",
                )
            return

        docs = [{
            "summary": ele.metadata.get("summary", ""),
            "chunk": ele.page_content,
            "page_source": ele.metadata.get("page_source", ele.metadata.get("source", ""))
        } for ele in retrieved_docs]

        if not docs:
            response_text = "No information found to answer your question."
            await safe_send(websocket, {"response": response_text, "sources": []})
            if chat_id:
                asyncio.create_task(save_interaction(chat_id, question, response_text, [], token))
            if active_interaction_id:
                await emit_status_update(websocket, active_interaction_id, token, "completed")
            return

        # Emit processing step
        await emit_processing_step(websocket, "processing", "Processing information")
        
        # Rerank the documents (fallback to original docs if reranking fails)
        # Use the primary rewritten query for reranking
        primary_query = queries_for_retrieval[0] if queries_for_retrieval else question
        try:
            ranked_docs = await asyncio.to_thread(rerank_docs, primary_query, docs, websocket.app.state.pc)
        except Exception as e:
            logger.exception("Reranking error:")
            ranked_docs = docs


        # Prepare the conversation messages
        messages = [{"role": "system", "content": get_system_prompt(response_detail_level)}]
        messages.extend(relevant_history)  # Use only the relevant history
        messages.append({"role": "user", "content": format_query(question, language, ranked_docs)})

        complete_answer = ""
        chunk_buffer = ""
        isResponseAvailable = True
        
        logger.debug("Starting OpenAI stream...")
        
        # Emit generating step
        await emit_processing_step(websocket, "generating", "Generating response")

        # Generate and stream the chat response
        streaming_started = False
        # Use dynamic token limit based on user's response detail level preference
        max_tokens = get_max_tokens_for_detail_level(response_detail_level)
        try:
            completion = await openai_client.chat.completions.create(
                model=MAIN_MODEL,
                messages=messages,
                temperature=MAIN_MODEL_TEMPERATURE,
                max_completion_tokens=max_tokens,
                stream=True
            )
            async for chunk in completion:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    if "🛑" in delta_content:
                        isResponseAvailable = False
                        break
                    complete_answer += delta_content
                    if not streaming_started and active_interaction_id:
                        streaming_started = True
                        await emit_status_update(websocket, active_interaction_id, token, "streaming")
                    # Don't remove inline citation markers - pass them through to frontend
                    # cleaned_content = re.sub(r'\[\d+\]', '', delta_content)
                    chunk_buffer += delta_content
                    if len(chunk_buffer) >= CHUNK_BUFFER_SIZE:
                        await safe_send(websocket, {"response": chunk_buffer})
                        chunk_buffer = ""
            if chunk_buffer:
                await safe_send(websocket, {"response": chunk_buffer})
        except Exception as e:
            logger.exception("Error during streaming response:")
            await safe_send(websocket, {"response": "Response generation failed. Please try again later.", "sources": []})
            if active_interaction_id:
                await emit_status_update(
                    websocket,
                    active_interaction_id,
                    token,
                    "error",
                    error_message="Generation failed.",
                )
                if chat_id:
                    logger.info("Saving error response to persistence.")
                    await save_interaction(
                        chat_id=chat_id,
                        user_message=question, 
                        assistant_message="Response generation failed. Please try again later.",
                        sources=[],
                        token=token,
                        query_type="Error",
                        interaction_id=active_interaction_id
                    )
            return

        logger.debug(f"Streaming finished. isResponseAvailable: {isResponseAvailable}, answer length: {len(complete_answer)}")

        # If the response indicates no answer available, inform the user
        if not isResponseAvailable:
            logger.info("Response not available (out of scope or no info). Sending apology and saving interaction.")
            response_text = "I apologize, but I don't have sufficient information in my knowledge base to provide a complete answer to your question. Please try rephrasing your question or providing more specific details about what you're looking for."
            query_type = agent_result.get("query_type", "General Information") # Default if not set
            await safe_send(websocket, {
                "response": response_text,
                "sources": []
            })
            if chat_id:
                logger.info(f"Saving interaction for chat_id: {chat_id}")
                await save_interaction(chat_id, question, response_text, [], token, query_type=query_type, interaction_id=active_interaction_id)
            if active_interaction_id:
                await emit_status_update(websocket, active_interaction_id, token, "completed")
            return

        # Emit finalizing step
        await emit_processing_step(websocket, "finalizing", "Adding citations")
        
        # Process and map citations in the final answer
        try:

            updated_answer, citations = process_citations(complete_answer, ranked_docs)

        except Exception as e:
            logger.exception("Error processing citations:")
            updated_answer, citations = complete_answer, []

        # 3. Async Persistence (Moved before final send to ensure it runs even if client disconnects)
        logger.debug(f"Checking persistence. chat_id: {chat_id}")
        if chat_id:
            logger.info(f"Initiating background persistence for chat_id: {chat_id}")
            
            query_type = agent_result.get("query_type", "General Information")
            
            asyncio.create_task(save_interaction(
                chat_id=chat_id,
                user_message=question,
                assistant_message=updated_answer,
                sources=citations,
                token=token,
                query_type=query_type,
                interaction_id=active_interaction_id
            ))
        else:
            logger.warning("No chat_id provided, skipping persistence")

        await safe_send(websocket, {
            "response": updated_answer,
            "replace": True,
            "sources": citations
        })
        if active_interaction_id:
            await emit_status_update(websocket, active_interaction_id, token, "completed")

    except WebSocketDisconnect:
        logger.debug("WebSocketDisconnect caught")
        logger.info("Client disconnected")
        if chat_id and question:
            logger.info(f"Saving interrupted interaction for chat_id: {chat_id}")
            # Save whatever response we have, marking it as interrupted
            response_to_save = complete_answer if complete_answer else "[Connection interrupted before response could be generated]"
            await save_interaction(
                chat_id, 
                question, 
                response_to_save, 
                [], 
                token,
                query_type="Other",
                interaction_id=active_interaction_id
            )
            # Update status to interrupted
            if active_interaction_id:
                await update_interaction_status(active_interaction_id, "interrupted", token)
    except Exception as e:
        logger.exception("Unexpected error in websocket endpoint:")
        try:
            await safe_send(websocket, {"response": "An unexpected error occurred. Please try again later.", "sources": []})
        except Exception:
            pass  # The connection may already be closed.
        if active_interaction_id:
            await emit_status_update(
                websocket,
                active_interaction_id,
                token,
                "error",
                error_message="Unexpected error in websocket endpoint.",
            )

# ------------------------------------------------------------------------------
# HTTP endpoint for Telegram chat
# ------------------------------------------------------------------------------
@app.post("/telegram-chat")
async def telegram_chat(chat_request: ChatRequest, request: Request):
    # Extract the question and language from the validated request body.
    logger.info(f"Received telegram chat request: {chat_request}")
    is_probe_request = is_health_probe(request)
    
    question = chat_request.question
    language = chat_request.language
    response_detail_level = chat_request.response_detail_level
    previous_chats = chat_request.previous_chats

    # Apply query rewriting agent to analyze and possibly rewrite the query
    agent_result = await query_rewriting_agent(question, language, previous_chats)
    
    # Handle direct responses (out of scope or clarification requests)
    if agent_result["action"] in ["respond", "clarify"]:
        if is_probe_request:
            return probe_failure_response(
                "Synthetic probe was handled as a non-answer path",
                agent_result.get("response"),
            )
        return {
            "response": agent_result["response"],
            "sources": []
        }
        
    # Use the rewritten query for retrieval if available
    query_for_retrieval = agent_result.get("rewritten_query", question)
    
    # Filter previous chat messages based on relevance
    relevant_history = []
    if "relevant_history_indices" in agent_result and previous_chats:
        indices = agent_result["relevant_history_indices"]
        
        # Create a set to track which indices to include (including assistants' responses)
        indices_to_include = set()
        
        # Include each relevant message index
        for idx in indices:
            if 0 <= idx < len(previous_chats):
                indices_to_include.add(idx)
                # If this is a user message and there's an assistant response right after,
                # include the assistant's response too
                if idx + 1 < len(previous_chats) and previous_chats[idx]["role"] == "user" and previous_chats[idx + 1]["role"] == "assistant":
                    indices_to_include.add(idx + 1)
        
        # Sort the indices to maintain conversation order
        sorted_indices = sorted(indices_to_include)
        
        # Get relevant messages in order
        relevant_history = [previous_chats[i] for i in sorted_indices]
        
        # Log the filtering of message history
        if len(relevant_history) < len(previous_chats):
            logger.info(f"Filtered message history from {len(previous_chats)} to {len(relevant_history)} relevant messages")
    else:
        # If no relevance info or no previous chats, use empty history
        relevant_history = []
    
    # Retrieve documents using the retriever.
    try:
        retrieved_docs = await asyncio.to_thread(request.app.state.retriever.invoke, query_for_retrieval)
    except Exception as e:
        logger.exception("Document retrieval error:")
        if is_probe_request:
            return probe_failure_response("Document retrieval failed during probe", str(e))
        return {
            "response": "This question is out of my scope. Please try again with another question.",
            "sources": []
        }

    docs = [{
        "summary": ele.metadata.get("summary", ""),
        "chunk": ele.page_content,
        "page_source": ele.metadata.get("page_source", ele.metadata.get("source", ""))
    } for ele in retrieved_docs]

    if not docs:
        if is_probe_request:
            return probe_failure_response("Synthetic probe retrieved no documents")
        return {
            "response": "No information found to answer your question.",
            "sources": []
        }

    # Rerank the documents (fallback to original docs if reranking fails)
    try:
        ranked_docs = await asyncio.to_thread(rerank_docs, query_for_retrieval, docs, request.app.state.pc)
    except Exception as e:
        logger.exception("Reranking error:")
        ranked_docs = docs

    # Prepare the conversation messages.
    messages = [{"role": "system", "content": get_system_prompt(response_detail_level)}]
    messages.extend(relevant_history)  # Use only the relevant history
    messages.append({"role": "user", "content": format_query(question, language, ranked_docs)})

    complete_answer = ""
    isResponseAvailable = True

    # Generate and stream the chat response.
    try:
        completion = await openai_client.chat.completions.create(
            model=MAIN_MODEL,
            messages=messages,
            temperature=MAIN_MODEL_TEMPERATURE,
            max_completion_tokens=get_max_tokens_for_detail_level(response_detail_level),
            stream=True
        )
        async for chunk in completion:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                if "🛑" in delta_content:
                    isResponseAvailable = False
                    break
                complete_answer += delta_content
                # Remove inline citation markers from the streamed chunk.
                cleaned_content = re.sub(r'\[\d+\]', '', delta_content)
    except Exception as e:
        logger.exception("Error during streaming response:")
        if is_probe_request:
            return probe_failure_response("Response generation failed during probe", str(e))
        return {
            "response": "Response generation failed. Please try again later.",
            "sources": []
        }

    # If the initial response indicates no answer, inform the user
    if not isResponseAvailable:
        if is_probe_request:
            return probe_failure_response("Synthetic probe did not yield a confident answer")
        return {
            "response": "I apologize, but I don't have sufficient information in my knowledge base to provide a complete answer to your question. Please try rephrasing your question or providing more specific details about what you're looking for.",
            "sources": []
        }

    # Process and map citations in the final answer.
    try:
        updated_answer, citations = process_citations(complete_answer, ranked_docs)
    except Exception as e:
        logger.exception("Error processing citations:")
        updated_answer, citations = complete_answer, []

    return {"response": updated_answer, "sources": citations}

# ------------------------------------------------------------------------------
# Simple health check endpoint
# ------------------------------------------------------------------------------
@app.get("/", response_class=JSONResponse)
async def root():
    return JSONResponse(content={"status": "working"})

@app.get("/api", response_class=JSONResponse)
async def api_root():
    return JSONResponse(content={"message": "API is working"})


@app.get("/demo/growth-dashboard", response_class=HTMLResponse)
async def growth_dashboard():
    return HTMLResponse(content=build_growth_dashboard_page())

@app.get("/health")
async def health(request: Request):
    checks = basic_health_checks(request)
    payload = build_health_payload(
        service=settings.SERVICE_NAME,
        environment=settings.SERVICE_ENVIRONMENT,
        checks=checks,
        release=build_release_metadata(),
        operations=build_operations_metadata(),
    )
    return JSONResponse(
        status_code=health_status_code(payload["status"]),
        content=payload,
    )


@app.get("/health/detailed")
async def health_detailed(request: Request):
    checks = await detailed_health_checks(request)
    payload = build_health_payload(
        service=settings.SERVICE_NAME,
        environment=settings.SERVICE_ENVIRONMENT,
        checks=checks,
        release=build_release_metadata(),
        operations=build_operations_metadata(),
    )
    return JSONResponse(
        status_code=health_status_code(payload["status"]),
        content=payload,
    )


@app.get("/health/generation")
async def health_generation(request: Request):
    return await run_generation_probe(request)
