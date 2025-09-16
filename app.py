import nltk
nltk.download("punkt")


import asyncio
import re

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# Import modules
from modules.config import (
    logger, validate_env_vars, system_prompt, 
    ENABLE_CLARIFICATION, MAX_CLARIFICATION_ATTEMPTS,
    MAIN_MODEL, FALLBACK_MODEL, MAX_COMPLETION_TOKENS,
    MAIN_MODEL_TEMPERATURE, FALLBACK_MODEL_TEMPERATURE,
    WEBSOCKET_TIMEOUT, CHUNK_BUFFER_SIZE, HOST, PORT, CORS_ORIGINS
)
from modules.schemas import ChatRequest, CitationSource
from modules.utils import safe_send, format_query
from modules.citations import process_citations
from modules.retrieval import initialize_pinecone, rerank_docs
from modules.query_rewriting import query_rewriting_agent, openai_client, handle_clarification_response

# ------------------------------------------------------------------------------
# Initialize application and validate environment
# ------------------------------------------------------------------------------
validate_env_vars()

# ------------------------------------------------------------------------------
# Initialize FastAPI app with CORS middleware (restrict origins in production)
# ------------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Initialize Pinecone retriever and client
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    app.state.retriever, app.state.pc = initialize_pinecone()

# ------------------------------------------------------------------------------
# WebSocket endpoint for chat functionality with improved error handling
# ------------------------------------------------------------------------------
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
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
        previous_chats = chat_request.previous_chats

        # Apply query rewriting agent to analyze and possibly rewrite the query
        agent_result = await query_rewriting_agent(question, language, previous_chats)
        
        # Handle direct responses (out of scope or clarification requests)
        if agent_result["action"] in ["respond", "clarify"]:
            await safe_send(websocket, {"response": agent_result["response"], "sources": []})
            return
            
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
        
        # Retrieve documents using the retriever
        try:
            retrieved_docs = await asyncio.to_thread(websocket.app.state.retriever.invoke, query_for_retrieval)
        except Exception as e:
            logger.exception("Document retrieval error:")
            await safe_send(websocket, {"response": "This question is out of my scope. Please try again with another question.", "sources": []})
            return

        docs = [{
            "summary": ele.metadata.get("summary", ""),
            "chunk": ele.page_content,
            "page_source": ele.metadata.get("page_source", ele.metadata.get("source", ""))
        } for ele in retrieved_docs]

        if not docs:
            await safe_send(websocket, {"response": "No information found to answer your question.", "sources": []})
            return

        # Rerank the documents (fallback to original docs if reranking fails)
        try:
            ranked_docs = await asyncio.to_thread(rerank_docs, query_for_retrieval, docs, websocket.app.state.pc)
        except Exception as e:
            logger.exception("Reranking error:")
            ranked_docs = docs


        # Prepare the conversation messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(relevant_history)  # Use only the relevant history
        messages.append({"role": "user", "content": format_query(question, language, ranked_docs)})

        complete_answer = ""
        chunk_buffer = ""
        isResponseAvailable = True

        # Generate and stream the chat response
        try:
            completion = await openai_client.chat.completions.create(
                model=MAIN_MODEL,
                messages=messages,
                temperature=MAIN_MODEL_TEMPERATURE,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                stream=True
            )
            async for chunk in completion:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    if "🛑" in delta_content:
                        isResponseAvailable = False
                        break
                    complete_answer += delta_content
                    # Remove inline citation markers from the streamed chunk before sending
                    cleaned_content = re.sub(r'\[\d+\]', '', delta_content)
                    chunk_buffer += cleaned_content
                    if len(chunk_buffer) >= CHUNK_BUFFER_SIZE:
                        await safe_send(websocket, {"response": chunk_buffer})
                        chunk_buffer = ""
            if chunk_buffer:
                await safe_send(websocket, {"response": chunk_buffer})
        except Exception as e:
            logger.exception("Error during streaming response:")
            await safe_send(websocket, {"response": "Response generation failed. Please try again later.", "sources": []})
            return

        # If the response indicates no answer available, inform the user
        if not isResponseAvailable:
            await safe_send(websocket, {
                "response": "I apologize, but I don't have sufficient information in my knowledge base to provide a complete answer to your question. Please try rephrasing your question or providing more specific details about what you're looking for.",
                "sources": []
            })
            return

        # Process and map citations in the final answer
        try:
            updated_answer, citations = process_citations(complete_answer, ranked_docs)
        except Exception as e:
            logger.exception("Error processing citations:")
            updated_answer, citations = complete_answer, []

        await safe_send(websocket, {
            "response": updated_answer,
            "sources": citations
        })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("Unexpected error in websocket endpoint:")
        try:
            await safe_send(websocket, {"response": "An unexpected error occurred. Please try again later.", "sources": []})
        except Exception:
            pass  # The connection may already be closed.

# ------------------------------------------------------------------------------
# HTTP endpoint for Telegram chat
# ------------------------------------------------------------------------------
@app.post("/telegram-chat")
async def telegram_chat(chat_request: ChatRequest, request: Request):
    # Extract the question and language from the validated request body.
    logger.info(f"Received telegram chat request: {chat_request}")
    
    question = chat_request.question
    language = chat_request.language
    previous_chats = chat_request.previous_chats

    # Apply query rewriting agent to analyze and possibly rewrite the query
    agent_result = await query_rewriting_agent(question, language, previous_chats)
    
    # Handle direct responses (out of scope or clarification requests)
    if agent_result["action"] in ["respond", "clarify"]:
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
        return {
            "response": "This question is out of my scope. Please try again with another question.",
            "sources": []
        }

    docs = [{
        "summary": ele.metadata.get("summary", ""),
        "chunk": ele.page_content,
        "page_source": ele.metadata.get("source", "")
    } for ele in retrieved_docs]

    if not docs:
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
    messages = [{"role": "system", "content": system_prompt}]
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
            max_completion_tokens=MAX_COMPLETION_TOKENS,
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
        return {
            "response": "Response generation failed. Please try again later.",
            "sources": []
        }

    # If the initial response indicates no answer, inform the user
    if not isResponseAvailable:
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

@app.get("/health")
async def health():
    return JSONResponse(content={"message": "working"})