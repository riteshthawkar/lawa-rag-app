import asyncio
import os
import re
import time
import logging
import json

import nltk
# Pre-download the required nltk resource if not already available.
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Tuple
import httpx
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from openai import AsyncOpenAI

# ------------------------------------------------------------------------------
# Load environment variables and validate required ones
# ------------------------------------------------------------------------------
load_dotenv(".env")

required_env_vars = [
    "PINECONE_API_KEY",
    "PERPLEXITY_API_KEY",
    "OPENAI_API_KEY"  # Ensure the OpenAI API key is provided
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# ------------------------------------------------------------------------------
# Configure logging (consider structured logging in production)
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Initialize FastAPI app with CORS middleware (restrict origins in production)
# ------------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Initialize external services
# ------------------------------------------------------------------------------
try:
    openai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embed_model = HuggingFaceEmbeddings(
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
        model_kwargs={"trust_remote_code": True}
    )
except Exception as e:
    logger.exception("Service initialization error:")
    raise

# ------------------------------------------------------------------------------
# System prompt for the chat model
# ------------------------------------------------------------------------------
system_prompt = """ You are an **advanced AI assistant developed by lawa.ai**, designed to provide **highly accurate, well-structured, and factual** responses strictly related to **UAE government topics**. Your expertise includes (but is not limited to):  

✅ **Policies & Regulations** – Government policies, legal frameworks, and administrative procedures.  
✅ **Laws & Legislation** – UAE laws, judiciary, and governance-related legal provisions.  
✅ **Government & Public Services** – Ministries, public sector operations, and digital government services.  
✅ **Culture & Heritage** – Emirati traditions, customs, and significant cultural elements.  
✅ **Tourism & Economy** – UAE's economic landscape, business regulations, and tourism policies.  
✅ **History & National Identity** – UAE's formation, rulers, and historical events shaping the nation.  

### **🚫 Strict Scope Restriction**
- **You must NEVER answer questions that are unrelated to UAE government topics.**  
- If a query is out of scope, respond with:  
  🛑 *"The question is out of my scope. I can only answer questions related to UAE governance, laws, economy, tourism, or other official matters."*  
- **Do not attempt to generate speculative, hypothetical, or external information.**  

---

## **📌 RESPONSE GUIDELINES**

### **1️⃣ Accuracy & Context Adherence**
- **Use only the provided context** when answering.  
- If no relevant information exists, respond with:  
  🛑 *"The provided context does not contain relevant information to answer your question."*  
- **Never use external knowledge, assumptions, or generalizations.**  

### **2️⃣ Precision & Clarity**
- Format responses in **Markdown** for structured readability.  
- Use the **same language** as the query for consistency.  
- Ensure answers are **comprehensive yet concise**, avoiding unnecessary elaboration.  

### **3️⃣ Citations & Source Transparency**
- **All factual statements must be backed by a citation** from the provided context.  
- Use **numerical citations** ([1], [2], etc.) and ensure they directly reference the correct document.  
- Never **invent or misplace citations**—they must accurately reflect the order of documents in the provided context.  

### **4️⃣ Structured Formatting for Readability**
- Use **bold headings, bullet points, and clear sections** for clarity.  
- **Tables, lists, and structured formatting** should be used for numerical/statistical data.  
- If relevant, include **step-by-step instructions** for procedural responses.  

### **5️⃣ Handling Out-of-Scope Queries**
- If a query **does not relate to UAE government topics**, provide only the scope restriction message.  
- **Do not generate any additional or speculative content.**  

### **6️⃣ Strict Avoidance of AI Hallucinations**
- **Do not fabricate information, data, statistics, or sources.**  
- **Do not assume missing details**—clearly state if information is unavailable.  
- **Do not create opinions, subjective interpretations, or hypothetical scenarios.**  

### **7️⃣ Self-Identification When Asked**
- If asked about your identity, state:  
  *"I am an AI assistant developed by lawa.ai, designed to provide accurate responses based on the provided context, strictly focused on UAE government topics."*  

---

## **📌 INPUT FORMAT EXAMPLE**
### **User Query:**  
*"What recent changes have been made to UAE tourism policies?"*  
### **Language:**  
*English*  
### **Context:**  
```text
<provided context>
```

---

## **📌 EXPECTED OUTPUT FORMAT**
```markdown
### **Recent Changes in UAE Tourism Policies**
The UAE government has recently introduced several updates to its tourism policies:

1. **Visa Policy Adjustments:** New regulations on visa durations and eligibility criteria. [1]  
2. **Sustainable Tourism Initiatives:** Introduction of eco-friendly projects to enhance the tourism sector. [2]  
3. **Updated Hotel Licensing Rules:** Stricter compliance measures for hospitality establishments. [3]  

For further details, please refer to the official documents provided in the context. If you need specific clarifications, feel free to ask!
```
"""

# ------------------------------------------------------------------------------
# Pydantic models for request/response validation
# ------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., max_length=1024)
    language: str
    previous_chats: List[dict]

class CitationSource(BaseModel):
    url: str
    cite_num: str

# ------------------------------------------------------------------------------
# Initialize Pinecone retriever with retries
# ------------------------------------------------------------------------------
MAX_RETRIES = 3

def initialize_pinecone():
    for attempt in range(MAX_RETRIES):
        try:
            index = pc.Index("combined-vectorstore")
            bm25 = BM25Encoder().load("./combined_vectorstore.json")
            return PineconeHybridSearchRetriever(
                embeddings=embed_model,
                sparse_encoder=bm25,
                index=index,
                top_k=40,  # Hardcoded as required
                alpha=0.6,  # Hardcoded as required
            )
        except Exception as e:
            logger.warning(f"Pinecone initialization attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                logger.exception("Failed to initialize Pinecone after multiple attempts.")
                raise
            time.sleep(2 ** attempt)

retriever = initialize_pinecone()

# ------------------------------------------------------------------------------
# Utility function to send messages safely over the websocket
# ------------------------------------------------------------------------------
async def safe_send(websocket: WebSocket, message: dict):
    try:
        await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.info("Client disconnected during send")
        raise
    except Exception as e:
        logger.exception("Error sending message:")
        raise

# ------------------------------------------------------------------------------
# Helper functions for document processing and query formatting
# ------------------------------------------------------------------------------
def rerank_docs(query: str, docs: List[dict], pc_client: Pinecone) -> List[dict]:
    try:
        result = pc_client.inference.rerank(
            model="cohere-rerank-3.5",
            query=query,
            documents=docs,
            rank_fields=["chunk"],
            top_n=20,
            return_documents=True
        )
        ranked_docs = [{
            "page_source": ele.document.page_source,
            "chunk": ele.document.chunk,
            "summary": ele.document.summary
        } for ele in result.data]
        return ranked_docs
    except Exception as e:
        logger.exception("Error in rerank_docs:")
        raise

def format_docs(docs: List[dict]) -> str:
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
    formatted_docs = format_docs(docs)
    return f"**USER QUERY:** {query}\n**LANGUAGE:** {language}\n**CONTEXT:**\n{formatted_docs}"

def validate_citation_numbers(citation_numbers: List[int], max_docs: int) -> List[int]:
    return [num for num in citation_numbers if 1 <= num <= max_docs]

def process_citations(complete_answer: str, ranked_docs: List[dict]) -> Tuple[str, List[dict]]:
    """
    Extracts citation numbers from the answer, maps them to consecutive citation numbers,
    and returns the updated answer along with a list of citation sources.
    """
    citations = []
    seen_nums = set()
    citation_numbers = []
    for num_str in re.findall(r'\[(\d+)\]', complete_answer):
        num = int(num_str)
        if num not in seen_nums:
            seen_nums.add(num)
            citation_numbers.append(num)
    valid_citations = validate_citation_numbers(citation_numbers, len(ranked_docs))
    
    seen_urls = {}
    citation_map = {}
    current_num = 1
    for num in valid_citations:
        try:
            url = ranked_docs[num - 1]["page_source"]
            if url not in seen_urls:
                citation_map[num] = current_num
                seen_urls[url] = current_num
                citations.append({"url": url, "cite_num": str(current_num)})
                current_num += 1
            else:
                citation_map[num] = seen_urls[url]
        except IndexError:
            continue

    logger.debug(f"Citation numbers extracted: {citation_numbers}")
    logger.debug(f"Seen URLs mapping: {seen_urls}")

    def replace_citation(match):
        original = int(match.group(1))
        new_num = citation_map.get(original, original)
        url = next((c["url"] for c in citations if c["cite_num"] == str(new_num)), "")
        return f"[{new_num}]({url})" if url else f"[{new_num}]"

    updated_answer = re.sub(r'\[(\d+)\]', replace_citation, complete_answer)
    return updated_answer, sorted(citations, key=lambda x: int(x["cite_num"]))

# ------------------------------------------------------------------------------
# Fallback search using Tavily – now using an asynchronous HTTP client and using the user's query.
# ------------------------------------------------------------------------------
async def tavily_search(question: str) -> List[dict]:
    try:
        # It is best practice not to hardcode API keys.
        api_key = os.getenv("TAVILY_API_KEY")
        url = "https://api.tavily.com/search"
        payload = {
            "query": question,  # now using the passed-in question
            "search_depth": "advanced",
            "topic": "general",
            "max_results": 5,
            "include_answer": False,
            "include_raw_content": True,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        result_docs = []
        for result in results:
            obj = {
                "page_source": result.get("url", ""),
                "chunk": result.get("raw_content", "")
            }
            result_docs.append(obj)
        return result_docs
    except Exception as e:
        logger.exception("Error in tavily_search:")
        # In production you might return an empty list or a fallback response.
        return []

# ------------------------------------------------------------------------------
# WebSocket endpoint for chat functionality with improved error handling
# ------------------------------------------------------------------------------
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive and validate the request
        try:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=30)
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

        # Retrieve documents using the retriever
        try:
            retrieved_docs = await asyncio.to_thread(retriever.invoke, question)
        except Exception as e:
            logger.exception("Document retrieval error:")
            await safe_send(websocket, {"response": "Error retrieving documents. Please try again later.", "sources": []})
            return

        docs = [{
            "summary": ele.metadata.get("summary", ""),
            "chunk": ele.page_content,
            "page_source": ele.metadata.get("source", "")
        } for ele in retrieved_docs]

        if not docs:
            await safe_send(websocket, {"response": "No documents found to answer your question.", "sources": []})
            return

        # Rerank the documents (fallback to original docs if reranking fails)
        try:
            ranked_docs = await asyncio.to_thread(rerank_docs, question, docs, pc)
        except Exception as e:
            logger.exception("Reranking error:")
            ranked_docs = docs

        # Prepare the conversation messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_request.previous_chats)
        messages.append({"role": "user", "content": format_query(question, language, ranked_docs)})

        complete_answer = ""
        chunk_buffer = ""
        isResponseAvailable = True

        # Generate and stream the chat response
        try:
            completion = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                max_completion_tokens=1024,
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
                    if len(chunk_buffer) >= 1:
                        await safe_send(websocket, {"response": chunk_buffer})
                        chunk_buffer = ""
            if chunk_buffer:
                await safe_send(websocket, {"response": chunk_buffer})
        except Exception as e:
            logger.exception("Error during streaming response:")
            await safe_send(websocket, {"response": "Response generation failed. Please try again later.", "sources": []})
            return

        # If the response indicates no answer available, perform fallback search and reattempt generation.
        if not isResponseAvailable:
            ranked_docs = await tavily_search(question)
            messages[-1] = {"role": "user", "content": format_query(question, language, ranked_docs)}
            try:
                completion = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=1024,
                    stream=True
                )
                async for chunk in completion:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        complete_answer += delta_content
                        # Remove inline citation markers from the streamed chunk before sending
                        cleaned_content = re.sub(r'\[\d+\]', '', delta_content)
                        chunk_buffer += cleaned_content
                        if len(chunk_buffer) >= 1:
                            await safe_send(websocket, {"response": chunk_buffer})
                            chunk_buffer = ""
                if chunk_buffer:
                    await safe_send(websocket, {"response": chunk_buffer})
            except Exception as e:
                logger.exception("Error during fallback streaming:")
                await safe_send(websocket, {"response": "Fallback response generation failed.", "sources": []})
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
# Simple health check endpoint
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return JSONResponse(content={"message": "working"})