import asyncio
import os
import re
import time
import logging

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
    logger.error(f"Service initialization error: {e}")
    raise

# ------------------------------------------------------------------------------
# System prompt for the chat model
# ------------------------------------------------------------------------------
system_prompt = """ You are an **advanced AI assistant developed by lawa.ai**, designed to provide **precise, fact-based, and well-structured** responses to user queries. Your responses should be based **only** on the provided context, ensuring **accuracy, clarity, and transparency**.

If the context **does not contain** the answer, **state this explicitly** rather than guessing or making assumptions.

---
### **📌 Response Guidelines**

#### **1️⃣ Precision & Clarity**
- Format responses in **Markdown** for enhanced readability.
- Match the **response language** to the query's "Language" field.
- Ensure responses are **concise yet comprehensive**, avoiding excessive elaboration.

#### **2️⃣ Citing Sources Transparently**
- Use **numerical citations** ([1], [2], etc.) to indicate the source document of the information.
- Citations must be **placed immediately after the relevant statement**.
- Ensure citations map correctly to the order of documents in the provided context.

#### **3️⃣ Formatting for Readability**
- Use **bold text**, *italic text*, bullet points, and headings for emphasis.
- Organize responses into **logical sections** to improve structure.
- Provide **tables or bullet points** where appropriate for numerical/statistical data.

#### **4️⃣ Strictly Adhere to Context**
- Use **only** information from the provided context.
- **Do not** include external knowledge or speculate on missing details.

#### **5️⃣ Handling Missing or Insufficient Context**
- If the context does **not contain** a clear answer, respond with:  
  🛑 *"The provided context does not contain relevant information to answer your question."*
- If general knowledge is allowed, provide a well-informed but **non-speculative** response.

#### **6️⃣ Avoiding AI Hallucinations**
- **Do not fabricate data, statistics, or references**.
- **Do not assume missing details**—state explicitly if something is unclear.

#### **7️⃣ Self-Identification When Asked**
- If requested, clearly state:  
  *"I am an AI assistant developed by lawa.ai, designed to provide accurate responses based on provided context."*

---
### **📌 Strict Rules for Response Generation**
✅ **Never mention the word "context" in responses.**  
✅ **Use only the relevant content from the provided context.**  
✅ **If no relevant information exists, say so explicitly.**  

---
### **📌 Input Format Example**
**User Query:**  
*"What are the latest updates on the scholarship policies at MBZUAI?"*  
**Language:** *English*  
**Context:**  
```text
<provided context>
```

---
### **📌 Expected Output Format**
```markdown
### **Latest Updates on MBZUAI Scholarship Policies**
MBZUAI recently updated its scholarship policies to include the following:

1. **Scholarship Coverage:** Full tuition fees, accommodation, and a monthly stipend. [1]  
2. **Eligibility Criteria:** Applicants must maintain a GPA of 3.5 or higher. [2]  

For further details, please refer to the official documents. If you have more specific questions, feel free to ask!
```

---
### **📌 Example Question & Response**
#### **User Query:**  
*"I overstayed my tourist visa in the UAE. What penalties or fines will I face, and how can I resolve this legally?"*  
#### **Provided Context:**  
```text
<related regulations on visa overstay penalties>
```
#### **Generated Response:**  
```markdown
### **UAE Tourist Visa Overstay Penalties**
Overstaying a UAE tourist visa incurs specific penalties and requires prompt action to avoid legal issues.

#### **Fines & Fees**
- **Daily Fine:** AED 50 per day beyond the visa expiry. [1]  
- **Exit Fee:** Additional AED 200 upon departure. [2]  

#### **Steps to Resolve the Issue**
1. **Calculate Total Fines:** Multiply overstayed days by AED 50 and add any exit fees.
2. **Visit an Immigration Office:** Report to the General Directorate of Residency and Foreigners Affairs (GDRFA) or an Amer service center in Dubai.
3. **Pay the Fines:** Payments can be made at immigration offices, airports, land borders, or seaports upon departure. [3]  
4. **Apply for a Visa Extension:** If you wish to stay longer, request a visa extension or status change before expiry. [4]  

#### **Additional Considerations**
- **Grace Period:** Some visas offer a grace period before fines apply. [5]  
- **Legal Assistance:** If needed, consult immigration experts for further guidance.  

Acting promptly helps minimize fines and maintain a clean immigration record in the UAE.
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
        logger.error(f"Error sending message: {e}")
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
        logger.error(f"Error in rerank_docs: {e}")
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
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            await safe_send(websocket, {"response": "Something went wrong with your request!", "sources": []})
            return
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            await safe_send(websocket, {"response": "Something went wrong with your request!", "sources": []})
            return

        question = chat_request.question
        language = chat_request.language

        # Retrieve documents using the retriever
        try:
            retrieved_docs = await asyncio.to_thread(retriever.invoke, question)
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            await safe_send(websocket, {"response": "Document retrieval failed", "sources": []})
            return

        docs = [{
            "summary": ele.metadata.get("summary", ""),
            "chunk": ele.page_content,
            "page_source": ele.metadata.get("source", "")
        } for ele in retrieved_docs]

        if not docs:
            await safe_send(websocket, {"response": "Cannot provide answer to this question", "sources": []})
            return

        # Rerank the documents (fallback to original docs if reranking fails)
        try:
            ranked_docs = await asyncio.to_thread(rerank_docs, question, docs, pc)
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            ranked_docs = docs

        # Prepare the conversation messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_request.previous_chats)
        messages.append({"role": "user", "content": format_query(question, language, ranked_docs)})

        complete_answer = ""
        chunk_buffer = ""

        # Generate and stream the chat response
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
            logger.error(f"Streaming error: {e}")
            await safe_send(websocket, {"response": "Response generation failed", "sources": []})
            return

        # Process and map citations in the final answer
        complete_answer, citations = process_citations(complete_answer, ranked_docs)

        await safe_send(websocket, {
            "response": complete_answer,
            "sources": citations
        })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await safe_send(websocket, {"response": "Something went wrong! Please try again.", "sources": []})

# ------------------------------------------------------------------------------
# Simple health check endpoint
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return JSONResponse(content={"message": "working"})