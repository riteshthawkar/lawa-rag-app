import nltk
import asyncio
import os
import re
import json
import time
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, constr
from typing import List, Optional
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import logging

# Ensure the required NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables from .env file
load_dotenv(".env")

# Validate required environment variables
required_env_vars = [
    "PINECONE_API_KEY", 
    "PERPLEXITY_API_KEY",
    "USER_AGENT",
    "OPENAI_API_KEY"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with CORS middleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize external services with retries
try:
    openai_client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embed_model = HuggingFaceEmbeddings(
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
        model_kwargs={"trust_remote_code": True}
    )
except Exception as e:
    logger.error(f"Service initialization error: {e}")
    raise

# System prompt (truncated for brevity)
system_prompt = """
You are an advanced AI assistant developed by lawa.ai, designed to answer questions with precision and thoroughness. Use the provided context to craft informative and detailed responses. If the answer is not in the context, state that you do not know.

**When responding, follow these guidelines:**

1. **Detailed and Clear Answers:**
   - Provide the response in Markdown format for better readability.
   - Respond in the language specified in the "Language" field of the query (e.g., English, Arabic). Match the response language to the query language.
   - Address the query comprehensively and accurately while ensuring clarity.

2. **Use of References:**
   - Include numerical citations ([1], [2], etc.) in the response to indicate the source document of the information.
   - Reference [1] corresponds to the first document in the context, [2] to the second, and so forth.

3. **Enhanced Formatting for Readability:**
   - Use Markdown formatting to emphasize important points, headings, and critical details (e.g., **bold text**, *italic text*, lists).
   - Organize content into sections or bullet points if needed.

4. **Relevant Information Only:**
   - Base your answers strictly on the provided context.
   - Avoid introducing external knowledge or speculative information.

5. **Avoid Assumptions:**
   - Refrain from making assumptions or fabricating details beyond the given context.

6. **Identify Yourself When Asked:**
   - If requested, clearly state that you are a highly intelligent assistant developed by lawa.ai.

**Rules:**
- Do not mention the term "context" in your answers.
- Use only the information relevant to the query.
- If no relevant context is provided, respond with general knowledge relevant to the query.

**Input Format Example:**
User Query: What are the latest updates on the scholarship policies at MBZUAI?
Language: English
context:
<provided context>

**Output Format Example:**
**Latest Updates on Scholarship Policies:**

MBZUAI recently updated its scholarship policies to include the following:

1. **Scholarship Coverage**: Full tuition fees, accommodation, and a monthly stipend. [1]
2. **Eligibility Criteria**: Applicants must maintain a GPA of 3.5 or higher. [2]

For further details, please refer to the official documents. If you have more specific questions, feel free to ask!
"""

# Pydantic models
class ChatRequest(BaseModel):
    question: str = Field(..., max_length=1024)
    language: str
    previous_chats: List

class CitationSource(BaseModel):
    url: str
    cite_num: str

class ChatResponse(BaseModel):
    content: str
    citations: List[CitationSource]

# Initialize Pinecone components with retries
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
                top_k=40,
                alpha=0.6,
            )
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            logger.warning(f"Pinecone initialization attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)

retriever = initialize_pinecone()

async def safe_send(websocket: WebSocket, message: dict):
    try:
        await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.info("Client disconnected during send")
        raise
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise

# Helper function to rerank documents using Pinecone's rerank feature
def rerank_docs(query, docs, pc_client):
    try:
        result = pc_client.inference.rerank(
            model="cohere-rerank-3.5",
            query=query,
            documents=docs,
            rank_fields=["chunk"],
            top_n=20,
            return_documents=True
        )

        ranked_docs = []
        for ele in result.data:
            ranked_docs.append({
                "page_source": ele.document.page_source,
                "chunk": ele.document.chunk,
                "summary": ele.document.summary
            })

        return ranked_docs
    except Exception as e:
        print(f"Error in rerank_docs: {e}")
        raise

# Helper function to format documents into a single context string
def format_docs(docs):
    context = ""
    for index, ele in enumerate(docs):
        context += f"\n{'=' * 150}\n**DOCUMENT:** {index + 1} \n**SOURCE:** {ele['page_source']}\n\n**CONTENT:** {ele['chunk']}\n\n"
    return context

# Helper function to format the query with language and context
def format_query(query, language, docs):
    formatted_docs = format_docs(docs)
    formatted_query = f"""
**USER QUERY:** {query} \n
**LANGUAGE:** {language} \n
**CONTEXT:** \n
{formatted_docs} 
    """
    return formatted_query


def validate_citation_numbers(citation_numbers: List[int], max_docs: int) -> List[int]:
    return [num for num in citation_numbers if 1 <= num <= max_docs]

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    send_task = None
    stop_event = asyncio.Event()

    try:
        # Receive and validate request
        try:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=30)
            chat_request = ChatRequest(**data)
        except ValidationError as e:
            print("validation Error", e)
            await safe_send(websocket, {"response": str(e)})
            return
        except json.JSONDecodeError:
            await safe_send(websocket, {"response": "Invalid JSON format"})
            return

        # Background task to send buffered chunks
        async def send_buffered_chunks():
            nonlocal chunk_buffer
            while not stop_event.is_set():
                await asyncio.sleep(2)
                if chunk_buffer:
                    try:
                        await safe_send(websocket, {"response": chunk_buffer})
                        chunk_buffer = ""
                    except:
                        break

        # Processing pipeline
        question = chat_request.question
        language = chat_request.language
        chunk_buffer = ""
        complete_answer = ""

        # Document retrieval
        await safe_send(websocket, {"status": "retrieving"})
        try:
            retrieve_documents = await asyncio.to_thread(retriever.invoke, question)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            await safe_send(websocket, {"error": "Document retrieval failed"})
            return

        docs = [{
            "summary": ele.metadata.get("summary", ""),
            "chunk": ele.page_content,
            "page_source": ele.metadata.get("source", "")
        } for ele in retrieve_documents]

        if not docs:
            await safe_send(websocket, {"status": "no_documents_found"})
            return

        # Document reranking
        await safe_send(websocket, {"status": "reranking"})
        try:
            ranked_docs = await asyncio.to_thread(rerank_docs, question, docs, pc)
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            ranked_docs = docs  # Fallback to original docs

        print(chat_request.previous_chats)
        # Prepare messages with history truncation
        messages = [{"role": "system", "content": system_prompt}]
        messages += [msg for msg in chat_request.previous_chats[:-1]]
        messages.append({"role": "user", "content": format_query(question, language, ranked_docs)})

        # Stream response
        send_task = asyncio.create_task(send_buffered_chunks())
        
        try:
            completion = openai_client.chat.completions.create(
                model="sonar",
                messages=messages,
                temperature=0.8,
                max_completion_tokens=1024,
                stream=True
            )

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    complete_answer += (delta := chunk.choices[0].delta.content)
                    chunk_buffer += delta

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            await safe_send(websocket, {"error": "Response generation failed"})
            return
        finally:
            stop_event.set()
            await send_task

        citations = []
        # Collect citation numbers in order of appearance without duplicates
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
                url = ranked_docs[num-1]["page_source"]
                if url not in seen_urls:
                    # New URL, assign the next consecutive number
                    citation_map[num] = current_num
                    seen_urls[url] = current_num
                    citations.append({"url": url, "cite_num": str(current_num)})
                    current_num += 1
                else:
                    # Map to existing citation number
                    citation_map[num] = seen_urls[url]
            except IndexError:
                continue

        # Replace citations in text
        def replace_citation(match):
            original = int(match.group(1))
            new_num = citation_map.get(original, original)  # Default to original if not mapped
            url = next((c["url"] for c in citations if c["cite_num"] == str(new_num)), "")
            return f"[{new_num}]({url})" if url else f"[{new_num}]"

        complete_answer = re.sub(r'\[(\d+)\]', replace_citation, complete_answer)

        await safe_send(websocket, {
            "response": complete_answer,
            "sources": sorted(citations, key=lambda x: int(x["cite_num"]))
        })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await safe_send(websocket, {"response": "Something went wrong! Please try again."})
    finally:
        stop_event.set()
        if send_task and not send_task.done():
            await send_task