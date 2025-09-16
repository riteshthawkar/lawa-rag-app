import asyncio
import time
import os
from typing import List, Dict
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from modules.config import (
    logger, PINECONE_INDEX_NAME, TOP_K_DOCS, RERANK_TOP_N, 
    ALPHA_HYBRID, RERANK_MODEL, EMBEDDING_MODEL, MAX_RETRIES
)

# Initialize retrieval components

def initialize_pinecone():
    """Initialize the Pinecone retriever with retries"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    for attempt in range(MAX_RETRIES):
        try:
            index = pc.Index(PINECONE_INDEX_NAME)
            bm25 = BM25Encoder().load("./LAWA_COMBINED_BM25_ENCODER.json")
            
            embed_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"trust_remote_code": True, "token": os.getenv("HF_TOKEN", "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")},
            )
            
            return (
                PineconeHybridSearchRetriever(
                    embeddings=embed_model,
                    sparse_encoder=bm25,
                    index=index,
                    top_k=TOP_K_DOCS,
                    alpha=ALPHA_HYBRID,
                ),
                pc
            )
        except Exception as e:
            logger.warning(f"Pinecone initialization attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                logger.exception("Failed to initialize Pinecone after multiple attempts.")
                raise
            time.sleep(2 ** attempt)

def rerank_docs(query: str, docs: List[dict], pc_client: Pinecone) -> List[dict]:
    """Reranks documents using Pinecone reranking"""
    try:
        result = pc_client.inference.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=docs,
            rank_fields=["chunk"],
            top_n=RERANK_TOP_N,
            return_documents=True
        )
        
        # Create a mapping of original docs by chunk content for source lookup
        docs_by_chunk = {doc["chunk"]: doc for doc in docs}
        
        ranked_docs = []
        for ele in result.data:
            # Try to get the original document to preserve source
            original_doc = docs_by_chunk.get(ele.document.chunk, {})
            
            ranked_docs.append({
                "page_source": original_doc.get("page_source", ""),
                "chunk": ele.document.chunk,
                "summary": original_doc.get("summary", "")
            })
        
        return ranked_docs
    except Exception as e:
        logger.exception("Error in rerank_docs:")
        raise
