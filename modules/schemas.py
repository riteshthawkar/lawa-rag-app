from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ChatRequest(BaseModel):
    """Request model for chat endpoints"""
    question: str = Field(..., max_length=1024)
    language: str
    response_detail_level: str = "medium"  # concise, medium, or detailed
    previous_chats: List[dict] = []
    interaction_id: Optional[str] = None

class CitationSource(BaseModel):
    """Model for citation sources"""
    url: str
    cite_num: str 
