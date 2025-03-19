from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    session_id: str
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None

class ProcessingStatus(BaseModel):
    status: str
    message: str
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    answer: str
    rating: str  # "thumbs_up" or "thumbs_down"

class ComparisonRequest(BaseModel):
    session_id1: str
    session_id2: str
    comparison_type: str = "semantic"  # "semantic" or "text"

class ComparisonResult(BaseModel):
    similarity_score: float
    common_topics: List[str]
    unique_topics1: List[str]
    unique_topics2: List[str]
    document1_name: str
    document2_name: str

class VisualizationData(BaseModel):
    topics: List[Dict[str, Any]]
    entities: Dict[str, List[str]]
    key_phrases: List[str]
    word_frequencies: Dict[str, int]

class DebugMode(BaseModel):
    session_id: str
    enabled: bool = False
    last_query: Optional[str] = None
    retrieved_docs: List[Dict[str, Any]] = []
    query_variations: List[str] = [] 