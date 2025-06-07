from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class DocumentBase(BaseModel):
    title: str
    authors: List[str]
    journal_name: Optional[str] = None
    publication_year: Optional[int] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    url: Optional[str] = None  # filepath in storage
    doi: Optional[str] = None  # DOI identifier
    arxiv_id: Optional[str] = None  # arXiv identifier
    rating: Optional[int] = Field(
        None, ge=1, le=5, description="User rating from 1 to 5"
    )


class DocumentCreate(DocumentBase):
    markdown: Optional[str] = None
    summary: Optional[str] = None
    previous_work: Optional[str] = None
    hypothesis: Optional[str] = None
    distinction: Optional[str] = None
    methodology: Optional[str] = None
    results: Optional[str] = None
    limitations: Optional[str] = None
    implications: Optional[str] = None
    title_embedding: Optional[List[float]] = None
    abstract_embedding: Optional[List[float]] = None
    status: Optional[str] = "pending"
    folder_name: Optional[str] = None


class Document(DocumentCreate):
    id: UUID


class DocumentMetadata(BaseModel):
    title: str
    authors: List[str]
    journal_name: Optional[str] = None
    publication_year: Optional[int] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    markdown: Optional[str] = None
    rating: Optional[int] = None


class DocumentSummary(BaseModel):
    summary: Optional[str] = None
    previous_work: Optional[str] = None
    hypothesis: Optional[str] = None
    distinction: Optional[str] = None
    methodology: Optional[str] = None
    results: Optional[str] = None
    limitations: Optional[str] = None
    implications: Optional[str] = None


class DocumentEmbedding(BaseModel):
    title_embedding: Optional[List[float]] = None
    abstract_embedding: Optional[List[float]] = None


class DocumentListItem(BaseModel):
    id: UUID
    title: str
    authors: List[str]
    journal_name: Optional[str]
    publication_year: Optional[int]
    abstract: Optional[str]
    folder_name: Optional[str]
    doi: Optional[str]
    arxiv_id: Optional[str]
    volume: Optional[str]
    issue: Optional[str]
    url: Optional[str]
    keywords: Optional[List[str]]
    rating: Optional[int] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentListItem]
    total: int
    skip: int
    limit: int


class SearchQuery(BaseModel):
    query: str
    folder_name: Optional[str] = None
    k: int = Field(default=4, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    id: UUID
    title: str
    authors: List[str]
    journal_name: Optional[str] = None
    publication_year: Optional[int] = None
    folder_name: Optional[str] = None
    keywords: Optional[List[str]] = None
    similarity_score: float
    snippet: Optional[str] = None
    url: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int


class DocumentFilters(BaseModel):
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None
    folder_name: Optional[str] = None


class FolderInfo(BaseModel):
    name: str
    path: str
    document_count: int = 0


class FoldersResponse(BaseModel):
    folders: List[FolderInfo]


class IngestRequest(BaseModel):
    folder_name: Optional[str] = None
    clean_ingest: bool = False


class IngestResponse(BaseModel):
    message: str
    folder_name: Optional[str] = None
    clean_ingest: bool


class UpdateSummaryRequest(BaseModel):
    summary: Optional[str] = None
    previous_work: Optional[str] = None
    hypothesis: Optional[str] = None
    distinction: Optional[str] = None
    methodology: Optional[str] = None
    results: Optional[str] = None
    limitations: Optional[str] = None
    implications: Optional[str] = None


class UpdateMetadataRequest(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    journal_name: Optional[str] = None
    publication_year: Optional[int] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    markdown: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)


class UpdateRatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="User rating from 1 to 5")


class StatusResponse(BaseModel):
    total_documents: int
    processed: int
    pending: int
    errors: int


# Legacy models for backward compatibility
PaperBase = DocumentBase
PaperCreate = DocumentCreate
Paper = Document
PaperListItem = DocumentListItem
PaperListResponse = DocumentListResponse


class SimilarPaper(BaseModel):
    id: UUID
    title: str
    authors: List[str]
    similarity_score: float


class SimilarPapersResponse(BaseModel):
    similar_papers: List[SimilarPaper]


class SimilarDocumentRequest(BaseModel):
    limit: int = Field(default=10, ge=1, le=50)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    title_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    abstract_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    include_snippet: bool = Field(default=True)


class SimilarDocument(BaseModel):
    id: UUID
    title: str
    authors: List[str]
    similarity_score: float
    title_similarity: Optional[float] = None
    abstract_similarity: Optional[float] = None
    snippet: Optional[str] = None
    folder_name: Optional[str] = None


class SimilarDocumentsResponse(BaseModel):
    similar_documents: List[SimilarDocument]
    reference_document_id: UUID
    query_weights: Dict[str, float]
    total_results: int


class KeywordSearchQuery(BaseModel):
    keywords: List[str] = Field(
        ..., min_length=1, description="List of keywords to search for"
    )
    search_mode: str = Field(
        default="any", description="Search mode: 'any' (OR logic) or 'all' (AND logic)"
    )
    exact_match: bool = Field(
        default=False, description="Whether to use exact keyword matching"
    )
    case_sensitive: bool = Field(
        default=False, description="Whether search is case sensitive"
    )
    folder_name: Optional[str] = Field(
        default=None, description="Optional folder to scope search"
    )
    limit: int = Field(
        default=50, ge=1, le=100, description="Maximum number of results"
    )
    include_snippet: bool = Field(default=True, description="Include abstract snippets")


class KeywordSearchResult(BaseModel):
    id: UUID
    title: str
    authors: List[str]
    keywords: List[str]
    matched_keywords: List[str]
    match_score: float  # Percentage of keywords matched
    relevance_score: Optional[float] = None  # Weighted relevance score
    snippet: Optional[str] = None
    folder_name: Optional[str] = None
    abstract: Optional[str] = None
    journal_name: Optional[str] = None
    publication_year: Optional[int] = None
    url: Optional[str] = None


class KeywordSearchResponse(BaseModel):
    results: List[KeywordSearchResult]
    query_keywords: List[str]
    search_mode: str
    total_results: int
    exact_match: bool
    case_sensitive: bool


# Chat models
class ChatMessage(BaseModel):
    id: str
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: str


class ChatRequest(BaseModel):
    message: str
    document_id: UUID


class ChatResponse(BaseModel):
    message: ChatMessage
    answer: str
