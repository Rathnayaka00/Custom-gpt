from pydantic import BaseModel

class ProcessResponse(BaseModel):
    filename: str
    message: str
    total_chunks_processed: int
    vector_index: str
    document_id: str | None = None
    session_id: str | None = None


class UploadSessionItem(BaseModel):
    filename: str
    document_id: str
    total_chunks_processed: int | None = None
    success: bool = True
    error: str | None = None


class UploadToSessionResponse(BaseModel):
    session_id: str
    vector_index: str
    items: list[UploadSessionItem]
    total_files: int
    success_files: int
    failed_files: int


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 15
    use_reranking: bool = True
    session_id: str | None = None
    s3_key: str | None = None


class AskWithContextResponse(BaseModel):
    question: str
    answer: str
    session_id: str | None = None
    document_ids: list[str] | None = None


class MetadataItem(BaseModel):
    s3_key: str
    filename: str
    content_type: str
    file_size: int
    total_chunks: int
    document_bucket: str | None = None
    document_key: str | None = None
    upload_date: str
    status: str
    created_at: str
    deleted_at: str | None = None


class ListDocumentsResponse(BaseModel):
    items: list[MetadataItem]
    total: int


class DeleteEmbeddingsRequest(BaseModel):
    s3_key: str
    delete_metadata: bool = False


class DeleteEmbeddingsResponse(BaseModel):
    s3_key: str
    deleted_vectors: int
    metadata_deleted: bool
    metadata_marked_deleted: bool


class SessionDocumentsResponse(BaseModel):
    session_id: str
    document_ids: list[str]
    documents: list[MetadataItem]
    missing_document_ids: list[str]