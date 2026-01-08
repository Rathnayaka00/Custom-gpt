from pydantic import BaseModel

class ProcessResponse(BaseModel):
    filename: str
    message: str
    total_chunks_processed: int
    vector_index: str


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 15
    use_reranking: bool = True


class AskWithContextResponse(BaseModel):
    question: str
    answer: str


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