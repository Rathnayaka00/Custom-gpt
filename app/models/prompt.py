from pydantic import BaseModel, Field


class GeneratePromptRequest(BaseModel):
    intent: str = Field(..., min_length=2, description="Short instruction, e.g. 'Create NDIS Report'")
    session_id: str | None = Field(default=None, description="If provided, uses session document scope + stores nothing else.")
    top_k: int = Field(default=15, ge=1, le=30, description="How many RAG chunks to consider for context building.")


class GeneratePromptResponse(BaseModel):
    prompt: str
    session_id: str | None = None
    document_ids: list[str] | None = None
    used_context: bool = False


