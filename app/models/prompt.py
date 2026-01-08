from pydantic import BaseModel, Field


class GeneratePromptRequest(BaseModel):
    """
    User passes a very short intent like:
    - "Create NDIS Report"
    - "Create SCHADS Report"
    - "Create Progress Report"
    - "Create Support Plan"
    """

    intent: str = Field(..., min_length=2, description="Short instruction, e.g. 'Create NDIS Report'")
    session_id: str | None = Field(default=None, description="If provided, uses session document scope + stores nothing else.")
    s3_key: str | None = Field(default=None, description="Optional explicit document scope (same as doc_id / metadata s3_key).")
    attachment_details: str | None = Field(
        default=None,
        description="Optional extra details from attachments/forms/emails that should be reflected in the prompt.",
    )
    top_k: int = Field(default=8, ge=1, le=30, description="How many RAG chunks to consider for context building.")


class GeneratePromptResponse(BaseModel):
    prompt: str
    session_id: str | None = None
    document_ids: list[str] | None = None
    used_context: bool = False


