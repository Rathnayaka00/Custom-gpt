from pydantic import BaseModel, Field


class ExecuteWithPromptRequest(BaseModel):
    session_id: str = Field(..., description="MongoDB session_id that contains uploaded document scope + generated prompts.")
    query: str = Field(..., min_length=1, description="User instruction/question to execute using the generated system prompt.")
    top_k: int = Field(default=15, ge=1, le=30, description="How many RAG chunks to consider.")
    use_reranking: bool = Field(default=True)


class ExecuteWithPromptResponse(BaseModel):
    session_id: str
    prompt_used: str
    answer: str
    used_context: bool = False
    document_ids: list[str] | None = None


