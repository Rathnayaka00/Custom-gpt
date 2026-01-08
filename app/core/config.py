from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    PROJECT_NAME: str
    API_V1_STR: str
    UVICORN_HOST: str
    UVICORN_PORT: int

    OLLAMA_BASE_URL: str
    OLLAMA_EMBEDDING_MODEL: str
    EMBEDDING_DIMENSIONS: int

    # LLM provider: "bedrock" (default) or "ollama" for fully offline answers
    LLM_PROVIDER: str = "bedrock"
    OLLAMA_LLM_MODEL: str = ""
    OLLAMA_LLM_TEMPERATURE: float = 0.2
    OLLAMA_LLM_NUM_PREDICT: int = 256

    OLLAMA_CONNECT_TIMEOUT_SECONDS: float = 5.0
    OLLAMA_READ_TIMEOUT_SECONDS: float = 60.0
    OLLAMA_MAX_RETRIES: int = 3
    OLLAMA_RETRY_BACKOFF_FACTOR: float = 0.5

    EMBEDDING_MAX_WORKERS: int = 4

    AWS_REGION: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str

    METADATA_BUCKET_NAME: Optional[str] = None

    BEDROCK_SERVICE_NAME: str
    BEDROCK_MODEL_ID: str
    BEDROCK_LLM_TEMPERATURE: float
    BEDROCK_LLM_MAX_TOKENS: int

    BEDROCK_CONNECT_TIMEOUT_SECONDS: float
    BEDROCK_READ_TIMEOUT_SECONDS: float
    BEDROCK_MAX_RETRIES: int
    BEDROCK_RETRY_MODE: str

    VECTOR_BUCKET_NAME: str
    INDEX_NAME: str
    VECTOR_PUT_BATCH_SIZE: int
    VECTOR_STORAGE_MAX_RETRIES: int
    VECTOR_STORAGE_BACKOFF_FACTOR: int

    TEXT_SPLITTER_CHUNK_SIZE: int
    TEXT_SPLITTER_CHUNK_OVERLAP: int

    # MongoDB (local session store)
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "rag_pipeline"
    MONGODB_SESSIONS_COLLECTION: str = "chat_sessions"
    MONGODB_SESSION_TTL_DAYS: int = 30

    # Retrieval tuning (safe defaults)
    RETRIEVAL_ALPHA: float = 0.75
    RETRIEVAL_LAMBDA_MMR: float = 0.6
    RETRIEVAL_CANDIDATE_K: int = 60
    RETRIEVAL_MAX_TOP_K: int = 30
    RETRIEVAL_MAX_CANDIDATES: int = 120

    PDF_SENTENCE_GEN_MAX_INPUT_CHARS: int
    PDF_SENTENCE_GEN_CHUNK_OVERLAP_CHARS: int

settings = Settings()