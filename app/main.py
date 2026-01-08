import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routes import document

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="API for a RAG pipeline using AWS and Ollama."
)

# CORS configuration for local development (Next.js on port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

app.include_router(document.router, prefix=settings.API_V1_STR, tags=["Document Processing"])
