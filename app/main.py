import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routes import document
from app.routes import prompt
from app.routes import execute
from app.db.mongo import close_mongo_client
from app.services.session_service import ensure_session_indexes

@asynccontextmanager
async def lifespan(_: FastAPI):
    await ensure_session_indexes()
    yield
    await close_mongo_client()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="API for a RAG pipeline using AWS and Ollama.",
    lifespan=lifespan,
)

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
app.include_router(prompt.router, prefix=settings.API_V1_STR, tags=["Prompt Generator"])
app.include_router(execute.router, prefix=settings.API_V1_STR, tags=["Execute"])
