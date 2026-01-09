import logging
import time
import threading
import concurrent.futures
from typing import List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from app.core.config import settings

logger = logging.getLogger(__name__)

_session_lock = threading.Lock()
_session: requests.Session | None = None

def _get_session() -> requests.Session:
    global _session
    if _session is not None:
        return _session
    with _session_lock:
        if _session is not None:
            return _session
        session = requests.Session()
        retry = Retry(
            total=settings.OLLAMA_MAX_RETRIES,
            connect=settings.OLLAMA_MAX_RETRIES,
            read=settings.OLLAMA_MAX_RETRIES,
            backoff_factor=settings.OLLAMA_RETRY_BACKOFF_FACTOR,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _session = session
        return _session

def _request_embedding(session: requests.Session, text: str) -> List[float]:
    text = text.replace("\n", " ").strip()
    t0 = time.perf_counter()
    logger.info("Requesting embedding (len=%d)", len(text))
    response = session.post(
        f"{settings.OLLAMA_BASE_URL}/api/embeddings",
        json={"model": settings.OLLAMA_EMBEDDING_MODEL, "prompt": text},
        timeout=(settings.OLLAMA_CONNECT_TIMEOUT_SECONDS, settings.OLLAMA_READ_TIMEOUT_SECONDS),
    )
    response.raise_for_status()
    result = response.json()
    embedding = result.get("embedding", [])
    if not embedding or len(embedding) != settings.EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding dimension mismatch from Ollama: expected {settings.EMBEDDING_DIMENSIONS}, got {len(embedding)}"
        )
    logger.info("Embedding received in %.2f ms", (time.perf_counter() - t0) * 1000)
    return embedding

def create_embedding(text: str) -> List[float]:
    session = _get_session()
    try:
        return _request_embedding(session, text)
    except requests.exceptions.RequestException as e:
        logger.exception("Error connecting to Ollama for embeddings")
        raise ConnectionError(
            f"Could not connect to Ollama at {settings.OLLAMA_BASE_URL} for embeddings."
        ) from e
    except Exception:
        logger.exception("Error generating Ollama embedding")
        raise

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    logger.info("Starting embeddings for %d chunks", len(texts))
    max_workers = max(1, int(settings.EMBEDDING_MAX_WORKERS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        embeddings = list(executor.map(create_embedding, texts))
    logger.info("Completed: %d embeddings created", len(embeddings))
    return embeddings