import logging
import tempfile
from pathlib import Path
from typing import Dict
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse
from app.core.config import settings
from app.models.document import (
    ProcessResponse,
    SemanticSearchRequest,
    AskWithContextResponse,
    MetadataItem,
    ListDocumentsResponse,
    DeleteEmbeddingsRequest,
    DeleteEmbeddingsResponse,
)
from app.services.metadata_service import list_all_documents, get_document_metadata
from app.services.document_service import process_pdf_and_store
from app.services.file_storage_service import store_uploaded_pdf, stream_pdf_from_s3
from app.services.qa_service import answer_with_context
from app.services.deletion_service import delete_embeddings_and_metadata

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=ProcessResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_process_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are accepted."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / file.filename
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        try:
            logger.info("Upload received: %s", file.filename)
            logger.info("Saved temp file to %s", temp_file_path)
            # Store the raw PDF in S3 as well (uses METADATA_BUCKET_NAME)
            doc_bucket, doc_key = store_uploaded_pdf(str(temp_file_path), file.filename)

            total_stored = process_pdf_and_store(
                str(temp_file_path),
                file.filename,
                document_bucket=doc_bucket,
                document_key=doc_key,
            )
        except Exception as e:
            logger.exception("Pipeline error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred during document processing: {e}"
            )

    return ProcessResponse(
        filename=file.filename,
        message="Document processed and vectors stored successfully.",
        total_chunks_processed=total_stored,
        vector_index=settings.INDEX_NAME
    )


@router.get("/metadata", response_model=ListDocumentsResponse)
async def list_metadata():
    try:
        items = list_all_documents()
        return ListDocumentsResponse(items=[MetadataItem(**item) for item in items], total=len(items))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing metadata: {str(e)}")


@router.get("/metadata/{s3_key}", response_model=MetadataItem)
async def get_metadata(s3_key: str):
    try:
        metadata = get_document_metadata(s3_key)
        if not metadata:
            raise HTTPException(status_code=404, detail="Metadata not found")
        return MetadataItem(**metadata)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching metadata: {str(e)}")


@router.get("/download/{s3_key}")
async def download_document(s3_key: str):
    """
    Download the original uploaded PDF for the given s3_key (document id).
    """
    try:
        metadata = get_document_metadata(s3_key)
        if not metadata:
            raise HTTPException(status_code=404, detail="Metadata not found")

        filename = metadata.get("filename") or f"{s3_key}.pdf"
        bucket = metadata.get("document_bucket") or settings.METADATA_BUCKET_NAME
        key = metadata.get("document_key") or f"documents/{s3_key}.pdf"
        if not bucket:
            raise HTTPException(status_code=500, detail="METADATA_BUCKET_NAME is not configured")

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
        return StreamingResponse(
            stream_pdf_from_s3(bucket=bucket, key=key),
            media_type="application/pdf",
            headers=headers,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error downloading document")
        raise HTTPException(status_code=500, detail=f"Error downloading document: {str(e)}")


@router.post("/ask-with-context", response_model=AskWithContextResponse)
async def ask_question_with_context(request: SemanticSearchRequest):
    try:
        direct_answer = answer_with_context(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
        )
        return AskWithContextResponse(question=request.query, answer=direct_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@router.post("/delete-embeddings", response_model=DeleteEmbeddingsResponse)
async def delete_embeddings(request: DeleteEmbeddingsRequest):
    try:
        deleted_count, meta_deleted, meta_marked = delete_embeddings_and_metadata(
            s3_key=request.s3_key,
            delete_metadata=request.delete_metadata,
        )
        return DeleteEmbeddingsResponse(
            s3_key=request.s3_key,
            deleted_vectors=deleted_count,
            metadata_deleted=bool(meta_deleted),
            metadata_marked_deleted=bool(meta_marked),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting embeddings: {str(e)}")