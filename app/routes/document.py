import logging
import tempfile
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi import Form
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
    UploadToSessionResponse,
    UploadSessionItem,
    SessionDocumentsResponse,
)
from app.services.metadata_service import list_all_documents, get_document_metadata
from app.services.document_service import process_pdf_and_store
from app.services.file_storage_service import store_uploaded_pdf, stream_pdf_from_s3
from app.services.qa_service import answer_with_context
from app.services.deletion_service import delete_embeddings_and_metadata
from app.services.session_service import create_session, get_session, append_messages, add_documents_to_session

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

    document_id = file.filename.rsplit(".", 1)[0]
    try:
        session_id = await create_session([document_id])
    except Exception as e:
        logger.exception("Failed to create MongoDB session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document uploaded but failed to create session (MongoDB): {e}",
            )

    return ProcessResponse(
        filename=file.filename,
        message="Document processed and vectors stored successfully.",
        total_chunks_processed=total_stored,
        vector_index=settings.INDEX_NAME,
        document_id=document_id,
        session_id=session_id,
    )


@router.post("/upload-to-session", response_model=UploadToSessionResponse, status_code=status.HTTP_201_CREATED)
async def upload_multiple_to_session(
    session_id: str | None = Form(default=None),
    files: List[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")

    if session_id:
        sess = await get_session(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="session_id not found")
    else:
        session_id = await create_session([])

    items: List[UploadSessionItem] = []
    new_doc_ids: List[str] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for f in files:
            if not f.filename or not f.filename.lower().endswith(".pdf"):
                items.append(
                    UploadSessionItem(
                        filename=f.filename or "",
                        document_id=(f.filename or "").rsplit(".", 1)[0],
                        success=False,
                        error="Invalid file type. Only PDF files are accepted.",
                    )
                )
                continue

            temp_file_path = Path(temp_dir) / f.filename
            try:
                with open(temp_file_path, "wb") as buffer:
                    buffer.write(await f.read())

                logger.info("Upload received (session=%s): %s", session_id, f.filename)
                doc_bucket, doc_key = store_uploaded_pdf(str(temp_file_path), f.filename)

                total_stored = process_pdf_and_store(
                    str(temp_file_path),
                    f.filename,
                    document_bucket=doc_bucket,
                    document_key=doc_key,
                )

                doc_id = f.filename.rsplit(".", 1)[0]
                new_doc_ids.append(doc_id)
                items.append(
                    UploadSessionItem(
                        filename=f.filename,
                        document_id=doc_id,
                        total_chunks_processed=total_stored,
                        success=True,
                    )
                )
            except Exception as e:
                logger.exception("Pipeline error for file %s", f.filename)
                items.append(
                    UploadSessionItem(
                        filename=f.filename or "",
                        document_id=(f.filename or "").rsplit(".", 1)[0],
                        success=False,
                        error=str(e),
                    )
                )

    try:
        await add_documents_to_session(session_id, new_doc_ids)
    except Exception:
        logger.warning("Failed to attach document_ids to session_id=%s", session_id)

    success_files = sum(1 for it in items if it.success)
    failed_files = sum(1 for it in items if not it.success)
    return UploadToSessionResponse(
        session_id=session_id,
        vector_index=settings.INDEX_NAME,
        items=items,
        total_files=len(items),
        success_files=success_files,
        failed_files=failed_files,
    )


@router.get("/metadata", response_model=ListDocumentsResponse)
async def list_metadata():
    try:
        items = list_all_documents()
        return ListDocumentsResponse(items=[MetadataItem(**item) for item in items], total=len(items))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing metadata: {str(e)}")


@router.get("/sessions/{session_id}/documents", response_model=SessionDocumentsResponse)
async def list_session_documents(session_id: str):
    try:
        sess = await get_session(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="session_id not found")

        doc_ids = list(dict.fromkeys([d for d in (sess.get("document_ids") or []) if d]))
        documents: List[MetadataItem] = []
        missing: List[str] = []

        for doc_id in doc_ids:
            meta = get_document_metadata(doc_id)
            if not meta:
                missing.append(doc_id)
                continue
            try:
                documents.append(MetadataItem(**meta))
            except Exception:
                missing.append(doc_id)

        return SessionDocumentsResponse(
            session_id=session_id,
            document_ids=doc_ids,
            documents=documents,
            missing_document_ids=missing,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching session documents: {str(e)}")


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
        doc_ids = None
        if request.session_id:
            sess = await get_session(request.session_id)
            if not sess:
                raise HTTPException(status_code=404, detail="session_id not found")
            doc_ids = sess.get("document_ids") or None
        elif request.s3_key:
            doc_ids = [request.s3_key]

        direct_answer = answer_with_context(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            doc_ids=doc_ids,
        )

        if request.session_id:
            try:
                await append_messages(
                    request.session_id,
                    [
                        {"role": "user", "content": request.query},
                        {"role": "assistant", "content": direct_answer},
                    ],
                )
            except Exception:
                logger.warning("Failed to persist chat messages for session_id=%s", request.session_id)

        return AskWithContextResponse(
            question=request.query,
            answer=direct_answer,
            session_id=request.session_id,
            document_ids=doc_ids,
        )
    except HTTPException:
        raise
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