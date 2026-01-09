
import logging
from pathlib import Path
from app.services import pdf_service, embedding_service, vector_service
from app.services.text_splitter import text_splitter 
from app.services.metadata_service import store_document_metadata

logger = logging.getLogger(__name__)

def process_pdf_and_store(
    file_path: str,
    filename: str,
    document_bucket: str | None = None,
    document_key: str | None = None,
) -> int:
    
    markdown_content = pdf_service.process_pdf_to_text(file_path)
    
    chunks_with_metadata = text_splitter.split_text(markdown_content)
    
    if not chunks_with_metadata:
        logger.warning("Could not extract any processable text chunks from document: %s", filename)
        raise ValueError("Could not extract any processable text chunks from the document.")

    chunks_text_only = [chunk['content'] for chunk in chunks_with_metadata]
    
    embeddings = embedding_service.create_embeddings_batch(chunks_text_only)
    
    total_stored = vector_service.store_vectors_in_s3(
        processed_chunks=chunks_with_metadata, 
        embeddings=embeddings, 
        filename=filename
    )

    try:
        file_size = Path(file_path).stat().st_size
        s3_key_base = filename.rsplit('.', 1)[0]
        store_document_metadata(
            s3_key=s3_key_base,
            filename=filename,
            content_type="application/pdf",
            file_size=file_size,
            total_chunks=total_stored,
            document_bucket=document_bucket,
            document_key=document_key,
        )
    except Exception as meta_err:
        logger.warning("Failed to store document metadata: %s", meta_err)

    return total_stored