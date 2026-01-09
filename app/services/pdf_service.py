# app/services/pdf_service.py
import logging
import pymupdf4llm
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def process_pdf_to_text(file_path: str) -> str:
    logger.info("Extracting raw markdown from PDF: %s", file_path)
    
    markdown_content = pymupdf4llm.to_markdown(file_path)

    if not markdown_content.strip():
        raise ValueError("PDF text extraction resulted in empty content.")

    try:
        project_root = Path(__file__).resolve().parents[2]
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        source_name = Path(file_path).name
        extracted_path = outputs_dir / f"{Path(source_name).stem}_extracted_markdown.md"
        
        with extracted_path.open("w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        logger.info("Saved extracted markdown to %s", extracted_path)
    except Exception:
        logger.warning("Failed to persist extracted markdown locally", exc_info=True)

    return markdown_content