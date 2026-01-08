# app/services/pdf_service.py
import logging
import pymupdf4llm
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def process_pdf_to_text(file_path: str) -> str:
    """
    Extracts raw markdown content from a PDF file.
    All LLM-based sentence generation is removed.
    """
    logger.info("Extracting raw markdown from PDF: %s", file_path)
    
    # Use pymupdf4llm to get structured markdown
    markdown_content = pymupdf4llm.to_markdown(file_path)

    if not markdown_content.strip():
        raise ValueError("PDF text extraction resulted in empty content.")

    # --- Optional: Save debug file ---
    # This is good practice to see what the splitter will receive
    try:
        project_root = Path(__file__).resolve().parents[2]
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        source_name = Path(file_path).name
        # Save as .md to make it easy to read
        extracted_path = outputs_dir / f"{Path(source_name).stem}_extracted_markdown.md"
        
        with extracted_path.open("w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        logger.info("Saved extracted markdown to %s", extracted_path)
    except Exception:
        logger.warning("Failed to persist extracted markdown locally", exc_info=True)
    # --- End Optional ---

    # Return the raw markdown, not the "cleaned" text
    return markdown_content