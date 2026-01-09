# app/services/text_splitter.py
import re
import sys
from typing import List, Dict, Optional, Tuple
from app.core.config import settings

class _ParagraphSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = []
        start_index = 0
        while start_index < len(text):
            end_index = start_index + self.chunk_size
            chunks.append(text[start_index:end_index])
            start_index += self.chunk_size - self.chunk_overlap
        return chunks

_paragraph_splitter = _ParagraphSplitter(
    chunk_size=settings.TEXT_SPLITTER_CHUNK_SIZE,
    chunk_overlap=settings.TEXT_SPLITTER_CHUNK_OVERLAP
)

_PAGE_FOOTER_RE = re.compile(r"\bAward Code:.*\s(\d+)\s*$", re.IGNORECASE)
_TABLE_TITLE_RE = re.compile(r"^\*{0,2}\s*Table\s+\d+(\s+of\s+\d+)?\s*\*{0,2}$", re.IGNORECASE)


def _clean_md(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"__(.*?)__", r"\1", t)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _classify_text_block(text: str) -> str:
    tl = (text or "").lower()
    if any(k in tl for k in ("definition:", "means ", "is defined as", "refers to")):
        return "definition"
    if any(k in tl for k in ("steps", "procedure", "how to", "process", "method")):
        return "procedure"
    return "text_paragraph"


def _build_prefix(section_path: str, table_title: Optional[str], page_number: Optional[int]) -> str:
    parts: List[str] = []
    if section_path:
        parts.append(f"Section: {section_path}")
    if table_title:
        parts.append(f"Table: {table_title}")
    if page_number is not None:
        parts.append(f"Page: {page_number}")
    return " | ".join(parts).strip()


def _process_text_chunk(buffer: List[str], section_path: str, page_number: Optional[int]) -> List[Dict[str, str]]:
    content = " ".join(buffer).strip()
    if not content:
        return []

    content = _clean_md(content)
    chunk_type = _classify_text_block(content)
    prefix = _build_prefix(section_path=section_path, table_title=None, page_number=page_number)

    if len(content) > _paragraph_splitter.chunk_size:
        sub_chunks = _paragraph_splitter.split_text(content)
        out: List[Dict[str, str]] = []
        for chunk in sub_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            final_text = f"{prefix}\n{chunk}".strip() if prefix else chunk
            out.append({
                "content": final_text,
                "type": chunk_type,
                "page_number": page_number,
                "section_path": section_path,
            })
        return out
    else:
        final_text = f"{prefix}\n{content}".strip() if prefix else content
        return [{
            "content": final_text,
            "type": chunk_type,
            "page_number": page_number,
            "section_path": section_path,
        }]

def _process_table_chunk(buffer: List[str], section_path: str, table_title: Optional[str], page_number: Optional[int]) -> List[Dict[str, str]]:
    chunks = []
    headers = []
    
    if not buffer:
        return []

    header_line = buffer[0]
    headers = [h.strip() for h in header_line.strip('| \n').split('|')]
    headers = [_clean_md(h) for h in headers]
    prefix = _build_prefix(section_path=section_path, table_title=table_title, page_number=page_number)
    
    for line in buffer[1:]: # Skip the header line itself
        line = line.strip()
        
        if line.startswith('|--') or not line:
            continue
            
        cells = [c.strip() for c in line.strip('| \n').split('|')]
        cells = [_clean_md(c) for c in cells]
        
        if len(cells) != len(headers):
            continue

        row_content_parts = []
        for header, cell in zip(headers, cells):
            if header and cell:
                row_content_parts.append(f"{header}: {cell}")
        
        row_content = ". ".join(row_content_parts)
        
        if row_content:
            row_content = _clean_md(row_content)
            final_text = f"{prefix}\n{row_content}".strip() if prefix else row_content
            chunks.append({
                "content": final_text,
                "type": "table_row",
                "page_number": page_number,
                "section_path": section_path,
            })
            
    return chunks

def split_text(markdown_text: str) -> List[Dict[str, str]]:
    all_chunks = []
    text_buffer = []
    table_buffer = []
    headings: List[str] = []
    current_page: Optional[int] = None
    current_table_title: Optional[str] = None

    lines = markdown_text.split('\n')

    for line in lines:
        stripped_line = line.strip()

        m = _PAGE_FOOTER_RE.search(stripped_line)
        if m:
            try:
                current_page = int(m.group(1))
            except Exception:
                pass
            continue

        if stripped_line.startswith("#"):
            if text_buffer:
                all_chunks.extend(_process_text_chunk(text_buffer, _section_path(headings), current_page))
                text_buffer = []
            if table_buffer:
                all_chunks.extend(_process_table_chunk(table_buffer, _section_path(headings), current_table_title, current_page))
                table_buffer = []
            current_table_title = None
            level = len(stripped_line) - len(stripped_line.lstrip("#"))
            heading_text = _clean_md(stripped_line[level:].strip())
            if heading_text:
                headings = _update_headings(headings, level, heading_text)
            continue

        if _TABLE_TITLE_RE.match(_clean_md(stripped_line)):
            current_table_title = _clean_md(stripped_line).strip("*").strip()
            continue

        if stripped_line.startswith('|'):
            if text_buffer:
                all_chunks.extend(_process_text_chunk(text_buffer, _section_path(headings), current_page))
                text_buffer = []
            table_buffer.append(stripped_line)
        else:
            if table_buffer:
                all_chunks.extend(_process_table_chunk(table_buffer, _section_path(headings), current_table_title, current_page))
                table_buffer = []
                current_table_title = None
            
            if stripped_line:
                text_buffer.append(stripped_line)
            elif text_buffer:
                all_chunks.extend(_process_text_chunk(text_buffer, _section_path(headings), current_page))
                text_buffer = []

    if text_buffer:
        all_chunks.extend(_process_text_chunk(text_buffer, _section_path(headings), current_page))
    if table_buffer:
        all_chunks.extend(_process_table_chunk(table_buffer, _section_path(headings), current_table_title, current_page))

    return all_chunks


def _update_headings(existing: List[str], level: int, heading_text: str) -> List[str]:
    lvl = max(1, min(6, int(level)))
    out = existing[:]
    if len(out) < lvl:
        out.extend([""] * (lvl - len(out)))
    out[lvl - 1] = heading_text
    if len(out) > lvl:
        out = out[:lvl]
    while out and not out[-1]:
        out.pop()
    return out


def _section_path(headings: List[str]) -> str:
    cleaned = [h for h in headings if h]
    if not cleaned:
        return ""
    return " > ".join(cleaned[-2:])

text_splitter = sys.modules[__name__]