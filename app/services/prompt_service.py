from __future__ import annotations

import logging
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import asyncio
from starlette.concurrency import run_in_threadpool

from app.core.aws import bedrock_runtime_client
from app.core.config import settings
from app.services.qa_service import ask_with_context

logger = logging.getLogger(__name__)


def _ensure_act_as(prompt: str, fallback_role: str) -> str:
    p = (prompt or "").strip()
    if not p:
        return f"Act as {fallback_role}. Produce the requested output."
    if p.lower().startswith("act as"):
        return p
    return f"Act as {fallback_role}. {p}"

def _prompt_too_short(prompt: str) -> bool:
    p = (prompt or "").strip()
    if not p:
        return True
    return (len(p) < 1200) or (p.count("\n") < 6)

def _strip_reasoning_tags(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", t, flags=re.IGNORECASE).strip()
    return t


def _build_background(intent: str, context: str) -> str:
    parts: List[str] = []
    parts.append(f"User intent (very short): {intent.strip()}")
    if context and context.strip():
        parts.append("Relevant extracted context (RAG snippets):\n" + context.strip())
    return "\n\n".join(parts).strip()


def _generate_prompt_sync(
    intent: str,
    context: str,
) -> Tuple[str, str]:
    background = _build_background(intent=intent, context=context)

    system = (
        "You are a senior prompt engineer for professional report-writing workflows. "
        "The user will provide a very short intent (3–8 words). "
        "Expand it into a LONG, well-explained, high-quality instruction prompt.\n\n"
        "CRITICAL RULES:\n"
        "- Output MUST be plain text only (no JSON, no markdown fences).\n"
        "- The output MUST start with exactly: 'Act as'.\n"
        "- The output MUST be long and detailed (aim for 8–15 sections, multiple paragraphs).\n"
        "- If RAG context is provided, instruct to use it; if it's missing, explicitly forbid hallucination.\n"
        "- If attachment details are provided, incorporate them.\n\n"
        "REQUIRED SECTIONS (use clear headings):\n"
        "1) Role/Persona (Act as ...)\n"
        "2) Objective (what to produce)\n"
        "3) Audience + tone\n"
        "4) Inputs to extract/use (from attachments + from context)\n"
        "5) Output format (headings, tables, bullet lists)\n"
        "6) Required content checklist\n"
        "7) Assumptions + questions to ask if info missing\n"
        "8) Privacy/safety constraints (no sensitive leakage)\n"
        "9) Quality checks before finalizing\n"
    )

    user = (
        "Create the best possible prompt based on the following. "
        "If the intent is ambiguous, make sensible defaults and include a short 'Assumptions' section.\n\n"
        f"{background}"
    )

    def _call_bedrock(messages: List[Dict[str, str]]) -> str:
        body = json.dumps({
            "max_tokens": int(settings.BEDROCK_LLM_MAX_TOKENS),
            "temperature": float(settings.BEDROCK_LLM_TEMPERATURE),
            "response_format": {"type": "text"},
            "verbosity": "low",
            "reasoning_effort": "low",
            "messages": messages,
        })
        resp = bedrock_runtime_client.invoke_model(
            modelId=settings.BEDROCK_MODEL_ID,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        resp_json = json.loads(resp.get("body").read())
        choices = resp_json.get("choices", [])
        msg = (choices[0].get("message") or {}) if choices else {}
        return (msg.get("content") or "").strip()

    content1 = _call_bedrock([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    role = "a senior professional report writer"
    prompt = _ensure_act_as(_strip_reasoning_tags(content1), fallback_role=role)

    if _prompt_too_short(prompt):
        expand_user = (
            "Your previous prompt is too short. Expand it substantially.\n"
            "Return plain text only. Keep the output starting with 'Act as'.\n"
            "Add more concrete structure, checklists, and explicit formatting instructions.\n\n"
            f"Previous output:\n{content1}"
        )
        content2 = _call_bedrock([
            {"role": "system", "content": system},
            {"role": "user", "content": expand_user},
        ])
        prompt2 = _ensure_act_as(_strip_reasoning_tags(content2), fallback_role=role)
        return prompt2, role

    return prompt, role


async def generate_dynamic_prompt(
    intent: str,
    *,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    intent = (intent or "").strip()
    if not intent:
        raise ValueError("intent is required")

    context = ""
    try:
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(
            None,
            lambda: ask_with_context(intent, top_k=top_k, use_reranking=True, doc_ids=doc_ids),
        )
        ctx = await asyncio.wait_for(fut, timeout=8.0)
        context = (ctx or {}).get("context", "") or ""
    except Exception:
        context = ""
    used_context = bool(context.strip())

    prompt, role = await run_in_threadpool(
        _generate_prompt_sync,
        intent,
        context,
    )
    return {
        "prompt": prompt,
        "role": role,
        "used_context": used_context,
    }


