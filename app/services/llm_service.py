import json
import re
from typing import Optional

import requests

from app.core.aws import bedrock_runtime_client
from app.core.config import settings


def _strip_reasoning_and_formatting(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^(Answer:|Final answer:)\s*", "", t, flags=re.IGNORECASE).strip()
    # Strip common markdown emphasis and code formatting just in case
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)  # bold **text**
    t = re.sub(r"__(.*?)__", r"\1", t)      # bold __text__
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", t)  # italics *text*
    t = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"\1", t)      # italics _text_
    t = re.sub(r"`([^`]*)`", r"\1", t)      # inline code `text`
    t = " ".join(t.splitlines()).strip()
    return t


def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer using the configured LLM provider.
    - bedrock: uses AWS Bedrock (current default)
    - ollama: uses local Ollama (/api/chat) for offline answers
    """
    provider = (settings.LLM_PROVIDER or "bedrock").strip().lower()
    if provider == "ollama":
        return _generate_answer_ollama(query, context)
    return _generate_answer_bedrock(query, context)


def _generate_answer_bedrock(query: str, context: str) -> str:
    system_prompt = (
        "You are a helpful, friendly assistant. "
        "Use the provided context when it is relevant. "
        "If the user asks something that does not require context (such as greetings or casual conversation), respond normally. "
        "If the user asks something that requires information from the context, answer using ONLY the provided context. "
        "Be natural, clear, and concise in a conversational tone, preferably 1–3 short sentences. "
        "If the context lacks information for a specific answer, say you don't know. "
        "Do NOT use markdown formatting such as bold, italics, underscores, or code fences in your answer."
    )
    body = json.dumps({
        "max_tokens": settings.BEDROCK_LLM_MAX_TOKENS,
        "temperature": settings.BEDROCK_LLM_TEMPERATURE,
        "response_format": {"type": "text"},
        "verbosity": "low",
        "reasoning_effort": "low",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Answer using ONLY the context below. "
                    "Write naturally in 1–3 short sentences. "
                    "Do not include hidden chain-of-thought, tags, or any markdown formatting.\n\n"
                    f"Context:\n{context}\n\nQuestion: {query}\n"
                ),
            },
        ],
    })

    response = bedrock_runtime_client.invoke_model(
        modelId=settings.BEDROCK_MODEL_ID,
        body=body,
        accept="application/json",
        contentType="application/json",
    )
    response_json = json.loads(response.get("body").read())
    choices = response_json.get("choices", [])
    if not choices:
        return "Insufficient information"
    message = choices[0].get("message") or {}
    answer = (message.get("content") or "").strip()
    answer = _strip_reasoning_and_formatting(answer)
    return answer or "Insufficient information"


def _generate_answer_ollama(query: str, context: str) -> str:
    if not (settings.OLLAMA_LLM_MODEL or "").strip():
        # Fail safe: don't error hard in production path
        return "Insufficient information"

    system_prompt = (
        "You are a helpful assistant. "
        "If the question requires the provided context, answer using ONLY the context. "
        "If the context does not contain the answer, say you don't know. "
        "Keep it concise: 1–3 short sentences. "
        "Do not use markdown formatting."
    )
    payload = {
        "model": settings.OLLAMA_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Answer using ONLY the context below.\n\n"
                    f"Context:\n{context}\n\nQuestion: {query}\n"
                ),
            },
        ],
        "stream": False,
        "options": {
            "temperature": settings.OLLAMA_LLM_TEMPERATURE,
            "num_predict": settings.OLLAMA_LLM_NUM_PREDICT,
        },
    }
    timeout = (settings.OLLAMA_CONNECT_TIMEOUT_SECONDS, settings.OLLAMA_READ_TIMEOUT_SECONDS)
    resp = requests.post(
        f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        json=payload,
        timeout=timeout,
    )
    if not resp.ok:
        return "Insufficient information"

    data = resp.json() or {}
    msg = data.get("message") or {}
    answer = (msg.get("content") or "").strip()
    answer = _strip_reasoning_and_formatting(answer)
    return answer or "Insufficient information"


