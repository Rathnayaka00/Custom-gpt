import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.models.execute import ExecuteWithPromptRequest, ExecuteWithPromptResponse
from app.services.llm_service import generate_answer_with_system_prompt
from app.services.qa_service import ask_with_context
from app.services.session_service import get_session, append_execution, append_messages

router = APIRouter()
logger = logging.getLogger(__name__)


def _select_prompt(sess: dict, prompt_index: int | None) -> str:
    prompts = (sess or {}).get("generated_prompts") or []
    if not prompts:
        return ""
    if prompt_index is None:
        return str(prompts[-1].get("prompt") or "").strip()
    if 0 <= int(prompt_index) < len(prompts):
        return str(prompts[int(prompt_index)].get("prompt") or "").strip()
    return ""


@router.post("/execute-with-prompt", response_model=ExecuteWithPromptResponse)
async def execute_with_prompt(request: ExecuteWithPromptRequest):
    try:
        sess = await get_session(request.session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="session_id not found")

        doc_ids = sess.get("document_ids") or None
        system_prompt = _select_prompt(sess, request.prompt_index)
        if not system_prompt:
            raise HTTPException(status_code=400, detail="No generated prompt found for this session_id. Call /generate-prompt first.")

        # Build doc-scoped context via existing embeddings/vector retrieval.
        context = ""
        used_context = False
        try:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(
                None,
                lambda: ask_with_context(request.query, top_k=request.top_k, use_reranking=request.use_reranking, doc_ids=doc_ids),
            )
            ctx = await asyncio.wait_for(fut, timeout=12.0)
            context = (ctx or {}).get("context", "") or ""
            used_context = bool(context.strip())
        except Exception:
            # If embeddings are down, still execute using system prompt only.
            context = ""
            used_context = False

        answer = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: generate_answer_with_system_prompt(system_prompt, request.query, context),
        )

        # Persist to Mongo
        try:
            await append_execution(
                request.session_id,
                query=request.query,
                prompt_used=system_prompt,
                answer=answer,
                document_ids=doc_ids,
                used_context=used_context,
            )
            await append_messages(
                request.session_id,
                [
                    {"role": "user", "content": request.query},
                    {"role": "assistant", "content": answer},
                ],
            )
        except Exception:
            logger.warning("Failed to store execution/chat history for session_id=%s", request.session_id)

        return ExecuteWithPromptResponse(
            session_id=request.session_id,
            prompt_used=system_prompt,
            answer=answer,
            used_context=used_context,
            document_ids=doc_ids,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error executing task with prompt")
        raise HTTPException(status_code=500, detail=f"Error executing task with prompt: {str(e)}")


