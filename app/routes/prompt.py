import logging

from fastapi import APIRouter, HTTPException

from app.models.prompt import GeneratePromptRequest, GeneratePromptResponse
from app.services.prompt_service import generate_dynamic_prompt
from app.services.session_service import get_session, create_session, append_generated_prompt

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/generate-prompt", response_model=GeneratePromptResponse)
async def generate_prompt(request: GeneratePromptRequest):
    try:
        session_id = request.session_id
        doc_ids = None

        if session_id:
            sess = await get_session(session_id)
            if not sess:
                raise HTTPException(status_code=404, detail="session_id not found")
            doc_ids = sess.get("document_ids") or None
        else:
            # No doc scope provided; still create a session so prompts can be stored.
            session_id = await create_session([])

        out = await generate_dynamic_prompt(
            request.intent,
            doc_ids=doc_ids,
            top_k=request.top_k,
        )

        try:
            await append_generated_prompt(
                session_id,
                intent=request.intent,
                prompt=out["prompt"],
                document_ids=doc_ids,
                used_context=bool(out.get("used_context")),
            )
        except Exception:
            logger.warning("Failed to store generated prompt for session_id=%s", session_id)

        return GeneratePromptResponse(
            prompt=out["prompt"],
            session_id=session_id,
            document_ids=doc_ids,
            used_context=bool(out.get("used_context")),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating prompt")
        raise HTTPException(status_code=500, detail=f"Error generating prompt: {str(e)}")


