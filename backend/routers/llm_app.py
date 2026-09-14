from fastapi import APIRouter, HTTPException

from backend.schemas import LLMAppChatRequest
from llm_app.orchestrator import LLMAppNotConfiguredError, run_chat

router = APIRouter(prefix="/llm_app", tags=["llm_app"])


@router.post("/chat")
def post_llm_app_chat(payload: LLMAppChatRequest):
    """AI Advisor: responde en lenguaje de gerente combinando RAG (carta,
    politicas, reseñas) y MCP/tools (previsiones, reservas, meteo...) --
    ver llm_app/system_prompt.py para las reglas de comportamiento."""
    try:
        history = [{"role": m.role, "content": m.content} for m in payload.history]
        result = run_chat(payload.message, history=history)
        return {"reply": result.reply, "sources": result.sources}
    except LLMAppNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e))
