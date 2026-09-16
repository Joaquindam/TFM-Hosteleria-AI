"""Orquestador del AI Advisor sobre la API de Groq (compatible con el
formato de tool-calling de OpenAI Chat Completions).

Reutiliza integramente:
- `mcp_server/tools.py` (via TOOL_DISPATCH de `llm_app/tool_schemas.py`) --
  las 17 funciones reales no cambian, son independientes del proveedor.
- `llm_app/tool_schemas.py` -- TOOL_DISPATCH y `to_openai_format()`
  (Groq entiende el mismo formato de tools que OpenAI).
- `llm_app/system_prompt.py` -- el texto de sistema es independiente del
  proveedor, no hay que reescribirlo.

Lo unico distinto de `llm_app/orchestrator.py` (la version de Anthropic)
es el bucle de conversacion en si: Groq/OpenAI devuelve las llamadas a
tools en `message.tool_calls` (con los argumentos como texto JSON que hay
que parsear a mano), no como bloques `tool_use` de Anthropic -- y se le
responde con mensajes de rol "tool", no con bloques "tool_result".
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import openai
from dotenv import load_dotenv
from openai import OpenAI

from llm_app.system_prompt import build_system_prompt
from llm_app.tool_schemas import TOOL_DISPATCH, to_openai_format

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = os.environ.get("LLM_APP_MODEL") or "openai/gpt-oss-120b"
MAX_ITERATIONS = 8

TOOLS = to_openai_format()


class LLMAppNotConfiguredError(RuntimeError):
    """No hay GROQ_API_KEY configurada -- ver .env.example."""


class LLMAppUnavailableError(RuntimeError):
    """La API de Groq ha fallado (cuota agotada, error de red, etc.) --
    error del proveedor, no un fallo de codigo. Se traduce a un 503 en el
    endpoint en vez de un 500 en bruto."""


@dataclass
class LLMAppReply:
    reply: str
    sources: list[str] = field(default_factory=list)


def _tool_result_to_json(result) -> str:
    return json.dumps(result, default=str, ensure_ascii=False)


def run_chat(message: str, history: list[dict] | None = None,
             client: OpenAI | None = None,
             model: str = DEFAULT_MODEL) -> LLMAppReply:
    """Ejecuta el bucle de tool-use para una pregunta del gerente, contra Groq.

    `history` son turnos anteriores ya resueltos (`{"role": "user"|"assistant",
    "content": str}`, sin llamadas a tools) -- da contexto conversacional sin
    arrastrar el historial completo de llamadas a tools de turnos previos.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMAppNotConfiguredError(
            "Falta GROQ_API_KEY. Definela en un archivo .env en la raiz del "
            "proyecto (ver .env.example) y reinicia el backend."
        )

    client = client or OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    messages: list[dict] = [{"role": "system", "content": build_system_prompt()}]
    messages.extend({"role": t["role"], "content": t["content"]} for t in (history or []))
    messages.append({"role": "user", "content": message})

    sources: set[str] = set()

    for _ in range(MAX_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=1500,
                tools=TOOLS,
                messages=messages,
            )
        except openai.RateLimitError as exc:
            raise LLMAppUnavailableError(
                "Se ha agotado la cuota de Groq por ahora (limite diario o por minuto). "
                "Vuelve a intentarlo en unos minutos, o revisa console.groq.com/settings/billing "
                f"si quieres subir de plan. Detalle: {exc}"
            ) from exc
        except (openai.APIStatusError, openai.APIConnectionError) as exc:
            raise LLMAppUnavailableError(
                f"La API de Groq no ha podido responder ahora mismo: {exc}"
            ) from exc
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        tool_calls = assistant_message.tool_calls
        if not tool_calls:
            return LLMAppReply(reply=assistant_message.content or "", sources=sorted(sources))

        for call in tool_calls:
            fn = TOOL_DISPATCH.get(call.function.name)
            if fn is None:
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": f"Tool desconocida: {call.function.name}",
                })
                continue
            try:
                args = json.loads(call.function.arguments or "{}")
                result = fn(**args)
                if isinstance(result, dict) and result.get("source"):
                    sources.add(str(result["source"]))
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": _tool_result_to_json(result),
                })
            except Exception as exc:  # un fallo de una tool no debe tumbar la conversacion
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": f"Error al ejecutar {call.function.name}: {exc}",
                })

    return LLMAppReply(
        reply=(
            "No he podido completar la respuesta en el número de pasos permitido "
            "-- prueba a reformular la pregunta o a dividirla en partes."
        ),
        sources=sorted(sources),
    )
