"""Orquestador del AI Advisor: bucle de tool-use sobre la Messages API de
Anthropic, reutilizando las tools de `mcp_server/` (que a su vez envuelven
`ml/` y `rag/`). Mismo patron ya probado en `src/tfm_agente.py` (bucle nativo
tool_use/tool_result del SDK, en vez de parsear texto tipo ReAct a mano),
aplicado aqui a una conversacion con el gerente en vez de a mapeo de ficheros.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv

from llm_app.system_prompt import build_system_prompt
from llm_app.tool_schemas import TOOL_DEFINITIONS, TOOL_DISPATCH

load_dotenv()

DEFAULT_MODEL = os.environ.get("LLM_APP_MODEL", "claude-sonnet-5")
MAX_ITERATIONS = 8


class LLMAppNotConfiguredError(RuntimeError):
    """No hay ANTHROPIC_API_KEY configurada -- ver .env.example."""


@dataclass
class LLMAppReply:
    reply: str
    sources: list[str] = field(default_factory=list)


def _tool_result_to_json(result) -> str:
    return json.dumps(result, default=str, ensure_ascii=False)


def run_chat(message: str, history: list[dict] | None = None,
             client: anthropic.Anthropic | None = None,
             model: str = DEFAULT_MODEL) -> LLMAppReply:
    """Ejecuta el bucle de tool-use para una pregunta del gerente.

    `history` son turnos anteriores ya resueltos (`{"role": "user"|"assistant",
    "content": str}`, sin bloques tool_use/tool_result) -- da contexto
    conversacional sin arrastrar el historial completo de llamadas a tools.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMAppNotConfiguredError(
            "Falta ANTHROPIC_API_KEY. Definela en un archivo .env en la raiz del "
            "proyecto (ver .env.example) y reinicia el backend."
        )

    client = client or anthropic.Anthropic()
    messages: list[dict] = [{"role": t["role"], "content": t["content"]} for t in (history or [])]
    messages.append({"role": "user", "content": message})

    sources: set[str] = set()

    for _ in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            system=build_system_prompt(),
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_blocks:
            text = "".join(b.text for b in response.content if b.type == "text")
            return LLMAppReply(reply=text, sources=sorted(sources))

        tool_results = []
        for block in tool_blocks:
            fn = TOOL_DISPATCH.get(block.name)
            if fn is None:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": f"Tool desconocida: {block.name}", "is_error": True,
                })
                continue
            try:
                result = fn(**block.input)
                if isinstance(result, dict) and result.get("source"):
                    sources.add(str(result["source"]))
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": _tool_result_to_json(result),
                })
            except Exception as exc:  # un fallo de una tool no debe tumbar la conversacion
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": f"Error al ejecutar {block.name}: {exc}", "is_error": True,
                })

        messages.append({"role": "user", "content": tool_results})

    return LLMAppReply(
        reply=(
            "No he podido completar la respuesta en el número de pasos permitido "
            "-- prueba a reformular la pregunta o a dividirla en partes."
        ),
        sources=sorted(sources),
    )
