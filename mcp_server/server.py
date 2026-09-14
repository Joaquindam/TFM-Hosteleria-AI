"""MCP Server "restaurant-intelligence-mcp" (ver PROJECT_BRIEF.md secciones
9 y 10). Expone como tools MCP las funciones de `tools.py` -- la logica de
negocio vive ahi, testable sin cliente MCP; este archivo solo la registra
y la sirve por stdio.

Ejecutar directamente para levantar el servidor:
    python -m mcp_server.server
"""
from mcp.server.mcpserver import MCPServer

from mcp_server import tools

mcp = MCPServer(
    name="restaurant-intelligence-mcp",
    instructions=(
        "Herramientas de datos estructurados para el restaurante La Roca "
        "(Pozuelo). Usa estas tools para cualquier cifra, prediccion o dato "
        "calculado -- nunca inventes un numero. Usa search_knowledge_base "
        "solo para conocimiento documental (carta, politicas, resenas)."
    ),
)

for _fn in [
    tools.get_revenue_forecast,
    tools.get_ticket_forecast,
    tools.get_historical_revenue,
    tools.get_reservation_status,
    tools.get_weather_forecast,
    tools.get_events,
    tools.get_review_analysis,
    tools.get_ai_visibility,
    tools.get_competitor_analysis,
    tools.run_revenue_scenario,
    tools.run_ticket_scenario,
    tools.search_knowledge_base,
]:
    mcp.add_tool(_fn)


if __name__ == "__main__":
    mcp.run(transport="stdio")
