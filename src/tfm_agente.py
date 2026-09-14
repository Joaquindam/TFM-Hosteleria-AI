"""
Agente de adaptación de datos de restaurantes externos al esquema Bronze.

Objetivo
--------
Dado un fichero o carpeta de datos de un restaurante distinto al del TFM
(formato, idioma de columnas y estructura desconocidos), producir los mismos
13 parquet de Bronze que genera hoy `01_ingesta_bronze.ipynb` para el
restaurante original: reservas, tickets, ventas, tips, articulos,
departamentos, menu, festivos, eventos, meteo_diaria, meteo_horaria,
total_articles y facturas.

Diseño (patrón ReAct, un único agente, sin sistema multiagente)
-----------------------------------------------------------------
Como se explica en el Tema 9 (Agentes) del máster, cargar un agente con
demasiadas tools y demasiado contexto degrada su fiabilidad ("El gran
problema": instrucciones contradictorias, pérdida de trazabilidad). Esta
tarea (identificar de qué entidad se trata y proponer un mapeo de columnas)
no necesita un sistema multiagente: un único agente con tres tools de solo
lectura es suficiente.

El bucle Observar -> Razonar -> Actuar del patrón ReAct no se implementa
aquí parseando texto tipo "Accion: X / Entrada: Y" (como en el ejemplo de
las diapositivas) porque el SDK de Anthropic ya materializa ese mismo bucle
de forma nativa a través de bloques `tool_use` / `tool_result`: es más
robusto que parsear el texto del modelo a mano, y es el ReAct real, solo que
implementado por el proveedor en vez de por nosotros.

Tools que el agente puede usar (todas deterministas, de solo lectura):
    - list_target_entities   : qué 13 entidades existen y qué representan.
    - peek_file               : cabecera + muestra de filas de un fichero.
    - peek_folder             : listado de una carpeta + muestra del primer
                                 fichero (para el caso "ventas", que es una
                                 carpeta de informes semanales, no un único
                                 fichero).
    - get_target_schema       : columnas, tipos y notas de una entidad.

Respuesta final: el agente NO devuelve texto libre. Llama a la tool
`registrar_resultado`, que fuerza una salida estructurada (entidad detectada,
confianza, fila donde empieza la tabla real, metadatos de cabecera si los
hay, mapeo columna origen -> columna destino, columnas sin correspondencia).
Este patrón (usar una tool como "salida forzada") es más fiable que pedir
JSON en texto libre y parsearlo a mano.

Este módulo NUNCA escribe el resultado final sin que un humano lo confirme
(ver `revisar_y_confirmar`) ni ejecuta la transformación hasta esa
confirmación explícita.

Requisitos
----------
    pip install anthropic pandas pyarrow openpyxl

Variable de entorno esperada: ANTHROPIC_API_KEY
(La API de Anthropic es un producto distinto de una suscripción de
claude.ai; la clave se genera en console.anthropic.com y tiene facturación
propia, separada de la suscripción de chat.)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

try:
    import anthropic
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Falta el paquete 'anthropic'. Instálalo con: "
        "pip install anthropic --break-system-packages"
    ) from exc


__all__ = [
    "TARGET_SCHEMAS",
    "EntitySchema",
    "MappingProposal",
    "peek_file",
    "peek_folder",
    "get_target_schema",
    "list_target_entities",
    "analizar_fichero",
    "aplicar_mapeo",
    "revisar_y_confirmar",
    "construir_bronze",
]


# Modelo por defecto. Para esta tarea (clasificación + mapeo de columnas,
# sin razonamiento matemático complejo) Haiku suele bastar y es más barato;
# si en las pruebas se equivoca en los casos ambiguos (informes con cabecera
# de metadatos, ficheros con esquema incompleto), subir a "claude-sonnet-5".
MODELO_POR_DEFECTO = "claude-haiku-4-5-20251001"

FILAS_MUESTRA = 25  # cuántas filas crudas se le enseñan al agente por fichero


# ---------------------------------------------------------------------------
# Esquema objetivo (Bronze) — extraído directamente de los *_raw.parquet
# reales del proyecto, no de la lógica interna de ingesta.
# ---------------------------------------------------------------------------

@dataclass
class EntitySchema:
    """
    Contrato de una entidad de Bronze.

    Parameters
    ----------
    columnas : dict[str, str]
        Nombre de columna de destino -> dtype esperado (a nivel descriptivo,
        no se fuerza el dtype exacto de pandas, se usa para dar contexto
        al agente y para la coerción posterior en `aplicar_mapeo`).
    obligatorias : list[str]
        Columnas sin las cuales la entidad no tiene sentido.
    estructura : {'tabla_simple', 'informe_metadatos', 'passthrough', \
'serie_temporal', 'factura'}
        Pista sobre la forma física del fichero de origen esperable.
    incluye_source_file : bool
        Si el parquet final debe llevar la columna `source_file`.
    notas : str
        Contexto en lenguaje natural que se le da al agente para ayudarle a
        reconocer la entidad y a interpretar su estructura.
    """

    columnas: dict[str, str]
    obligatorias: list[str]
    estructura: Literal[
        "tabla_simple", "informe_metadatos", "passthrough", "serie_temporal", "factura"
    ]
    incluye_source_file: bool = True
    notas: str = ""


TARGET_SCHEMAS: dict[str, EntitySchema] = {
    "reservas": EntitySchema(
        columnas={
            "reservation_datetime": "datetime", "created_datetime": "datetime",
            "reservation_date": "datetime", "reservation_time": "str",
            "status": "str", "shift": "str", "people": "Int64", "origin": "str",
            "referrer": "str", "created_date": "datetime", "created_time": "str",
            "restaurant": "str", "reservation_type": "str", "table": "str",
            "zone": "str", "entered_by": "str", "group": "str",
            "reference": "str", "reference_code": "str",
        },
        obligatorias=["reservation_date", "status", "shift", "people"],
        estructura="tabla_simple",
        notas="Una fila por reserva. Columnas típicas de origen: fecha, hora, "
              "estado, turno/servicio, número de personas, canal/origen.",
    ),
    "tickets": EntitySchema(
        columnas={
            "report_start": "datetime", "report_end": "datetime",
            "report_generated_on": "datetime", "terminal_start": "str",
            "terminal_end": "str", "turn": "str", "date": "datetime",
            "document_id": "str", "document_total": "float", "receipt_count": "Int64",
        },
        obligatorias=["date", "document_total"],
        estructura="informe_metadatos",
        notas="Informe de TPV: varias filas de cabecera con rango de fechas, "
              "terminal y turno ANTES de la tabla real. Una fila por documento/ticket.",
    ),
    "ventas": EntitySchema(
        columnas={
            "report_start": "datetime", "report_end": "datetime",
            "report_generated_on": "datetime", "terminal_start": "str",
            "terminal_end": "str", "turn": "str", "department_code": "Int64",
            "department_name": "str", "article_code": "Int64", "article_name": "str",
            "units": "float", "amount": "float",
        },
        obligatorias=["department_code", "article_code", "units", "amount"],
        estructura="informe_metadatos",
        notas="Igual que 'total_articles' pero SUELE venir como una CARPETA de "
              "varios informes semanales, no un único fichero. Usa peek_folder.",
    ),
    "total_articles": EntitySchema(
        columnas={
            "report_start": "datetime", "report_end": "datetime",
            "report_generated_on": "datetime", "terminal_start": "str",
            "terminal_end": "str", "turn": "str", "department_code": "Int64",
            "department_name": "str", "article_code": "Int64", "article_name": "str",
            "units": "float", "amount": "float",
        },
        obligatorias=["department_code", "article_code", "units", "amount"],
        estructura="informe_metadatos",
        notas="Mismo esquema que 'ventas' pero es UN ÚNICO fichero acumulado "
              "de todo el periodo, no una carpeta semanal.",
    ),
    "tips": EntitySchema(
        columnas={
            "report_start": "datetime", "report_end": "datetime",
            "report_generated_on": "datetime", "terminal_start": "str",
            "terminal_end": "str", "turn": "str", "document_id": "str",
            "document_amount": "float", "tip": "float", "document_total": "float",
        },
        obligatorias=["document_id", "tip"],
        estructura="informe_metadatos",
        notas="Informe de propinas por documento/ticket, mismo estilo de "
              "cabecera de metadatos que 'tickets'.",
    ),
    "articulos": EntitySchema(
        columnas={
            "article_code": "Int64", "article_name": "str",
            "article_short_name": "str", "department_code": "Int64",
        },
        obligatorias=["article_code", "article_name", "department_code"],
        estructura="tabla_simple",
        notas="Catálogo maestro de platos/artículos con su departamento. "
              "Una fila por artículo.",
    ),
    "departamentos": EntitySchema(
        columnas={
            "department_code": "Int64", "department_name": "str",
            "department_short_name": "str",
        },
        obligatorias=["department_code", "department_name"],
        estructura="tabla_simple",
        notas="Catálogo maestro de departamentos/categorías de carta "
              "(ENTRANTES, CARNES, POSTRES...).",
    ),
    "menu": EntitySchema(
        columnas={"article_code": "Int64", "article_name": "str"},
        obligatorias=["article_code", "article_name"],
        estructura="tabla_simple",
        notas="Versión reducida del catálogo de artículos, solo código y "
              "nombre (sin departamento).",
    ),
    "festivos": EntitySchema(
        columnas={
            "fecha": "str", "festivo_nombre": "str", "es_festivo": "int",
            "dia_semana": "str", "nivel": "str",
        },
        obligatorias=["fecha", "festivo_nombre"],
        estructura="passthrough",
        incluye_source_file=False,
        notas="Calendario de festivos. A diferencia de las demás entidades, "
              "NO lleva columna source_file en el destino.",
    ),
    "eventos": EntitySchema(
        columnas={
            "event_id": "str", "fecha_inicio": "str", "fecha_fin": "str",
            "nombre_evento": "str", "categoria": "str", "subcategoria": "str",
            "ambito": "str", "ubicacion": "str", "proximidad_la_roca": "str",
            "impacto_esperado": "str", "intensidad_sugerida": "int",
            "direccion_demanda": "str", "franja_probable": "str",
            "segmento_cliente": "str", "confianza_fecha": "str", "estimado": "int",
            "feature_sugerida": "str", "fuente_tipo": "str", "source_url": "str",
            "notas_modelado": "str",
        },
        obligatorias=["fecha_inicio", "nombre_evento"],
        estructura="passthrough",
        incluye_source_file=False,
        notas="Eventos locales relevantes. El esquema completo tiene 20 "
              "columnas, pero un restaurante nuevo normalmente solo aportará "
              "fecha/nombre/categoría — el resto se deja a null, eso es "
              "correcto y esperado, no un error.",
    ),
    "meteo_diaria": EntitySchema(
        columnas={
            "date": "str", "temperature_2m_mean (°C)": "float",
            "temperature_2m_max (°C)": "float", "temperature_2m_min (°C)": "float",
            "precipitation_sum (mm)": "float", "rain_sum (mm)": "float",
            "precipitation_hours (h)": "float", "wind_speed_10m_max (km/h)": "float",
            "sunshine_duration (s)": "float",
        },
        obligatorias=["date", "temperature_2m_mean (°C)"],
        estructura="serie_temporal",
        incluye_source_file=False,
        notas="Meteorología diaria. Los nombres de columna de destino llevan "
              "la unidad entre paréntesis, cópialos literalmente así.",
    ),
    "meteo_horaria": EntitySchema(
        columnas={
            "datetime": "datetime", "temperature_2m (°C)": "float",
            "weather_code (wmo code)": "int", "rain (mm)": "float",
        },
        obligatorias=["datetime", "temperature_2m (°C)"],
        estructura="serie_temporal",
        incluye_source_file=False,
        notas="Meteorología horaria. Mismo criterio de nombres con unidades.",
    ),
    "facturas": EntitySchema(
        columnas={
            "ticket_id": "str", "archivo_pdf": "str", "ruta_pdf": "str",
            "restaurante": "str", "cif": "str", "telefono": "str", "mesa": "str",
            "fecha": "str", "hora": "str", "base": "float",
            "porcentaje_iva": "float", "iva": "float", "total": "float",
            "efectivo": "float", "tarjeta": "float", "num_items": "int",
            "suma_items": "float", "diferencia_total_vs_items": "float",
            "cuadra_total": "bool",
        },
        obligatorias=["ticket_id", "fecha", "total"],
        estructura="factura",
        incluye_source_file=False,
        notas="Datos extraídos de facturas/tickets en PDF. Tabla plana, sin "
              "bloque de metadatos de cabecera.",
    ),
}


@dataclass
class MappingProposal:
    """Resultado estructurado que el agente produce para un fichero."""

    fichero: str
    entidad: str | None  # None si no se reconoce ninguna entidad
    confianza: float
    fila_inicio_tabla: int
    metadatos: dict[str, str]
    mapeo_columnas: dict[str, str]
    columnas_no_mapeadas: list[str]
    notas: str
    formato_fecha_dia_primero: bool = True  # español por defecto; el agente lo confirma
    aprobado: bool = False  # se marca True solo tras revisión humana


# ---------------------------------------------------------------------------
# Tools deterministas (sin LLM) — el agente solo las invoca, no las escribe
# ---------------------------------------------------------------------------

def _leer_crudo(path: Path, filas: int = FILAS_MUESTRA) -> pd.DataFrame:
    """Lee un fichero sin asumir cabecera ni tipos, para inspección visual."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, header=None, nrows=filas)
    return pd.read_csv(path, header=None, nrows=filas, engine="python", sep=None)


def peek_file(path: str) -> str:
    """
    Devuelve una vista previa cruda (sin asumir cabecera) de un fichero.

    Parameters
    ----------
    path : str
        Ruta al fichero a inspeccionar (.csv, .xls o .xlsx).

    Returns
    -------
    str
        Las primeras filas del fichero en formato texto tabulado, tal cual
        aparecen en el fichero (incluye posibles filas de metadatos antes
        de la tabla real).
    """
    df = _leer_crudo(Path(path))
    return df.to_string(index=True, header=False, max_colwidth=40)


def peek_folder(path: str) -> str:
    """
    Lista los ficheros de una carpeta y muestra una vista previa del primero.

    Útil para la entidad 'ventas', que llega como una carpeta de informes
    semanales en vez de un único fichero.

    Parameters
    ----------
    path : str
        Ruta a la carpeta.

    Returns
    -------
    str
        Listado de ficheros de la carpeta + vista previa del primero.
    """
    carpeta = Path(path)
    ficheros = sorted(
        f.name for f in carpeta.iterdir()
        if f.suffix.lower() in {".csv", ".xls", ".xlsx"}
    )
    if not ficheros:
        return f"La carpeta {path} no contiene ficheros .csv/.xls/.xlsx."
    preview = peek_file(str(carpeta / ficheros[0]))
    return (
        f"Ficheros en la carpeta ({len(ficheros)}): {ficheros}\n\n"
        f"Vista previa del primero ({ficheros[0]}):\n{preview}"
    )


def get_target_schema(entidad: str) -> str:
    """
    Devuelve el contrato de columnas de una entidad de Bronze.

    Parameters
    ----------
    entidad : str
        Nombre de una de las 13 entidades (ver `list_target_entities`).

    Returns
    -------
    str
        Descripción en texto de las columnas, tipos, cuáles son
        obligatorias y notas de contexto.
    """
    schema = TARGET_SCHEMAS.get(entidad)
    if schema is None:
        return f"Entidad desconocida: {entidad}. Usa list_target_entities primero."
    columnas_fmt = "\n".join(f"  - {c} ({t})" for c, t in schema.columnas.items())
    return (
        f"Entidad: {entidad}\n"
        f"Estructura esperada del fichero de origen: {schema.estructura}\n"
        f"Incluye columna source_file en el destino: {schema.incluye_source_file}\n"
        f"Columnas obligatorias: {schema.obligatorias}\n"
        f"Todas las columnas del destino:\n{columnas_fmt}\n"
        f"Notas: {schema.notas}"
    )


def list_target_entities() -> str:
    """
    Lista las 13 entidades de Bronze reconocidas por el proyecto.

    Returns
    -------
    str
        Nombre de cada entidad y su nota de contexto en una línea.
    """
    return "\n".join(
        f"- {nombre}: {schema.notas}" for nombre, schema in TARGET_SCHEMAS.items()
    )


_TOOLS_LECTURA = [
    {
        "name": "list_target_entities",
        "description": "Lista las 13 entidades de Bronze posibles y qué representa cada una.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "peek_file",
        "description": "Muestra las primeras filas crudas de un fichero (.csv/.xls/.xlsx), "
                        "sin asumir dónde está la cabecera real.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Ruta al fichero."}},
            "required": ["path"],
        },
    },
    {
        "name": "peek_folder",
        "description": "Lista los ficheros de una carpeta y muestra una vista previa del "
                        "primero. Úsala si sospechas que estás ante la entidad 'ventas', "
                        "que llega como carpeta de informes semanales.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Ruta a la carpeta."}},
            "required": ["path"],
        },
    },
    {
        "name": "get_target_schema",
        "description": "Devuelve el contrato de columnas y notas de una entidad concreta.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entidad": {
                    "type": "string",
                    "enum": list(TARGET_SCHEMAS.keys()),
                }
            },
            "required": ["entidad"],
        },
    },
]

_TOOL_RESULTADO = {
    "name": "registrar_resultado",
    "description": "Registra el resultado final del análisis de un fichero. "
                    "Llama a esta tool UNA sola vez, cuando ya tengas toda la información "
                    "necesaria (tras usar las tools de lectura las veces que haga falta).",
    "input_schema": {
        "type": "object",
        "properties": {
            "entidad": {
                "type": ["string", "null"],
                "enum": list(TARGET_SCHEMAS.keys()) + [None],
                "description": "Entidad detectada, o null si el fichero no corresponde "
                                "a ninguna de las 13 entidades conocidas.",
            },
            "confianza": {"type": "number", "minimum": 0, "maximum": 1},
            "fila_inicio_tabla": {
                "type": "integer",
                "description": "Índice (0-based) de la FILA QUE CONTIENE LOS NOMBRES DE "
                                "COLUMNA (la cabecera), no la primera fila de datos. Si la "
                                "cabecera está en la fila 0, este valor es 0, no 1.",
            },
            "formato_fecha_dia_primero": {
                "type": "boolean",
                "description": "true si las fechas del origen van en formato dia/mes/año, "
                                "false si van en mes/dia/año. Obligatorio si el mapeo incluye "
                                "alguna columna de fecha.",
            },
            "metadatos": {
                "type": "object",
                "description": "Solo para estructura 'informe_metadatos': "
                                "report_start, report_end, terminal_start, terminal_end, "
                                "turn, si se detectan en las filas de cabecera.",
            },
            "mapeo_columnas": {
                "type": "object",
                "description": "Columna de origen (tal cual aparece en el fichero) -> "
                                "columna de destino del esquema Bronze.",
            },
            "columnas_no_mapeadas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columnas obligatorias del destino para las que NO se "
                                "encontró correspondencia en el origen.",
            },
            "notas": {"type": "string"},
        },
        "required": ["entidad", "confianza", "mapeo_columnas", "notas"],
    },
}

_TOOL_IMPL = {
    "list_target_entities": lambda **kw: list_target_entities(),
    "peek_file": lambda **kw: peek_file(kw["path"]),
    "peek_folder": lambda **kw: peek_folder(kw["path"]),
    "get_target_schema": lambda **kw: get_target_schema(kw["entidad"]),
}


_SYSTEM_PROMPT = """\
Eres un agente que identifica a qué entidad de datos de un restaurante \
corresponde un fichero desconocido, y propone cómo mapear sus columnas al \
esquema Bronze del proyecto.

Sigue este proceso:
1. Usa peek_file (o peek_folder si parece una carpeta de informes semanales) \
para ver el contenido crudo del fichero.
2. Usa list_target_entities si no tienes claro qué entidades existen.
3. Cuando tengas una hipótesis de qué entidad es, usa get_target_schema para \
confirmar las columnas y la estructura esperada.
4. Si el fichero tiene filas de metadatos antes de la tabla real (rangos de \
fecha, terminal, turno...), identifica en qué fila empieza la tabla y extrae \
esos metadatos tú mismo del texto, sea cual sea el idioma o la etiqueta \
exacta que use el fichero.
5. Si una columna obligatoria del destino no tiene ninguna correspondencia \
razonable en el origen, no la inventes: decláralo en columnas_no_mapeadas.
8. Si el fichero no encaja con ninguna de las 13 entidades (por ejemplo, \
notas internas, backups sin relación, o cualquier cosa ajena al dominio del \
restaurante), llama a registrar_resultado con entidad=null y explica por qué \
en las notas. Es preferible decir "no lo reconozco" a forzar un mapeo falso.
9. Llama a registrar_resultado UNA sola vez, al final, con tu conclusión.

No inventes valores de columnas que no puedas justificar con lo que has \
visto en el fichero.
"""


def analizar_fichero(
    path: str,
    cliente: anthropic.Anthropic | None = None,
    modelo: str = MODELO_POR_DEFECTO,
    max_iteraciones: int = 8,
) -> MappingProposal:
    """
    Ejecuta el bucle del agente sobre un fichero (o carpeta) y devuelve su
    propuesta de mapeo SIN aplicarla todavía.

    Parameters
    ----------
    path : str
        Ruta al fichero o carpeta a analizar.
    cliente : anthropic.Anthropic, optional
        Cliente ya inicializado. Si no se pasa, se crea uno leyendo
        ANTHROPIC_API_KEY del entorno.
    modelo : str, default=MODELO_POR_DEFECTO
        Modelo a usar. Sube a "claude-sonnet-5" si el modelo por defecto
        falla en casos ambiguos.
    max_iteraciones : int, default=8
        Límite de ciclos de tool-use, por seguridad (evita bucles infinitos
        si el modelo no termina de decidirse).

    Returns
    -------
    MappingProposal
        Propuesta sin confirmar (`aprobado=False`). Debe pasar por
        `revisar_y_confirmar` antes de ejecutarse.
    """
    cliente = cliente or anthropic.Anthropic()  # lee ANTHROPIC_API_KEY del entorno

    mensajes: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": f"Analiza este fichero y determina a qué entidad corresponde: {path}",
        }
    ]

    for _ in range(max_iteraciones):
        respuesta = cliente.messages.create(
            model=modelo,
            max_tokens=1500,
            system=_SYSTEM_PROMPT,
            tools=_TOOLS_LECTURA + [_TOOL_RESULTADO],
            messages=mensajes,
        )
        mensajes.append({"role": "assistant", "content": respuesta.content})

        bloques_tool = [b for b in respuesta.content if b.type == "tool_use"]
        if not bloques_tool:
            break  # el modelo respondió en texto sin usar ninguna tool; no debería pasar

        resultados_tool = []
        propuesta_final = None
        for bloque in bloques_tool:
            if bloque.name == "registrar_resultado":
                propuesta_final = bloque.input
                resultados_tool.append({
                    "type": "tool_result", "tool_use_id": bloque.id,
                    "content": "Resultado registrado.",
                })
            else:
                salida = _TOOL_IMPL[bloque.name](**bloque.input)
                resultados_tool.append({
                    "type": "tool_result", "tool_use_id": bloque.id, "content": salida,
                })

        mensajes.append({"role": "user", "content": resultados_tool})

        if propuesta_final is not None:
            return MappingProposal(
                fichero=path,
                entidad=propuesta_final.get("entidad"),
                confianza=float(propuesta_final.get("confianza", 0)),
                fila_inicio_tabla=int(propuesta_final.get("fila_inicio_tabla", 0)),
                metadatos=propuesta_final.get("metadatos", {}) or {},
                mapeo_columnas=propuesta_final.get("mapeo_columnas", {}) or {},
                columnas_no_mapeadas=propuesta_final.get("columnas_no_mapeadas", []) or [],
                notas=propuesta_final.get("notas", ""),
                formato_fecha_dia_primero=propuesta_final.get("formato_fecha_dia_primero", True),
            )

    # Se agotaron las iteraciones sin resultado -> tratar como no reconocido,
    # nunca forzar un mapeo a medias.
    return MappingProposal(
        fichero=path, entidad=None, confianza=0.0, fila_inicio_tabla=0,
        metadatos={}, mapeo_columnas={}, columnas_no_mapeadas=[],
        notas="El agente no llegó a una conclusión en el número de "
              "iteraciones permitido; requiere revisión manual.",
    )


def _cabecera_real(path: Path, fila_inicio_tabla: int) -> list[str]:
    """Lee la fila que el agente dice que es la cabecera y la devuelve tal
    cual, para poder validar el mapeo propuesto contra la realidad del
    fichero antes de que un humano lo apruebe a ciegas."""
    if path.is_dir():
        ficheros = sorted(
            f for f in path.iterdir() if f.suffix.lower() in {".csv", ".xls", ".xlsx"}
        )
        path = ficheros[0]
    raw = _leer_crudo(path, filas=fila_inicio_tabla + 1)
    return [str(c) for c in raw.iloc[fila_inicio_tabla].tolist()]


def revisar_y_confirmar(
    propuestas: list[MappingProposal],
    auto_aprobar_umbral: float | None = None,
) -> list[MappingProposal]:
    """
    Muestra las propuestas del agente para revisión humana antes de ejecutar
    nada. Este paso es obligatorio: `construir_bronze` no procesa ninguna
    propuesta con `aprobado=False`.

    Valida, ADEMÁS, que las claves de `mapeo_columnas` existan literalmente
    en la cabecera real del fichero (según `fila_inicio_tabla`). Si alguna
    clave no coincide byte a byte, se avisa en rojo antes de preguntar si
    se aprueba — un mapeo con claves que no casan produce columnas 100%
    nulas de forma silenciosa si se aprueba sin revisar esto.

    Parameters
    ----------
    propuestas : list[MappingProposal]
        Salida de `analizar_fichero` para uno o varios ficheros.
    auto_aprobar_umbral : float, optional
        Si se indica, las propuestas con confianza >= umbral Y sin
        discrepancias de cabecera se marcan como aprobadas automáticamente;
        el resto queda pendiente de revisión manual. Si es None (por
        defecto), TODO requiere revisión manual explícita.

    Returns
    -------
    list[MappingProposal]
        Las mismas propuestas, con `aprobado` actualizado.
    """
    for p in propuestas:
        print(f"\n{'='*70}\nFichero: {p.fichero}")
        print(f"Entidad detectada: {p.entidad}  (confianza: {p.confianza:.2f})")
        if p.entidad is None:
            print(f"No reconocido. Motivo: {p.notas}")
            continue
        print(f"Fila de inicio de la tabla: {p.fila_inicio_tabla}")
        print(f"Formato de fecha día-primero: {p.formato_fecha_dia_primero}")
        if p.metadatos:
            print(f"Metadatos detectados: {p.metadatos}")

        try:
            cabecera_real = _cabecera_real(Path(p.fichero), p.fila_inicio_tabla)
        except Exception as exc:  # no bloquear la revisión si esto falla
            cabecera_real = None
            print(f"(No se pudo releer la cabecera para validar: {exc})")

        claves_no_encontradas = []
        if cabecera_real is not None:
            claves_no_encontradas = [
                k for k in p.mapeo_columnas if k not in cabecera_real
            ]

        print("Mapeo de columnas propuesto:")
        for origen, destino in p.mapeo_columnas.items():
            marca = " <-- NO EXISTE EN LA CABECERA REAL" if origen in claves_no_encontradas else ""
            print(f"   {origen!r:35s} -> {destino}{marca}")

        if claves_no_encontradas:
            print(f"\n*** ATENCIÓN: {len(claves_no_encontradas)} clave(s) del mapeo no "
                  f"coinciden con ninguna columna real de la cabecera detectada. "
                  f"Si apruebas esto, esas columnas de destino saldrán vacías. ***")
            print(f"Cabecera real leída: {cabecera_real}")

        if p.columnas_no_mapeadas:
            print(f"Columnas obligatorias SIN correspondencia: {p.columnas_no_mapeadas}")
        print(f"Notas del agente: {p.notas}")

        if (
            auto_aprobar_umbral is not None
            and p.confianza >= auto_aprobar_umbral
            and not claves_no_encontradas
        ):
            p.aprobado = True
            print(">> Aprobado automáticamente (confianza suficiente y cabecera coincide).")
            continue

        respuesta = input("¿Aprobar este mapeo? [y/N]: ").strip().lower()
        p.aprobado = respuesta == "y"

    return propuestas


def aplicar_mapeo(propuesta: MappingProposal) -> pd.DataFrame:
    """
    Ejecuta de forma determinista (sin LLM) el mapeo ya aprobado por un
    humano, y devuelve el DataFrame en el esquema Bronze de la entidad.

    Parameters
    ----------
    propuesta : MappingProposal
        Debe tener `aprobado=True`; si no, se lanza ValueError.

    Returns
    -------
    pandas.DataFrame
        Datos transformados, con las columnas del esquema Bronze de la
        entidad (las que no tengan correspondencia quedan a NA).
    """
    if not propuesta.aprobado:
        raise ValueError(
            f"La propuesta para {propuesta.fichero} no está aprobada. "
            "Pasa primero por revisar_y_confirmar."
        )

    schema = TARGET_SCHEMAS[propuesta.entidad]
    path = Path(propuesta.fichero)

    if path.is_dir():
        # Entidad tipo 'ventas': concatenar todos los ficheros de la carpeta
        # aplicando el mismo mapeo aprobado a cada uno.
        ficheros = sorted(
            f for f in path.iterdir() if f.suffix.lower() in {".csv", ".xls", ".xlsx"}
        )
        piezas = [
            _mapear_fichero_individual(f, propuesta, schema) for f in ficheros
        ]
        resultado = pd.concat(piezas, ignore_index=True)
    else:
        resultado = _mapear_fichero_individual(path, propuesta, schema)

    return _coercionar_tipos(resultado, schema, dayfirst=propuesta.formato_fecha_dia_primero)


def _mapear_fichero_individual(
    path: Path, propuesta: MappingProposal, schema: EntitySchema,
) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None)
    else:
        raw = pd.read_csv(path, header=None, engine="python", sep=None)

    fila = _corregir_fila_cabecera(raw, propuesta.fila_inicio_tabla, propuesta.mapeo_columnas)

    columnas_origen = raw.iloc[fila].tolist()
    datos = raw.iloc[fila + 1:].copy()
    datos.columns = columnas_origen
    datos = datos.rename(columns=propuesta.mapeo_columnas)

    columnas_destino = list(schema.columnas.keys())
    for col in columnas_destino:
        if col not in datos.columns:
            datos[col] = pd.NA

    if schema.incluye_source_file:
        datos.insert(0, "source_file", path.name)
        columnas_destino = ["source_file"] + columnas_destino

    for campo, valor in propuesta.metadatos.items():
        if campo in datos.columns:
            datos[campo] = valor

    return datos[columnas_destino].reset_index(drop=True)


def _corregir_fila_cabecera(
    raw: pd.DataFrame, fila_propuesta: int, mapeo_columnas: dict[str, str],
) -> int:
    """
    Salvaguarda determinista contra el error más frecuente del agente: usar
    la primera fila de DATOS en vez de la fila de CABECERA como
    `fila_inicio_tabla`, pese a la aclaración del prompt. Se detecta porque,
    si la fila elegida es la correcta, sus valores deberían coincidir con
    las claves de `mapeo_columnas`; si no coincide ninguna, se prueba con
    la fila de arriba y la de abajo y se usa la que mejor encaje.
    """
    origen_esperado = {str(c).strip().lower() for c in mapeo_columnas.keys()}
    if not origen_esperado:
        return fila_propuesta

    def _solape(idx: int) -> int:
        if idx < 0 or idx >= len(raw):
            return -1
        fila_valores = {str(v).strip().lower() for v in raw.iloc[idx].tolist()}
        return len(origen_esperado & fila_valores)

    candidatas = [fila_propuesta, fila_propuesta - 1, fila_propuesta + 1]
    solapes = [(idx, _solape(idx)) for idx in candidatas if _solape(idx) >= 0]
    mejor_fila, mejor_solape = max(solapes, key=lambda x: x[1])

    if mejor_fila != fila_propuesta and mejor_solape > 0:
        print(
            f"  [auto-corrección] fila_inicio_tabla propuesta={fila_propuesta} no "
            f"coincidía con ninguna columna del mapeo; usando fila={mejor_fila} "
            f"({mejor_solape} coincidencias)."
        )
        return mejor_fila
    return fila_propuesta


def _parsear_fecha_robusto(serie: pd.Series, dayfirst_sugerido: bool) -> pd.Series:
    """
    Convierte una columna a datetime probando dayfirst=True y dayfirst=False,
    y se queda con el que produzca menos valores nulos.

    Es necesario porque `dayfirst` no es un ajuste neutro: aplicado a un
    formato ISO (año-mes-día, ya sin ambigüedad) puede ROMPER filas válidas
    en vez de ignorarlas — p. ej. "2025-01-13" con dayfirst=True puede
    interpretarse como día=01/mes=13, un mes imposible, y descartarse como
    nulo. Probar ambas y quedarnos con la de menos nulos es más fiable que
    fiarnos de un único valor, venga de donde venga.
    """
    opcion_a = pd.to_datetime(serie, errors="coerce", dayfirst=dayfirst_sugerido)
    opcion_b = pd.to_datetime(serie, errors="coerce", dayfirst=not dayfirst_sugerido)
    nulos_a, nulos_b = opcion_a.isna().sum(), opcion_b.isna().sum()
    return opcion_a if nulos_a <= nulos_b else opcion_b


def _coercionar_tipos(
    df: pd.DataFrame, schema: EntitySchema, dayfirst: bool = True,
) -> pd.DataFrame:
    """Aplica los dtypes esperados con errors='coerce' (nunca lanza excepción
    por un valor que no convierte; lo deja como nulo y sigue).

    Parameters
    ----------
    dayfirst : bool, default=True
        Si las columnas de fecha están en formato día/mes/año (True, el caso
        más común en datos de restaurantes españoles) o mes/día/año (False).
        Viene de lo que el propio agente detectó para ese fichero
        (`propuesta.formato_fecha_dia_primero`), no es una suposición fija.
    """
    df = df.copy()
    for columna, tipo in schema.columnas.items():
        if columna not in df.columns:
            continue
        if tipo == "datetime":
            df[columna] = _parsear_fecha_robusto(df[columna], dayfirst_sugerido=dayfirst)
        elif tipo == "float":
            df[columna] = pd.to_numeric(df[columna], errors="coerce").astype("float64")
        elif tipo == "Int64":
            df[columna] = pd.to_numeric(df[columna], errors="coerce").astype("Int64")
        elif tipo == "int":
            df[columna] = pd.to_numeric(df[columna], errors="coerce").astype("Int64")
        elif tipo == "str":
            df[columna] = df[columna].astype("string")
        elif tipo == "bool":
            mapa_bool = {
                True: True, False: False, "true": True, "false": False,
                "True": True, "False": False, "1": True, "0": False,
                1: True, 0: False, "sí": True, "si": True, "no": False,
            }
            df[columna] = df[columna].map(mapa_bool).astype("boolean")
    return df


def construir_bronze(
    ruta_entrada: str,
    ruta_salida: str,
    modelo: str = MODELO_POR_DEFECTO,
    auto_aprobar_umbral: float | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Orquesta el pipeline completo: analiza cada fichero/carpeta de
    `ruta_entrada`, pide confirmación humana, y escribe los parquet de
    Bronze resultantes en `ruta_salida`.

    Parameters
    ----------
    ruta_entrada : str
        Carpeta con los ficheros del restaurante nuevo (nivel superior:
        ficheros sueltos y, opcionalmente, subcarpetas tipo 'ventas').
    ruta_salida : str
        Carpeta donde escribir los `<entidad>_raw.parquet` resultantes.
    modelo : str, default=MODELO_POR_DEFECTO
        Modelo de Claude a usar.
    auto_aprobar_umbral : float, optional
        Ver `revisar_y_confirmar`. None = revisión manual de todo
        (recomendado hasta validar el agente con casos reales).

    Returns
    -------
    dict[str, pandas.DataFrame]
        Un DataFrame por entidad reconocida y aprobada, ya en esquema Bronze.
    """
    entrada = Path(ruta_entrada)
    salida = Path(ruta_salida)
    salida.mkdir(parents=True, exist_ok=True)

    objetivos = [
        f for f in entrada.iterdir()
        if f.is_dir() or f.suffix.lower() in {".csv", ".xls", ".xlsx"}
    ]

    cliente = anthropic.Anthropic()
    propuestas = []
    for i, obj in enumerate(objetivos, start=1):
        print(f"[{i}/{len(objetivos)}] Analizando {obj.name} ...")
        propuestas.append(analizar_fichero(str(obj), cliente=cliente, modelo=modelo))
    propuestas = revisar_y_confirmar(propuestas, auto_aprobar_umbral=auto_aprobar_umbral)

    resultado: dict[str, pd.DataFrame] = {}
    for p in propuestas:
        if not p.aprobado or p.entidad is None:
            continue
        df = aplicar_mapeo(p)
        if p.entidad in resultado:
            resultado[p.entidad] = pd.concat([resultado[p.entidad], df], ignore_index=True)
        else:
            resultado[p.entidad] = df

    for entidad, df in resultado.items():
        df.to_parquet(salida / f"{entidad}_raw.parquet", index=False)
        print(f"Escrito {entidad}_raw.parquet ({len(df)} filas)")

    no_reconocidos = [p.fichero for p in propuestas if p.entidad is None]
    if no_reconocidos:
        print(f"\nFicheros no reconocidos (revisar a mano): {no_reconocidos}")

    return resultado


def _elegir_carpeta_entrada() -> str:
    """
    Abre un selector nativo de carpeta (Finder en Mac, Explorador en
    Windows, vía tkinter) y devuelve la ruta elegida por el usuario.
 
    Esta función es deliberadamente independiente de `construir_bronze`:
    esta última sigue aceptando `ruta_entrada` como string normal, para que
    el día de mañana un backend (SaaS, API) pueda llamarla directamente con
    la ruta de un fichero recién subido, sin pasar por ningún diálogo
    gráfico. El selector solo tiene sentido para uso local e interactivo,
    como el bloque `if __name__ == "__main__":` de abajo.
 
    Si no hay entorno gráfico disponible (tkinter no instalado, o un
    servidor sin pantalla), recurre a pedir la ruta por teclado en vez de
    fallar.
 
    Returns
    -------
    str
        Ruta absoluta de la carpeta elegida.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
 
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)  # evita que el diálogo se abra detrás de otras ventanas
        carpeta = filedialog.askdirectory(
            title="Selecciona la carpeta con los datos del restaurante"
        )
        root.destroy()
    except SystemExit:
        raise
    except Exception:
        print(
            "No se pudo abrir el selector gráfico de carpetas (¿tkinter no "
            "disponible, o entorno sin pantalla?). Introduce la ruta a mano."
        )
        carpeta = input("Ruta de la carpeta de entrada: ").strip()
 
    if not carpeta:
        raise SystemExit("No se seleccionó ninguna carpeta. Abortando.")
    return carpeta
 
 
if __name__ == "__main__":
    # Demo mínima de uso manual.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "Define ANTHROPIC_API_KEY en el entorno antes de ejecutar este módulo."
        )

    print(f"Selecciona la carpeta donde se encuentran tus ficheros:")
    ruta_entrada = _elegir_carpeta_entrada()
    print(f"Carpeta de entrada seleccionada: {ruta_entrada}")
 
    # La salida SÍ se deja fija: pensando en una futura versión SaaS, el
    # destino de los Bronze generados no debería depender de una elección
    # manual del usuario en cada ejecución.
    construir_bronze(
        ruta_entrada=ruta_entrada,
        ruta_salida="agente/prueba2_bronze",
    )
