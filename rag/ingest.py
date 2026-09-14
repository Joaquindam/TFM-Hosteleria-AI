"""Carga y trocea (chunking) las fuentes de conocimiento del RAG.

Dos tipos de fuente (ver PROJECT_BRIEF.md seccion 7):
- Documentos estaticos en `knowledge/` (carta, descripcion, politicas).
- Reseñas reales ya depuradas en `data/silver/snapshots/resenas_silver.parquet`
  (965 reseñas con texto completo y aspectos positivos/negativos ya
  extraidos -- no se reprocesa nada, solo se reutiliza).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ml.paths import PROJECT_ROOT, SILVER_SNAP

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


def _chunk_markdown(path: Path) -> list[dict]:
    """Trocea un .md por encabezados de nivel 2 (##). El encabezado de nivel
    1 (titulo del documento) se antepone a cada chunk como contexto."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    title = lines[0].lstrip("#").strip() if lines and lines[0].startswith("#") else path.stem

    chunks = []
    current_heading = None
    current_lines: list[str] = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append({
                "text": f"# {title}\n" + (f"## {current_heading}\n" if current_heading else "") + body,
                "source": path.name,
                "heading": current_heading,
            })

    for line in lines[1:]:
        if line.startswith("## "):
            flush()
            current_heading = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    if not chunks:  # documento sin subtitulos: un unico chunk
        chunks = [{"text": text, "source": path.name, "heading": None}]

    return chunks


def load_knowledge_documents() -> list[dict]:
    """Chunks de todos los .md en knowledge/."""
    chunks = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        chunks.extend(_chunk_markdown(path))
    for i, c in enumerate(chunks):
        c["id"] = f"doc:{c['source']}:{i}"
    return chunks


def load_review_documents(limit: int | None = None) -> list[dict]:
    """Un chunk por reseña real (texto + aspectos ya extraidos en Silver)."""
    resenas = pd.read_parquet(SILVER_SNAP / "resenas_silver.parquet")
    if limit:
        resenas = resenas.head(limit)

    chunks = []
    for _, r in resenas.iterrows():
        parts = [f"Reseña de {r['author']} en {r['platform']} ({r['review_date'].date() if pd.notna(r['review_date']) else 'fecha desconocida'}):"]
        if pd.notna(r.get("review_text")):
            parts.append(str(r["review_text"]))
        if pd.notna(r.get("positive_aspects_text")):
            parts.append(f"Aspectos positivos: {r['positive_aspects_text']}")
        if pd.notna(r.get("negative_aspects_text")):
            parts.append(f"Aspectos negativos: {r['negative_aspects_text']}")

        chunks.append({
            "id": f"review:{r['review_id']}",
            "text": "\n".join(parts),
            "source": f"reseña ({r['platform']})",
            "review_id": r["review_id"],
            "author": r["author"],
            "date": r["review_date"].date().isoformat() if pd.notna(r["review_date"]) else None,
        })
    return chunks


def load_all_documents() -> list[dict]:
    return load_knowledge_documents() + load_review_documents()
