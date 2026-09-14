"""Construye y cachea en memoria el corpus vectorizado del RAG.

Se reconstruye una vez por proceso (lru_cache): con ~1400 chunks y
TF-IDF esto tarda menos de un segundo, no hace falta persistir un indice
en disco (ver rag/embeddings.py para la justificacion de TF-IDF vs.
vector store real).
"""
from __future__ import annotations

from functools import lru_cache

from rag.embeddings import build_vectorizer
from rag.ingest import load_all_documents


@lru_cache(maxsize=1)
def get_knowledge_base():
    documents = load_all_documents()
    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform([d["text"] for d in documents])
    return documents, vectorizer, matrix


def document_count() -> int:
    documents, _, _ = get_knowledge_base()
    return len(documents)
