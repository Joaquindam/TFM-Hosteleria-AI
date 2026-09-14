"""Busqueda del RAG -- ver PROJECT_BRIEF.md seccion 8.

`retrieve_context` es la unica funcion que el resto del sistema (MCP, LLM app)
deberia usar. Nunca devuelve cifras ni calculos: solo texto ya escrito
(documentos de `knowledge/` o reseñas reales), para que el LLM lo use como
contexto de lectura, no como fuente de numeros.
"""
from __future__ import annotations

from sklearn.metrics.pairwise import cosine_similarity

from rag.knowledge_base import get_knowledge_base

MIN_SCORE = 0.05  # por debajo de esto, el chunk no es relevante de verdad


def retrieve_context(query: str, k: int = 4) -> list[dict]:
    documents, vectorizer, matrix = get_knowledge_base()
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]

    ranked_idx = scores.argsort()[::-1][:k]
    results = []
    for i in ranked_idx:
        if scores[i] < MIN_SCORE:
            continue
        doc = documents[i]
        results.append({
            "text": doc["text"],
            "source": doc["source"],
            "score": round(float(scores[i]), 4),
            **({"review_id": doc["review_id"], "author": doc["author"], "date": doc["date"]} if "review_id" in doc else {}),
        })
    return results
