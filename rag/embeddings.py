"""Vectorizacion de texto para la busqueda del RAG.

Decision de la Fase 3 (ver PROJECT_BRIEF.md seccion 8): con el volumen
actual de conocimiento (~400 platos + ~965 reseñas + 3 documentos), un
vector store con embeddings neuronales (Chroma/FAISS) es mas infraestructura
de la que hace falta. Se usa TF-IDF + similitud coseno (scikit-learn, ya
instalado, sin API key ni modelo pesado que descargar). La interfaz de
`retriever.py` no depende de esto -- si el corpus crece o hace falta mejor
matching semantico, se puede sustituir este modulo por embeddings reales
sin tocar el resto del sistema.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
    )
