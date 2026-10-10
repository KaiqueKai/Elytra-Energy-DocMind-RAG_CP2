"""Reranking com cross-encoder (diferencial do PDF: cross-encoder/ms-marco-MiniLM-L-6-v2).

Se o `sentence-transformers` ou o download do modelo falharem, o pipeline
continua sem reranking (e avisa uma única vez) em vez de quebrar a execução.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from .config import RERANKER_MODEL

_falhou = False  # evita repetir o aviso e novas tentativas de download


@lru_cache(maxsize=1)
def _modelo():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL, device="cpu")


def reranquear(consulta: str, documentos: list[Document], habilitado: bool = True) -> list[Document]:
    """Reordena os candidatos pela pontuação do cross-encoder (maior relevância primeiro)."""
    global _falhou
    if not habilitado or _falhou or len(documentos) < 2:
        return documentos
    try:
        pares = [(consulta, doc.page_content[:2000]) for doc in documentos]
        pontuacoes = _modelo().predict(pares, show_progress_bar=False)
    except Exception as exc:  # sem a lib, sem internet para baixar os pesos etc.
        _falhou = True
        print(f"[reranker] Desativado nesta execução ({type(exc).__name__}: {exc}).")
        return documentos
    ordenados = sorted(zip(documentos, pontuacoes), key=lambda par: float(par[1]), reverse=True)
    return [doc for doc, _ in ordenados]
