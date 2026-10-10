"""Recuperação semântica (retrieve) com metadata filtering e reranking (diferenciais do PDF)."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import CATEGORIAS, ENABLE_RERANKER, TOP_K
from .reranker import reranquear


def montar_filtro(categoria: str | None = None, instituicao: str | None = None) -> dict | None:
    """Monta o `where` do Chroma a partir dos metadados (categoria e/ou instituição)."""
    condicoes = []
    if categoria and categoria != "Todas":
        if categoria not in CATEGORIAS:
            raise ValueError(f"Categoria não reconhecida: {categoria}. Use uma de {CATEGORIAS}.")
        condicoes.append({"categoria": categoria})
    if instituicao and instituicao != "Todas":
        condicoes.append({"instituicao": instituicao})
    if not condicoes:
        return None
    return condicoes[0] if len(condicoes) == 1 else {"$and": condicoes}


def recuperar(
    vectorstore: Chroma,
    consulta: str,
    categoria: str | None = None,
    instituicao: str | None = None,
    k: int = TOP_K,
    usar_reranking: bool | None = None,
) -> list[Document]:
    """Busca os k chunks mais relevantes; com reranking, busca mais candidatos e reordena."""
    if not consulta.strip():
        raise ValueError("A consulta de recuperação não pode estar vazia.")
    if k < 1:
        raise ValueError("k precisa ser maior que zero.")

    reranking_ativo = ENABLE_RERANKER if usar_reranking is None else usar_reranking
    kwargs: dict = {"k": max(k * 4, 20) if reranking_ativo else k}
    filtro = montar_filtro(categoria, instituicao)
    if filtro:
        kwargs["filter"] = filtro  # o langchain-chroma converte em `where=` do ChromaDB
    candidatos = vectorstore.similarity_search(consulta, **kwargs)
    ordenados = reranquear(consulta, candidatos, habilitado=reranking_ativo)
    return ordenados[:k]
