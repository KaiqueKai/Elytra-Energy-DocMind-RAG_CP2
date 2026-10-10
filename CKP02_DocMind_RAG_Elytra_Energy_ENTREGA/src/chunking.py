"""Chunking exigido pelo PDF: RecursiveCharacterTextSplitter com os separadores da Aula 06
e chunk_overlap entre 10% e 15% do chunk_size (usamos 12%)."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZES, OVERLAP_RATIO
from .schemas import ChunkRecord

# Separadores exatamente como no enunciado.
SEPARADORES = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int
    chunk_overlap: int


def config_chunking(chunk_size: int) -> ChunkConfig:
    """Devolve a configuração (tamanho + overlap de 12%) para um tamanho válido."""
    if chunk_size not in CHUNK_SIZES:
        raise ValueError(f"Tamanho inválido {chunk_size}. Valores aceitos: {CHUNK_SIZES}.")
    return ChunkConfig(chunk_size=chunk_size, chunk_overlap=round(chunk_size * OVERLAP_RATIO))


def criar_splitter(chunk_size: int) -> RecursiveCharacterTextSplitter:
    config = config_chunking(chunk_size)
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=SEPARADORES,
        length_function=len,
        add_start_index=True,
    )


def dividir_documentos(documentos: list[Document], chunk_size: int) -> list[Document]:
    """Divide as páginas em chunks, completa a metadata e valida arquivo/página via Pydantic."""
    if not documentos:
        raise ValueError("Não há documentos para dividir em chunks.")

    chunks = criar_splitter(chunk_size).split_documents(documentos)
    for indice, chunk in enumerate(chunks):
        meta = chunk.metadata
        meta.setdefault("arquivo", "documento_sem_nome")
        meta.setdefault("fonte", "Fonte não informada")
        meta.setdefault("categoria", "Geral")
        meta.setdefault("tipo", "Desconhecido")
        meta.setdefault("data", "Não informada")
        try:
            pagina = max(1, int(meta.get("pagina", 1)))
        except (TypeError, ValueError):
            pagina = 1
        meta["pagina"] = pagina
        meta["chunk_indice"] = indice
        meta["chunk_size_configurado"] = chunk_size
        registro = ChunkRecord(
            id=f"{meta['arquivo']}:{pagina}:{indice}:{chunk_size}",
            conteudo=chunk.page_content,
            arquivo=str(meta["arquivo"]),
            pagina=pagina,
            categoria=str(meta["categoria"]),
            fonte=str(meta["fonte"]),
        )
        meta["chunk_id"] = registro.id
    return chunks
