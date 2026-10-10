"""Carrega a base, divide em chunks e grava os índices ChromaDB (uma coleção por chunk size)."""
from __future__ import annotations

import argparse

from src.chunking import config_chunking, dividir_documentos
from src.config import CHUNK_SIZES
from src.loaders import carregar_documentos
from src.vectorstore import construir_vectorstore, carregar_vectorstore


def construir(chunk_size: int, documentos=None) -> int:
    documentos = documentos if documentos is not None else carregar_documentos()
    config = config_chunking(chunk_size)
    chunks = dividir_documentos(documentos, chunk_size)
    try:
        store = carregar_vectorstore(chunk_size)
        if store._collection.count() != len(chunks):
            raise ValueError("Índice incompleto")
        print(f"[REUTILIZADO] chunk_size={chunk_size}: {len(chunks)} chunks.", flush=True)
    except (FileNotFoundError, ValueError):
        store = construir_vectorstore(chunks, chunk_size, substituir=True)
    print(f"chunk_size={chunk_size} | overlap={config.chunk_overlap} ({config.chunk_overlap / chunk_size:.0%}) "
          f"| páginas: {len(documentos)} | chunks: {len(chunks)} | coleção: {store._collection.name}")
    return len(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, choices=CHUNK_SIZES, default=512)
    parser.add_argument("--all", action="store_true", help="Constrói as coleções 256, 512 e 1024.")
    args = parser.parse_args()
    documentos = carregar_documentos()
    tamanhos = CHUNK_SIZES if args.all else (args.chunk_size,)
    total = sum(construir(t, documentos) for t in tamanhos)
    print(f"\nConcluído. Total de chunks indexados: {total}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
