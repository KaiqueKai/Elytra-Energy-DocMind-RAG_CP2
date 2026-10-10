"""ChromaDB persistente com cache retomável dos embeddings por lote no Google Drive."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import chromadb
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import CHROMA_DIR, EMBEDDING_MODEL, collection_name
from .embeddings import criar_embeddings

TAMANHO_LOTE = 96  # mais eficiência por chamada; reduz overhead comparado a lotes pequenos


def _nomes_colecoes(client) -> set[str]:
    return {item.name if hasattr(item, "name") else str(item) for item in client.list_collections()}


def _ids_deterministicos(chunks: list[Document]) -> list[str]:
    ids = []
    for indice, doc in enumerate(chunks):
        bruto = "|".join([
            str(doc.metadata.get("arquivo", "")),
            str(doc.metadata.get("pagina", 1)),
            str(indice),
            doc.page_content,
        ])
        ids.append(hashlib.sha256(bruto.encode("utf-8")).hexdigest())
    return ids


def _cache_path(chunk_size: int, batch_ids: list[str]) -> Path | None:
    pasta = os.getenv("EMBEDDING_CACHE_DIR", "").strip()
    if not pasta:
        return None
    try:
        destino = Path(pasta)
        destino.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[cache] Não foi possível acessar o cache persistente: {exc}", flush=True)
        return None
    assinatura = hashlib.sha256(
        (f"v2|{EMBEDDING_MODEL}|{chunk_size}|" + "|".join(batch_ids)).encode("utf-8")
    ).hexdigest()
    return destino / f"{chunk_size}_{assinatura}.npz"


def construir_vectorstore(chunks: list[Document], chunk_size: int, substituir: bool = True) -> Chroma:
    """Gera/recupera embeddings por lote e grava a coleção no ChromaDB local."""
    if not chunks:
        raise ValueError("Não é possível criar um índice vazio.")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    nome = collection_name(chunk_size)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existentes = _nomes_colecoes(client)
    embeddings = criar_embeddings()

    if nome in existentes:
        atual = Chroma(collection_name=nome, embedding_function=embeddings,
                       persist_directory=str(CHROMA_DIR))
        if not substituir and atual._collection.count() == len(chunks):
            print(f"[reutilizado] {nome}: {len(chunks)} chunks já indexados.", flush=True)
            return atual
        client.delete_collection(name=nome)

    store = Chroma(collection_name=nome, embedding_function=embeddings,
                   persist_directory=str(CHROMA_DIR))
    ids = _ids_deterministicos(chunks)
    total = len(chunks)
    lotes = (total + TAMANHO_LOTE - 1) // TAMANHO_LOTE
    hits_cache = 0
    inicio_total = time.time()

    for numero_lote, inicio in enumerate(range(0, total, TAMANHO_LOTE), start=1):
        fim = min(inicio + TAMANHO_LOTE, total)
        lote = chunks[inicio:fim]
        lote_ids = ids[inicio:fim]
        textos = [doc.page_content for doc in lote]
        metadados = [doc.metadata for doc in lote]
        arquivo_cache = _cache_path(chunk_size, lote_ids)
        vetores = None

        usou_cache = False
        if arquivo_cache is not None and arquivo_cache.is_file():
            try:
                with np.load(arquivo_cache) as salvo:
                    matriz = salvo["embeddings"]
                if matriz.shape[0] == len(lote):
                    vetores = matriz.tolist()
                    hits_cache += 1
                    usou_cache = True
            except Exception:
                vetores = None  # cache corrompido: calcula novamente

        if vetores is None:
            vetores = embeddings.embed_documents(textos)
            if arquivo_cache is not None:
                try:
                    temporario = arquivo_cache.with_name(arquivo_cache.stem + ".tmp.npz")
                    with temporario.open("wb") as f:
                        np.savez_compressed(f, embeddings=np.asarray(vetores, dtype=np.float32))
                    os.replace(temporario, arquivo_cache)
                except OSError as exc:
                    print(f"[cache] Falha ao salvar lote (o índice segue normalmente): {exc}", flush=True)

        store._collection.add(ids=lote_ids, embeddings=vetores,
                              documents=textos, metadatas=metadados)
        elapsed = time.time() - inicio_total
        origem = "cache" if usou_cache else "embedding calculado"
        print(f"  [{nome}] {fim}/{total} chunks | lote {numero_lote}/{lotes} | cache usados: {hits_cache}/{numero_lote} | {elapsed/60:.1f} min | {origem}", flush=True)

    print(f"[concluído] {nome}: {store._collection.count()} registros em {(time.time()-inicio_total)/60:.1f} min.", flush=True)
    return store


def carregar_vectorstore(chunk_size: int) -> Chroma:
    """Abre coleção pronta; falha com mensagem clara se ela não existir ou estiver vazia."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    nome = collection_name(chunk_size)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if nome not in _nomes_colecoes(client):
        raise FileNotFoundError(
            f"Índice '{nome}' não existe. Execute `python -m scripts.build_index --chunk-size {chunk_size}`."
        )
    store = Chroma(collection_name=nome, embedding_function=criar_embeddings(),
                   persist_directory=str(CHROMA_DIR))
    if store._collection.count() == 0:
        raise ValueError(f"A coleção '{nome}' existe, mas está vazia. Reconstrua o índice.")
    return store
