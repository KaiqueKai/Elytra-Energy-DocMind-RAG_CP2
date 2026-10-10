"""Embeddings: OllamaEmbeddings(model="nomic-embed-text") — único modelo aprovado no PDF.

Estratégia (modo `auto`, padrão):
1. Tenta o endpoint de embeddings do Ollama Cloud (https://ollama.com/api/embed)
   com a chave OLLAMA_API_KEY.
2. Se o Cloud não atender (HTTP 401/403/404 — o que aconteceu no desenvolvimento),
   usa o servidor Ollama do runtime (Colab) ou da máquina, em OLLAMA_EMBEDDING_HOST.

O modelo, a classe e as chamadas (embed_query / embed_documents) são sempre os do enunciado.
"""
from __future__ import annotations

from functools import lru_cache

import requests
from langchain_ollama import OllamaEmbeddings

from .config import (
    EMBEDDING_LOCAL_HOST, EMBEDDING_MODE, EMBEDDING_MODEL,
    OLLAMA_API_KEY, OLLAMA_CLOUD_HOST,
)


def _cabecalhos_cloud() -> dict[str, str]:
    return {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}


def _cloud_atende_embeddings() -> tuple[bool, str]:
    """Testa uma chamada real de embedding no Ollama Cloud. Devolve (ok, detalhe)."""
    if not OLLAMA_API_KEY:
        return False, "OLLAMA_API_KEY ausente"
    try:
        resp = requests.post(
            f"{OLLAMA_CLOUD_HOST}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": "teste de embedding"},
            headers=_cabecalhos_cloud(), timeout=30,
        )
    except requests.RequestException as exc:
        return False, f"erro de rede: {exc}"
    if resp.status_code == 200 and resp.json().get("embeddings"):
        return True, "HTTP 200"
    return False, f"HTTP {resp.status_code}"


def _local_disponivel() -> tuple[bool, str]:
    """Verifica se há Ollama local/Colab com o nomic-embed-text baixado."""
    try:
        resp = requests.get(f"{EMBEDDING_LOCAL_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return False, f"Ollama não responde em {EMBEDDING_LOCAL_HOST} ({exc})"
    nomes = [m.get("name", "") for m in resp.json().get("models", [])]
    if not any(n.split(":")[0] == EMBEDDING_MODEL for n in nomes):
        return False, f"modelo '{EMBEDDING_MODEL}' não baixado; execute `ollama pull {EMBEDDING_MODEL}`"
    return True, "ok"


@lru_cache(maxsize=1)
def modo_embeddings_efetivo() -> str:
    """Decide (uma única vez) entre 'cloud' e 'local' e informa o motivo na tela."""
    if EMBEDDING_MODE == "cloud":
        ok, detalhe = _cloud_atende_embeddings()
        if not ok:
            raise RuntimeError(f"Embeddings via Ollama Cloud indisponíveis: {detalhe}.")
        return "cloud"
    if EMBEDDING_MODE == "auto":
        ok, detalhe = _cloud_atende_embeddings()
        if ok:
            print("[embeddings] Usando nomic-embed-text no Ollama Cloud.")
            return "cloud"
        print(f"[embeddings] Ollama Cloud não atendeu embeddings ({detalhe}); usando Ollama do runtime.")
    ok, detalhe = _local_disponivel()
    if not ok:
        raise RuntimeError(
            f"Embeddings indisponíveis: {detalhe}. No Colab, execute a célula de preparação; "
            "localmente, inicie o Ollama e rode `ollama pull nomic-embed-text`."
        )
    return "local"


def criar_embeddings() -> OllamaEmbeddings:
    """Retorna OllamaEmbeddings(model='nomic-embed-text') apontando para Cloud ou runtime."""
    if modo_embeddings_efetivo() == "cloud":
        return OllamaEmbeddings(
            model=EMBEDDING_MODEL, base_url=OLLAMA_CLOUD_HOST,
            client_kwargs={"headers": _cabecalhos_cloud()},
        )
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=EMBEDDING_LOCAL_HOST)
