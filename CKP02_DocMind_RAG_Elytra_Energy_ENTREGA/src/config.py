"""Configuração central do DocMind RAG — Elytra Energy (CKP02).

Todas as constantes do projeto ficam aqui para que o restante do código seja
modular e para que o pipeline possa ser reutilizado como @tool no CKP03.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ----------------------------------------------------------------------------
# Pastas e arquivos
# ----------------------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data" / "originals"
SOURCES_FILE = ROOT_DIR / "data" / "fontes.json"
QUESTIONS_FILE = ROOT_DIR / "evals" / "perguntas.json"
CHROMA_DIR = ROOT_DIR / "chroma_db"
REPORTS_DIR = ROOT_DIR / "reports"
PROMPT_FILE = ROOT_DIR / "prompts" / "system_prompt.md"

# ----------------------------------------------------------------------------
# Modelo de chat — exigência do PDF: exclusivamente gemma4:cloud (Ollama Cloud)
# ----------------------------------------------------------------------------
# "gemma4:cloud" é o nome pedido no enunciado. A API HTTP direta do Ollama Cloud
# usa o nome devolvido por https://ollama.com/api/tags (hoje "gemma4:31b").
# O módulo generator.py resolve o nome automaticamente; veja docs/DECISOES_TECNICAS.md.
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:cloud").strip()
OLLAMA_CLOUD_HOST = os.getenv("OLLAMA_CLOUD_HOST", "https://ollama.com").rstrip("/")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()  # Secret; nunca no código
CHAT_TEMPERATURE = 0  # exigido pelo PDF: respostas fundamentadas

# ----------------------------------------------------------------------------
# Embeddings — exigência do PDF: nomic-embed-text com OllamaEmbeddings
# ----------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text").strip()
# auto  -> tenta o endpoint de embeddings do Ollama Cloud; se não houver suporte
#          (HTTP 401/403/404), usa o Ollama do runtime (Colab) ou da máquina.
# cloud -> exige o Ollama Cloud. local -> exige o Ollama em EMBEDDING_LOCAL_HOST.
EMBEDDING_MODE = os.getenv("OLLAMA_EMBEDDING_MODE", "auto").strip().lower()
EMBEDDING_LOCAL_HOST = os.getenv("OLLAMA_EMBEDDING_HOST", "http://127.0.0.1:11434").rstrip("/")

# ----------------------------------------------------------------------------
# RAG
# ----------------------------------------------------------------------------
CHUNK_SIZES = (256, 512, 1024)          # PDF: comparar no mínimo 2 configurações
OVERLAP_RATIO = 0.12                    # PDF: overlap de 10% a 15% do chunk_size
DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").strip().lower() in {"1", "true", "yes", "sim"}
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # citado no PDF (Aula 07)

# Nome da coleção = nome do domínio (exigência do PDF) + tamanho do chunk.
COLLECTION_PREFIX = "eficiencia_energetica_residencial"

CATEGORIAS = [
    "Climatização", "Refrigeração", "Ventilação",
    "Residencial", "Indicadores energéticos",
]


def collection_name(chunk_size: int) -> str:
    """Devolve o nome da coleção ChromaDB para cada configuração de chunking."""
    if chunk_size not in CHUNK_SIZES:
        raise ValueError(f"chunk_size inválido: {chunk_size}. Use um de {CHUNK_SIZES}.")
    return f"{COLLECTION_PREFIX}_{chunk_size}"
