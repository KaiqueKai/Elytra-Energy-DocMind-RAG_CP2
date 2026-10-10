"""Diagnóstico rápido: chave, chat (gemma4:cloud) e embeddings (nomic-embed-text)."""
from __future__ import annotations

from src.config import CHAT_MODEL, EMBEDDING_MODE, EMBEDDING_MODEL, OLLAMA_API_KEY, OLLAMA_CLOUD_HOST


def main() -> int:
    print("=== Diagnóstico DocMind RAG ===")
    print(f"Chave OLLAMA_API_KEY: {'CARREGADA' if OLLAMA_API_KEY else 'AUSENTE'}")
    print(f"Chat: {CHAT_MODEL} em {OLLAMA_CLOUD_HOST} | Embeddings: {EMBEDDING_MODEL} (modo {EMBEDDING_MODE})")
    falhas = 0
    try:
        from src.embeddings import criar_embeddings, modo_embeddings_efetivo
        vetor = criar_embeddings().embed_query("teste de eficiência energética residencial")
        print(f"[OK] Embedding de {len(vetor)} dimensões (origem: {modo_embeddings_efetivo()}).")
    except Exception as exc:
        falhas += 1
        print(f"[FALHA] Embeddings: {type(exc).__name__}: {exc}")
    try:
        from src.generator import criar_modelo_chat, resolver_nome_modelo
        modelo = criar_modelo_chat()
        texto = str(modelo.invoke("Responda apenas: conexão OK").content).strip()
        print(f"[OK] Chat '{resolver_nome_modelo()}' respondeu: {texto[:80]}")
    except Exception as exc:
        falhas += 1
        print(f"[FALHA] Chat Ollama Cloud: {type(exc).__name__}: {exc}")
    print("\nDiagnóstico " + ("INCOMPLETO — corrija os pontos acima." if falhas else "concluído com sucesso."))
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
