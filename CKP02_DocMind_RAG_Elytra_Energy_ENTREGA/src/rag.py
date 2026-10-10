"""Pipeline RAG: retrieve → generate, com citação do chunk de origem.

`buscar(consulta)` é a interface pública usada como @tool no CKP03.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from .config import DEFAULT_CHUNK_SIZE, TOP_K
from .generator import criar_modelo_chat, gerar_resposta
from .retriever import recuperar
from .schemas import RAGResponse, SourceCitation
from .vectorstore import carregar_vectorstore


def _contexto(documentos: list[Document]) -> str:
    """Monta o contexto numerado enviado ao LLM ([Trecho N] | arquivo | página)."""
    partes = []
    for i, doc in enumerate(documentos, start=1):
        meta = doc.metadata
        partes.append(
            f"[Trecho {i} | arquivo: {meta.get('arquivo', 'documento sem nome')} "
            f"| página: {int(meta.get('pagina', 1))}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(partes)


def _fontes(documentos: list[Document]) -> list[SourceCitation]:
    """Uma citação por chunk recuperado (arquivo, página, chunk_id, URL de origem)."""
    fontes = []
    for i, doc in enumerate(documentos, start=1):
        meta = doc.metadata
        try:
            pagina = max(1, int(meta.get("pagina", 1)))
        except (TypeError, ValueError):
            pagina = 1
        fontes.append(SourceCitation(
            trecho_numero=i,
            arquivo=str(meta.get("arquivo", "documento sem nome")),
            pagina=pagina,
            fonte=str(meta.get("fonte", "Fonte não informada")),
            categoria=str(meta.get("categoria", "Geral")),
            chunk_id=str(meta.get("chunk_id", "")),
            trecho=doc.page_content[:300],
        ))
    return fontes


class RAGPipeline:
    """Recupera evidências no ChromaDB, gera a resposta e anexa a proveniência."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.vectorstore = carregar_vectorstore(chunk_size)
        self.modelo_chat = criar_modelo_chat()

    def consultar(self, consulta: str, categoria: str | None = None,
                  instituicao: str | None = None) -> RAGResponse:
        pergunta = consulta.strip()
        if not pergunta:
            raise ValueError("Digite uma pergunta para consultar a base.")
        documentos = recuperar(self.vectorstore, pergunta, categoria=categoria,
                               instituicao=instituicao, k=TOP_K)
        if not documentos:
            return RAGResponse(
                consulta=pergunta,
                resposta="Não encontrei trechos relevantes na base para responder com segurança. "
                         "Tente reformular a pergunta ou remover o filtro de categoria.",
                configuracao_chunking=self.chunk_size,
            )
        resposta = gerar_resposta(pergunta, _contexto(documentos), modelo=self.modelo_chat)
        return RAGResponse(
            consulta=pergunta,
            resposta=resposta,
            fontes=_fontes(documentos),
            contextos=[d.page_content for d in documentos],
            chunks_recuperados=len(documentos),
            configuracao_chunking=self.chunk_size,
        )


@lru_cache(maxsize=3)
def obter_pipeline(chunk_size: int = DEFAULT_CHUNK_SIZE) -> RAGPipeline:
    return RAGPipeline(chunk_size=chunk_size)


def buscar_estruturado(consulta: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> RAGResponse:
    """Versão que devolve o objeto Pydantic completo (resposta, fontes e contextos)."""
    return obter_pipeline(chunk_size).consultar(consulta)


def buscar(consulta: str) -> str:
    """Consulta a base de eficiência energética e devolve a resposta com as fontes citadas.

    Função pública simples, pronta para virar @tool do agente no CKP03.
    """
    return buscar_estruturado(consulta).para_texto()
