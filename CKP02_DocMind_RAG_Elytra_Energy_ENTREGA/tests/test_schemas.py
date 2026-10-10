"""Testes dos schemas Pydantic."""
import pytest
from pydantic import ValidationError

from src.schemas import DocumentMetadata, DocumentRecord, RAGResponse, SourceCitation


def _meta():
    return DocumentMetadata(arquivo="a.pdf", tipo="PDF", categoria="Residencial",
                            data="2025", fonte="https://exemplo.gov.br/a.pdf")


def test_documento_valida_conteudo():
    assert DocumentRecord(conteudo="  texto  ", metadata=_meta()).conteudo == "texto"


def test_documento_rejeita_conteudo_vazio():
    with pytest.raises(ValidationError):
        DocumentRecord(conteudo="   ", metadata=_meta())


def test_resposta_mostra_arquivo_pagina_chunk_e_url():
    r = RAGResponse(consulta="Como economizar?", resposta="Limpe os filtros [Trecho 1].",
                    fontes=[SourceCitation(trecho_numero=1, arquivo="dicas.pdf", pagina=3,
                                           fonte="https://exemplo.gov.br/dicas.pdf",
                                           chunk_id="dicas.pdf:3:0:512", trecho="texto do chunk")])
    texto = r.para_texto()
    for esperado in ("dicas.pdf", "página 3", "dicas.pdf:3:0:512", "https://exemplo.gov.br/dicas.pdf"):
        assert esperado in texto
