"""Testes do chunking exigido pelo PDF."""
from langchain_core.documents import Document

from src.chunking import (
    SEPARADORES,
    config_chunking,
    criar_splitter,
    dividir_documentos,
)


def test_separadores_exigidos_pelo_enunciado():
    assert SEPARADORES == ["\n\n", "\n", ". ", " ", ""]


def test_overlap_fica_entre_dez_e_quinze_porcento():
    for tamanho in (256, 512, 1024):
        c = config_chunking(tamanho)
        assert 0.10 <= c.chunk_overlap / c.chunk_size <= 0.15


def test_divisao_preserva_proveniencia_e_gera_chunk_id():
    texto = (
        "Eficiência energética residencial. "
        "Compare o consumo informado pelo fabricante.\n\n"
    ) * 30

    origem = Document(
        page_content=texto,
        metadata={
            "arquivo": "teste.pdf",
            "pagina": 4,
            "fonte": "https://exemplo.gov.br/t.pdf",
            "categoria": "Residencial",
            "tipo": "PDF",
            "data": "2025",
        },
    )

    chunks = dividir_documentos([origem], 256)

    assert len(chunks) > 1
    assert all(
        c.metadata["arquivo"] == "teste.pdf"
        and c.metadata["pagina"] == 4
        for c in chunks
    )
    assert chunks[0].metadata["chunk_id"].startswith(
        "teste.pdf:4:"
    )


def test_splitter_usa_tamanho_configurado():
    splitter = criar_splitter(512)

    # Compatibilidade entre versões do LangChain.
    tamanho = getattr(
        splitter, "chunk_size",
        getattr(splitter, "_chunk_size", None)
    )
    overlap = getattr(
        splitter, "chunk_overlap",
        getattr(splitter, "_chunk_overlap", None)
    )

    assert tamanho == 512
    assert overlap == config_chunking(512).chunk_overlap
