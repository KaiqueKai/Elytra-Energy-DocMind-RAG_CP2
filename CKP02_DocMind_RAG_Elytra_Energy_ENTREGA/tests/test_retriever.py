"""Testes do metadata filtering (where) e do reranking."""
import pytest
from langchain_core.documents import Document

from src import retriever
from src.reranker import reranquear


def test_filtro_por_categoria():
    assert retriever.montar_filtro("Climatização") == {"categoria": "Climatização"}


def test_filtro_todas_nao_filtra():
    assert retriever.montar_filtro("Todas") is None and retriever.montar_filtro(None) is None


def test_filtro_combinado_usa_and():
    f = retriever.montar_filtro("Climatização", "INMETRO")
    assert f == {"$and": [{"categoria": "Climatização"}, {"instituicao": "INMETRO"}]}


def test_categoria_invalida_gera_erro():
    with pytest.raises(ValueError):
        retriever.montar_filtro("Inexistente")


def test_reranking_desligado_mantem_ordem():
    docs = [Document(page_content="a"), Document(page_content="b")]
    assert reranquear("q", docs, habilitado=False) == docs
