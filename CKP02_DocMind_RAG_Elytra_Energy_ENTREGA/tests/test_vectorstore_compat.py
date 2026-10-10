from types import SimpleNamespace

from src.vectorstore import _nomes_colecoes


def test_aceita_nomes_como_texto():
    class C:
        def list_collections(self):
            return ["a", "b"]
    assert _nomes_colecoes(C()) == {"a", "b"}


def test_aceita_objetos_com_nome():
    class C:
        def list_collections(self):
            return [SimpleNamespace(name="x")]
    assert _nomes_colecoes(C()) == {"x"}
