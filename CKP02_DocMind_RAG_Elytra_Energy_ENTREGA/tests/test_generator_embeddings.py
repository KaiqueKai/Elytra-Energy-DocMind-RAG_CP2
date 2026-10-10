"""Testes do nome do modelo de chat (gemma4:cloud) e da escolha Cloud/local dos embeddings."""
from types import SimpleNamespace

from src import embeddings, generator


class _Resp:
    def __init__(self, status=200, corpo=None):
        self.status_code, self._corpo = status, corpo or {}

    def json(self):
        return self._corpo

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_resolve_gemma4_cloud_para_nome_da_api(monkeypatch):
    generator.resolver_nome_modelo.cache_clear()
    monkeypatch.setattr(generator, "OLLAMA_API_KEY", "chave-de-teste")
    monkeypatch.setattr(generator, "CHAT_MODEL", "gemma4:cloud")
    monkeypatch.setattr(generator.requests, "get",
                        lambda *a, **k: _Resp(200, {"models": [{"name": "gpt-oss:120b"}, {"name": "gemma4:31b"}]}))
    assert generator.resolver_nome_modelo() == "gemma4:31b"
    generator.resolver_nome_modelo.cache_clear()


def test_embeddings_cai_para_local_quando_cloud_da_401(monkeypatch):
    embeddings.modo_embeddings_efetivo.cache_clear()
    monkeypatch.setattr(embeddings, "EMBEDDING_MODE", "auto")
    monkeypatch.setattr(embeddings, "OLLAMA_API_KEY", "chave-de-teste")
    monkeypatch.setattr(embeddings.requests, "post", lambda *a, **k: _Resp(401))
    monkeypatch.setattr(embeddings.requests, "get",
                        lambda *a, **k: _Resp(200, {"models": [{"name": "nomic-embed-text:latest"}]}))
    assert embeddings.modo_embeddings_efetivo() == "local"
    embeddings.modo_embeddings_efetivo.cache_clear()


def test_embeddings_usa_cloud_quando_ele_responde(monkeypatch):
    embeddings.modo_embeddings_efetivo.cache_clear()
    monkeypatch.setattr(embeddings, "EMBEDDING_MODE", "auto")
    monkeypatch.setattr(embeddings, "OLLAMA_API_KEY", "chave-de-teste")
    monkeypatch.setattr(embeddings.requests, "post", lambda *a, **k: _Resp(200, {"embeddings": [[0.1, 0.2]]}))
    assert embeddings.modo_embeddings_efetivo() == "cloud"
    embeddings.modo_embeddings_efetivo.cache_clear()


def test_adaptador_usa_modelo_aprovado(monkeypatch):
    embeddings.modo_embeddings_efetivo.cache_clear()
    capturado = {}
    monkeypatch.setattr(embeddings, "modo_embeddings_efetivo", lambda: "local")
    monkeypatch.setattr(embeddings, "OllamaEmbeddings", lambda **kw: capturado.update(kw) or SimpleNamespace(**kw))
    embeddings.criar_embeddings()
    assert capturado["model"] == "nomic-embed-text"
