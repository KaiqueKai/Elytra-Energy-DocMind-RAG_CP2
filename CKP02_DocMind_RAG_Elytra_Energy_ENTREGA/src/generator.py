"""Geração com modelo Gemma no endpoint OpenAI-compatible oficial do Ollama Cloud."""
from __future__ import annotations

import time
from functools import lru_cache

import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import CHAT_MODEL, CHAT_TEMPERATURE, OLLAMA_API_KEY, OLLAMA_CLOUD_HOST, PROMPT_FILE


@lru_cache(maxsize=1)
def _prompt() -> ChatPromptTemplate:
    instrucoes = PROMPT_FILE.read_text(encoding="utf-8")
    return ChatPromptTemplate.from_messages([
        ("system", "{instrucoes}"),
        ("human",
         "Pergunta do usuário: {question}\n\n"
         "Trechos recuperados da base oficial:\n{context}\n\n"
         "Responda somente com base nesses trechos, citando [Trecho N]. "
         "Se a evidência for insuficiente, diga isso claramente."),
    ]).partial(instrucoes=instrucoes)


@lru_cache(maxsize=1)
def resolver_nome_modelo() -> str:
    """Escolhe uma variante Gemma retornada pela API do Ollama Cloud."""
    if not OLLAMA_API_KEY:
        raise EnvironmentError("OLLAMA_API_KEY ausente. Configure o Secret no Colab.")
    try:
        resp = requests.get(
            f"{OLLAMA_CLOUD_HOST}/api/tags",
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}, timeout=20,
        )
        resp.raise_for_status()
        nomes = [m.get("name") or m.get("model", "") for m in resp.json().get("models", [])]
    except requests.RequestException as exc:
        raise RuntimeError(f"Não foi possível consultar modelos do Ollama Cloud: {exc}") from exc
    base = CHAT_MODEL.removesuffix(":cloud").removesuffix("-cloud")
    candidatos = [CHAT_MODEL, base, f"{base}:31b", f"{base}:31b-cloud"]
    for nome in candidatos:
        if nome in nomes:
            return nome
    variantes = sorted(n for n in nomes if n.split(":")[0] == base)
    return variantes[-1] if variantes else f"{base}:31b"


def criar_modelo_chat() -> ChatOpenAI:
    """Cliente autenticado para https://ollama.com/v1; evita o 401 do ChatOllama nesse endpoint."""
    return ChatOpenAI(
        model=resolver_nome_modelo(),
        api_key=OLLAMA_API_KEY,
        base_url=f"{OLLAMA_CLOUD_HOST}/v1",
        temperature=CHAT_TEMPERATURE,
        timeout=90,
        max_retries=2,
    )


def _texto(resposta) -> str:
    conteudo = resposta.content
    if isinstance(conteudo, str):
        return conteudo.strip()
    return "\n".join(str(item) for item in conteudo).strip()


def gerar_resposta(pergunta: str, contexto: str, modelo=None, tentativas: int = 2) -> str:
    """Gera resposta fundamentada; usa poucas tentativas para evitar esperas longas."""
    if not pergunta.strip():
        raise ValueError("A pergunta não pode estar vazia.")
    chat = modelo or criar_modelo_chat()
    prompt = _prompt().invoke({"question": pergunta, "context": contexto})
    ultimo_erro: Exception | None = None
    for tentativa in range(1, tentativas + 1):
        try:
            texto = _texto(chat.invoke(prompt))
            if texto:
                return texto
        except Exception as exc:
            ultimo_erro = exc
        if tentativa < tentativas:
            time.sleep(2 * tentativa)
    if ultimo_erro:
        raise RuntimeError(f"O modelo de chat falhou após {tentativas} tentativas: {ultimo_erro}") from ultimo_erro
    return "Não consegui produzir uma resposta com os trechos recuperados."
