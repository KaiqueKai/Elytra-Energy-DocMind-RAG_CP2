"""Carrega PDFs, TXT e Markdown da base e anexa metadata rastreável (tipo, data, categoria).

Usa `pypdf` diretamente (sem langchain_community) para reduzir dependências e
evitar incompatibilidades entre versões do LangChain.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pypdf import PdfReader

from .config import DATA_DIR, SOURCES_FILE
from .schemas import DocumentMetadata, DocumentRecord

EXTENSOES = {".pdf", ".txt", ".md", ".markdown"}


def _manifesto() -> dict[str, dict[str, Any]]:
    """Lê data/fontes.json (fonte, data, categoria, tipo, instituição de cada arquivo)."""
    if not SOURCES_FILE.exists():
        return {}
    payload = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    return payload.get("documentos", {})


def _categoria_padrao(nome: str) -> str:
    """Categoria deduzida do nome do arquivo quando ele não está no manifesto."""
    nome = nome.lower()
    if "condicionador" in nome or "climat" in nome:
        return "Climatização"
    if "refrigerador" in nome or "geladeira" in nome:
        return "Refrigeração"
    if "ventilador" in nome:
        return "Ventilação"
    if "atlas" in nome or "epe" in nome:
        return "Indicadores energéticos"
    return "Residencial"


def _paginas_pdf(path: Path, max_paginas: int | None) -> list[tuple[int, str]]:
    """Extrai (número da página começando em 1, texto) de cada página do PDF."""
    leitor = PdfReader(str(path))
    if leitor.is_encrypted:
        try:
            leitor.decrypt("")
        except Exception as exc:  # PDF protegido por senha real
            raise RuntimeError(f"PDF protegido por senha: {path.name}") from exc
    total = len(leitor.pages)
    limite = min(total, max_paginas) if max_paginas else total
    paginas = []
    for indice in range(limite):
        try:
            texto = leitor.pages[indice].extract_text() or ""
        except Exception:
            texto = ""  # página com erro de extração: ignora e segue
        paginas.append((indice + 1, texto))
    return paginas


def _carregar_arquivo(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    registro_fonte = _manifesto().get(path.name, {})
    max_paginas = registro_fonte.get("max_paginas")
    # Modo rápido: limita páginas por arquivo; 0 significa todas as páginas.
    limite_global = int(os.getenv("RAG_MAX_PAGES_PER_PDF", "20"))
    if limite_global > 0:
        max_paginas = min(int(max_paginas), limite_global) if max_paginas else limite_global
    if ext == ".pdf":
        paginas = _paginas_pdf(path, int(max_paginas) if max_paginas else None)
    elif ext in {".txt", ".md", ".markdown"}:
        paginas = [(1, path.read_text(encoding="utf-8", errors="replace"))]
    else:
        return []

    fonte = str(registro_fonte.get("fonte") or path.resolve().as_uri())
    categoria = str(registro_fonte.get("categoria") or _categoria_padrao(path.name))
    data = str(registro_fonte.get("data") or "Não informada")
    tipo = str(registro_fonte.get("tipo") or ext.removeprefix(".").upper())
    instituicao = str(registro_fonte.get("instituicao") or "Não informada")

    documentos: list[Document] = []
    for pagina, texto in paginas:
        conteudo = texto.strip()
        if len(conteudo) < 20:  # páginas vazias ou só com número
            continue
        # Valida conteúdo e metadata com Pydantic antes de entrar no RAG.
        metadata = DocumentMetadata(
            arquivo=path.name, tipo=tipo, categoria=categoria,
            data=data, fonte=fonte, instituicao=instituicao,
        )
        registro = DocumentRecord(conteudo=conteudo, metadata=metadata)
        meta = registro.metadata.model_dump()
        meta["pagina"] = max(1, pagina)
        documentos.append(Document(page_content=registro.conteudo, metadata=meta))
    return documentos


def carregar_documentos(diretorio: Path | None = None) -> list[Document]:
    """Carrega todos os documentos suportados; falha cedo se a base estiver vazia."""
    origem = diretorio or DATA_DIR
    if not origem.exists():
        raise FileNotFoundError(
            f"Pasta da base não encontrada: {origem}. Execute `python -m scripts.baixar_fontes`."
        )
    arquivos = sorted(p for p in origem.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSOES)
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum PDF/TXT/Markdown em {origem}. Execute `python -m scripts.baixar_fontes`."
        )
    documentos: list[Document] = []
    erros: list[str] = []
    for arquivo in arquivos:
        try:
            documentos.extend(_carregar_arquivo(arquivo))
        except Exception as exc:
            erros.append(f"{arquivo.name}: {exc}")
    if erros:
        raise RuntimeError("Falha ao carregar documentos:\n" + "\n".join(f"- {e}" for e in erros))
    if not documentos:
        raise ValueError("Os documentos existem, mas nenhum texto legível foi extraído (PDF escaneado?).")
    return documentos
