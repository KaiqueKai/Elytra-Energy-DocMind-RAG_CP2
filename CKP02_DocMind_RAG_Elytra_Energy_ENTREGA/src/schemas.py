"""Schemas Pydantic: metadados, chunks e resposta com proveniência (fonte/página/chunk)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arquivo: str = Field(min_length=1)
    tipo: str = Field(min_length=1)
    categoria: str = Field(min_length=1)
    data: str = Field(min_length=4)
    fonte: str = Field(min_length=1)
    instituicao: str = Field(default="Não informada", min_length=1)


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conteudo: str = Field(min_length=1)
    metadata: DocumentMetadata

    @field_validator("conteudo")
    @classmethod
    def conteudo_nao_pode_ser_vazio(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("O conteúdo do documento não pode estar vazio.")
        return value


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    conteudo: str = Field(min_length=1)
    arquivo: str = Field(min_length=1)
    pagina: int = Field(ge=1)
    categoria: str = Field(min_length=1)
    fonte: str = Field(min_length=1)


class SourceCitation(BaseModel):
    """Citação do chunk de origem (exigência do PDF: a resposta cita o chunk)."""
    model_config = ConfigDict(extra="forbid")
    trecho_numero: int = Field(default=1, ge=1)
    arquivo: str = Field(min_length=1)
    pagina: int = Field(ge=1)
    fonte: str = Field(min_length=1)
    categoria: str = Field(default="Geral", min_length=1)
    chunk_id: str = Field(default="")
    trecho: str = Field(default="")  # início do texto do chunk recuperado


class RAGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    consulta: str = Field(min_length=1)
    resposta: str = Field(min_length=1)
    fontes: list[SourceCitation] = Field(default_factory=list)
    contextos: list[str] = Field(default_factory=list)  # textos enviados ao LLM (usados no RAGAS)
    chunks_recuperados: int = Field(default=0, ge=0)
    configuracao_chunking: int = Field(default=512, ge=1)

    def para_texto(self) -> str:
        """Resposta + fontes rastreáveis (arquivo, página, chunk e URL de origem)."""
        if not self.fontes:
            return self.resposta + "\n\n**Fontes consultadas:** nenhum trecho relevante foi recuperado."
        linhas = [self.resposta, "", "**Fontes consultadas**"]
        for f in self.fontes:
            resumo = " ".join(f.trecho.split())[:160]
            linhas.append(
                f"- [Trecho {f.trecho_numero}] `{f.arquivo}`, página {f.pagina} "
                f"(chunk `{f.chunk_id}`) — {f.categoria} — {f.fonte}"
                + (f"\n  > {resumo}…" if resumo else "")
            )
        return "\n".join(linhas)
