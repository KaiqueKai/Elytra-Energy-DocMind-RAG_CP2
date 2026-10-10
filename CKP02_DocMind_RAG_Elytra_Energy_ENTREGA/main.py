"""Interface Gradio do DocMind RAG — Elytra Energy (diferencial do PDF)."""
from __future__ import annotations

import gradio as gr

from src.config import CATEGORIAS, DEFAULT_CHUNK_SIZE
from src.rag import obter_pipeline


def criar_interface() -> gr.Blocks:
    pipeline = obter_pipeline(DEFAULT_CHUNK_SIZE)

    def responder(mensagem: str, historico, categoria: str) -> str:
        del historico  # o RAG responde pela base documental, não por memória de conversa
        filtro = None if not categoria or categoria == "Todas" else categoria
        try:
            return pipeline.consultar(mensagem, categoria=filtro).para_texto()
        except Exception as exc:  # mostra o erro na tela em vez de travar a interface
            return f"Não consegui responder: {exc}"

    with gr.Blocks(title="Elytra Energy · DocMind RAG") as demo:
        gr.Markdown(
            "# Elytra Energy — DocMind RAG\n"
            "Assistente de eficiência energética residencial baseado em documentos oficiais "
            "(INMETRO, EPE e MME/CEPEL). Cada resposta cita arquivo, página, chunk e link de origem.\n\n"
            f"Índice ativo: `chunk_size={DEFAULT_CHUNK_SIZE}` · chat: `gemma4:cloud` (Ollama Cloud) "
            "· embeddings: `nomic-embed-text`"
        )
        gr.ChatInterface(
            fn=responder,
            additional_inputs=[gr.Dropdown(choices=["Todas"] + CATEGORIAS, value="Todas",
                                           label="Filtrar por categoria (metadata filtering)")],
            examples=[
                ["Quais sistemas são abordados no Guia para Eficiência Energética nas Edificações Públicas?", "Eficiência energética em edificações"],
                ["O que a tabela do Inmetro informa sobre ventiladores?", "Ventilação"],
                ["Como a EPE acompanha a eficiência energética no Brasil?", "Indicadores energéticos"],
            ],
        )
    return demo


if __name__ == "__main__":
    criar_interface().launch()
