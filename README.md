# CKP02 — DocMind RAG | Elytra Energy

**FIAP · Prompt Engineering and Artificial Intelligence · 2º semestre de 2026**  
**Integrantes:** Kaique Da Silva Assis (RM 572718) · Andre Debiazzi (RM 569062) · Vinicius Cristal de Oliveira (RM 572049)

## Objetivo
Pipeline RAG sobre fontes oficiais de eficiência energética, com `load → split → embed → store → retrieve → generate`, ChromaDB, citações por arquivo/página, comparação de chunking e avaliação RAGAS.

## Como executar no Google Colab
1. Abra o notebook CKP02 no Colab.
2. Em *Ambiente de execução → Alterar tipo de ambiente de execução*, selecione T4 GPU caso esteja disponível.
3. Crie o Secret `OLLAMA_API_KEY` no painel Secrets e habilite o acesso ao notebook.
4. Execute as células na ordem. A primeira célula instala dependências, inicia o Ollama e monta o cache no Google Drive.
5. A célula de fontes baixa (ou restaura do Drive) os cinco PDFs oficiais.
6. Execute a comparação, consulta RAG, testes e avaliação RAGAS. Ao final, baixe o ZIP de entrega e exporte também o notebook executado (`File → Download → Download .ipynb`).

## Fontes e documentos
A lista completa de fontes está em `data/fontes.json`; os PDFs ficam em `data/originals/`. Para incluir um documento novo, coloque o PDF/TXT/Markdown em `data/originals/` e adicione uma entrada em `data/fontes.json` com a URL oficial, instituição, data, categoria e tipo.

## Modo rápido e retomada
Por padrão, o notebook indexa as primeiras 15 páginas de cada PDF para acelerar a execução. Os arquivos PDF completos são preservados no pacote. Se precisar indexar todas as páginas, mude `RAG_MAX_PAGES_PER_PDF` para `0` na célula de preparação. Embeddings calculados são guardados em cache por lote no Google Drive, permitindo retomar lotes já calculados após uma desconexão. A comparação principal utiliza `chunk_size=512` e `chunk_size=1024`, duas estratégias distintas.

## Arquitetura
- `src/`: loaders, chunking, embeddings, vectorstore, retriever, reranker, generator e pipeline RAG.
- `scripts/`: download das fontes, diagnóstico e construção de índices.
- `evals/`: perguntas de referência e métricas RAGAS (`faithfulness` e `answer_relevancy`).
- `tests/`: testes automatizados.
- `prompts/system_prompt.md`: regras de resposta fundamentada com citação.
- `reports/`: resultados reais da avaliação, gerados ao executar o RAGAS.

A chave da API deve permanecer nos Secrets do Colab ou em `.env` local; nunca publique uma chave real no repositório.
