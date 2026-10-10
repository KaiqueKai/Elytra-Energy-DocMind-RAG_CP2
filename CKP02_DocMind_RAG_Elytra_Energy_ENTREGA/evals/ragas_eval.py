"""Avaliação RAGAS: faithfulness + answer_relevancy, 5 perguntas × 2 configurações de chunking (512 e 1024).

Gera:
- reports/ragas_por_pergunta.csv  (tabela por pergunta e por configuração — exigência do PDF)
- reports/ragas_resumo.md         (médias, tabela por pergunta e a escolha justificada pelos dados)
Os números vêm do RAGAS; nada é preenchido manualmente.
"""
from __future__ import annotations

import csv
import json
import math
import statistics

from src.chunking import config_chunking, dividir_documentos
from src.config import CHUNK_SIZES, QUESTIONS_FILE, REPORTS_DIR
EVALUATION_CHUNK_SIZES = (512, 1024)
from src.embeddings import criar_embeddings
from src.generator import criar_modelo_chat
from src.loaders import carregar_documentos
from src.rag import RAGPipeline
from src.vectorstore import carregar_vectorstore, construir_vectorstore


def carregar_perguntas() -> list[dict]:
    perguntas = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if len(perguntas) < 5:
        raise ValueError("O PDF exige no mínimo 5 perguntas em evals/perguntas.json.")
    return perguntas


def _metricas():
    """Instancia as métricas compatíveis com a versão instalada do RAGAS."""
    try:
        from ragas.metrics import Faithfulness, ResponseRelevancy
        return [Faithfulness(), ResponseRelevancy()]
    except ImportError:  # versões antigas
        from ragas.metrics import answer_relevancy, faithfulness
        return [faithfulness, answer_relevancy]


def _valido(valor) -> bool:
    try:
        return valor is not None and math.isfinite(float(valor))
    except (TypeError, ValueError):
        return False


def _rodar_ragas(amostras: list[dict]):
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset.from_list(amostras)
    resultado = evaluate(
        dataset,
        metrics=_metricas(),
        llm=LangchainLLMWrapper(criar_modelo_chat()),
        embeddings=LangchainEmbeddingsWrapper(criar_embeddings()),
        run_config=RunConfig(timeout=120, max_retries=2, max_workers=2),
        raise_exceptions=False,
        show_progress=True,
    )
    return resultado.to_pandas().to_dict(orient="records")


def avaliar_configuracao(chunk_size: int, perguntas: list[dict]) -> list[dict]:
    """Roda o pipeline real nas perguntas e mede as métricas dessa configuração."""
    pipeline = RAGPipeline(chunk_size=chunk_size)
    amostras = []
    for item in perguntas:
        resposta = pipeline.consultar(item["question"])
        amostras.append({
            "user_input": item["question"],
            "response": resposta.resposta,
            "retrieved_contexts": resposta.contextos,  # exatamente o que o LLM recebeu
            "reference": item["ground_truth"],
        })
    registros = _rodar_ragas(amostras)
    # Reavalia uma vez as linhas que voltaram sem faithfulness (JSON inválido do LLM etc.).
    pendentes = [i for i, r in enumerate(registros) if not _valido(r.get("faithfulness"))]
    if pendentes:
        print(f"  [aviso] {len(pendentes)} linha(s) sem faithfulness; reavaliando uma vez.")
        novos = _rodar_ragas([amostras[i] for i in pendentes])
        for i, novo in zip(pendentes, novos):
            for campo in ("faithfulness", "answer_relevancy"):
                if _valido(novo.get(campo)):
                    registros[i][campo] = novo[campo]
    if not registros:
        raise RuntimeError(f"RAGAS não retornou linhas para chunk_size={chunk_size}.")
    return [{
        "chunk_size": chunk_size,
        "overlap": config_chunking(chunk_size).chunk_overlap,
        "question": perguntas[i]["question"],
        "faithfulness": r.get("faithfulness"),
        "answer_relevancy": r.get("answer_relevancy"),
    } for i, r in enumerate(registros)]


def _media(valores: list[float]):
    return statistics.mean(valores) if valores else None


def _fmt(valor) -> str:
    return f"{float(valor):.3f}" if _valido(valor) else "N/D"


def _garantir_indices(documentos) -> None:
    """Cria as coleções que ainda não existem (a avaliação funciona mesmo sem pré-indexar)."""
    for tamanho in EVALUATION_CHUNK_SIZES:
        try:
            carregar_vectorstore(tamanho)
        except (FileNotFoundError, ValueError):
            construir_vectorstore(dividir_documentos(documentos, tamanho), tamanho, substituir=True)


def executar_avaliacao() -> list[dict]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    perguntas = carregar_perguntas()
    _garantir_indices(carregar_documentos())

    linhas: list[dict] = []
    for tamanho in EVALUATION_CHUNK_SIZES:
        print(f"\n[Avaliando] chunk_size={tamanho} | {len(perguntas)} perguntas")
        linhas.extend(avaliar_configuracao(tamanho, perguntas))

    caminho_csv = REPORTS_DIR / "ragas_por_pergunta.csv"
    with caminho_csv.open("w", newline="", encoding="utf-8-sig") as arquivo:
        gravador = csv.DictWriter(arquivo, fieldnames=["chunk_size", "overlap", "question",
                                                       "faithfulness", "answer_relevancy"])
        gravador.writeheader()
        gravador.writerows(linhas)

    medias = []
    for tamanho in EVALUATION_CHUNK_SIZES:
        grupo = [l for l in linhas if l["chunk_size"] == tamanho]
        faith = [float(l["faithfulness"]) for l in grupo if _valido(l["faithfulness"])]
        relev = [float(l["answer_relevancy"]) for l in grupo if _valido(l["answer_relevancy"])]
        medias.append({"chunk_size": tamanho, "faith": _media(faith), "relev": _media(relev), "n": len(faith)})

    validos = [m for m in medias if m["faith"] is not None]
    escolhido = max(validos, key=lambda m: (m["faith"], m["relev"] or -1)) if validos else None

    md = ["# Relatório RAGAS — Elytra Energy (CKP02)", "",
          f"Perguntas por configuração: {len(perguntas)}. Valores gerados pelo RAGAS nesta execução.", "",
          "## Médias por configuração de chunking", "",
          "| Chunk size | Overlap | Faithfulness média | Answer relevancy média | Perguntas com faithfulness |",
          "|---:|---:|---:|---:|---:|"]
    for m in medias:
        md.append(f"| {m['chunk_size']} | {config_chunking(m['chunk_size']).chunk_overlap} "
                  f"| {_fmt(m['faith'])} | {_fmt(m['relev'])} | {m['n']}/{len(perguntas)} |")
    md += ["", "## Resultado por pergunta", "",
           "| Chunk | # | Faithfulness | Answer relevancy | Pergunta |", "|---:|---:|---:|---:|---|"]
    for tamanho in EVALUATION_CHUNK_SIZES:
        for i, l in enumerate([x for x in linhas if x["chunk_size"] == tamanho], start=1):
            md.append(f"| {tamanho} | {i} | {_fmt(l['faithfulness'])} | {_fmt(l['answer_relevancy'])} "
                      f"| {l['question']} |")
    md += ["", "## Configuração escolhida", ""]
    if escolhido:
        f = escolhido["faith"]
        md.append(f"Pelos dados desta execução, a melhor configuração é **chunk_size={escolhido['chunk_size']}** "
                  f"(faithfulness média {f:.3f}; answer relevancy média {_fmt(escolhido['relev'])}). "
                  "O critério foi maior faithfulness média, com answer relevancy como desempate.")
        md.append("")
        if f < 0.5:
            md.append("**Atenção:** faithfulness abaixo de 0,5 indica alucinação significativa; itere no chunking antes de entregar.")
        elif f < 0.7:
            md.append("**Atenção:** faithfulness média abaixo da meta de aprovação (0,7); revise chunks, top-k e prompt.")
        elif f < 0.9:
            md.append("Faithfulness média acima da meta de aprovação (0,7), abaixo da zona ideal (0,9).")
        else:
            md.append("Faithfulness média na zona ideal (≥ 0,9).")
    else:
        md.append("O RAGAS não devolveu métricas válidas; não escolha uma configuração até corrigir a avaliação.")
    (REPORTS_DIR / "ragas_resumo.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nCSV: {caminho_csv}\nResumo: {REPORTS_DIR / 'ragas_resumo.md'}")
    for m in medias:
        print(f"chunk {m['chunk_size']}: faithfulness={_fmt(m['faith'])} | answer_relevancy={_fmt(m['relev'])}")
    return linhas


def main() -> int:
    executar_avaliacao()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
