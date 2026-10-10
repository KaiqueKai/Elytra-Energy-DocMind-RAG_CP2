"""Executa o fluxo otimizado: diagnóstico → fontes → índices 512/1024 → consulta → RAGAS."""
from __future__ import annotations

from scripts import baixar_fontes, diagnostico, build_index


def main() -> int:
    for etapa, funcao in (("Diagnóstico", diagnostico.main), ("Download das fontes", baixar_fontes.main)):
        print(f"\n######## {etapa} ########", flush=True)
        if funcao() != 0:
            print(f"Etapa '{etapa}' falhou; corrija e execute novamente.")
            return 1
    from src.loaders import carregar_documentos
    docs = carregar_documentos()
    print("\n######## Índices ChromaDB (512, 1024) ########", flush=True)
    for tamanho in (512, 1024):
        build_index.construir(tamanho, docs)
    print("\n######## Consulta de exemplo ########", flush=True)
    from src.rag import buscar
    print(buscar("Quais cuidados de uso ajudam a reduzir o consumo de energia com o ar-condicionado?"))
    print("\n######## Avaliação RAGAS ########", flush=True)
    from evals.ragas_eval import executar_avaliacao
    executar_avaliacao()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
