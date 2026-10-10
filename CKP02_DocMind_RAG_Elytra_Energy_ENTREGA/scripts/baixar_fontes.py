"""Baixa os cinco documentos oficiais (PDF) para data/originals/.

Para cada documento, tenta em ordem:
1. as URLs de download diretas conhecidas;
2. a descoberta automática de links .pdf dentro da página oficial de origem;
3. (se tudo falhar) mostra o nome exato do arquivo para você enviar manualmente
   pelo painel de arquivos do Colab / copiar para data/originals/.

Só aceita arquivos que realmente começam com "%PDF-".
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urljoin

import requests

from src.config import DATA_DIR, SOURCES_FILE

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36 ElytraEnergy-CKP02"),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
}

# URLs diretas conhecidas (a página oficial ".../view" é convertida em download).
URLS_DIRETAS = {
    "04_epe_atlas_eficiencia_energetica_2024.pdf": [],
    "05_mme_guia_eficiencia_energetica_edificios_publicos.pdf": [
        "https://www.gov.br/mme/pt-br/centrais-de-conteudo/publicacoes/eficiencia-energetica/guia-de-eficiencia-energetica-em-edificios-publicos.pdf/@@download/file",
    ],
}
# Palavras que identificam o PDF certo ao procurar links dentro de uma página oficial.
PALAVRAS_CHAVE = {
    "01_inmetro_condicionadores_de_ar_2025.pdf": ["condicionadores", "2025"],
    "02_inmetro_refrigeradores_2025.pdf": ["refrigerador"],
    "03_inmetro_ventiladores_2025.pdf": ["ventilador"],
    "04_epe_atlas_eficiencia_energetica_2024.pdf": ["atlas", "2024"],
    "05_mme_guia_eficiencia_energetica_edificios_publicos.pdf": ["guia", "eficiencia"],
}


def _get(url: str, **kwargs) -> requests.Response:
    """GET com fallback para certificados incompletos de sites .gov.br."""
    try:
        return requests.get(url, headers=HEADERS, timeout=(20, 180), **kwargs)
    except requests.exceptions.SSLError:
        print("   [aviso] certificado SSL do site oficial não validou; tentando sem verificação.")
        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
        return requests.get(url, headers=HEADERS, timeout=(20, 180), verify=False, **kwargs)


def _gravar_pdf(url: str, destino: Path) -> None:
    temporario = destino.with_suffix(".part")
    with _get(url, stream=True) as resposta:
        resposta.raise_for_status()
        with temporario.open("wb") as arquivo:
            for bloco in resposta.iter_content(chunk_size=1024 * 128):
                if bloco:
                    arquivo.write(bloco)
    with temporario.open("rb") as arquivo:
        assinatura = arquivo.read(5)
    if assinatura != b"%PDF-" or temporario.stat().st_size < 5000:
        temporario.unlink(missing_ok=True)
        raise ValueError("a URL não devolveu um PDF válido")
    temporario.replace(destino)


def _candidatas(nome: str, fonte: str) -> list[str]:
    """URLs de download a tentar para um arquivo."""
    urls = list(URLS_DIRETAS.get(nome, []))
    if fonte.endswith(".pdf/view"):
        base = fonte[: -len("/view")]
        urls += [base + "/@@download/file", base]
    elif fonte.lower().endswith(".pdf"):
        urls.append(fonte)
    return list(dict.fromkeys(urls))


def _descobrir_na_pagina(nome: str, pagina: str) -> list[str]:
    """Procura links .pdf na página oficial e escolhe os que casam com as palavras-chave."""
    try:
        html = _get(pagina).text
    except Exception:
        return []
    links = re.findall(r'href=["\']([^"\']+?\.pdf[^"\']*)["\']', html, flags=re.I)
    absolutos = [urljoin(pagina, unquote(link)) for link in links]
    chaves = PALAVRAS_CHAVE.get(nome, [])
    boas = [u for u in absolutos if all(c in u.lower() for c in chaves[:1])]
    return list(dict.fromkeys(boas or absolutos))[:6]


def baixar_documento(nome: str, fonte: str, destino: Path) -> None:
    erros = []
    for url in _candidatas(nome, fonte):
        for tentativa in (1, 2):
            try:
                _gravar_pdf(url, destino)
                return
            except Exception as exc:
                erros.append(f"{url}: {exc}")
                time.sleep(2 * tentativa)
    for url in _descobrir_na_pagina(nome, fonte.removesuffix("/view")):
        try:
            _gravar_pdf(url, destino)
            return
        except Exception as exc:
            erros.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(erros[-3:]) or "nenhuma URL candidata")


def _pdf_valido(caminho: Path) -> bool:
    if not caminho.exists() or caminho.stat().st_size < 5000:
        return False
    with caminho.open("rb") as arquivo:
        return arquivo.read(5) == b"%PDF-"


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifesto = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))["documentos"]
    validos, falhas = 0, []
    for nome, info in manifesto.items():
        destino = DATA_DIR / nome
        if _pdf_valido(destino):
            print(f"[OK] já existe: {nome}")
            validos += 1
            continue
        print(f"[BAIXANDO] {nome}")
        try:
            baixar_documento(nome, info["fonte"], destino)
            print(f"[OK] {nome} ({destino.stat().st_size:,} bytes)")
            validos += 1
        except Exception as exc:
            falhas.append(nome)
            print(f"[ERRO] {nome}: {exc}", file=sys.stderr)
    print(f"\nDocumentos válidos: {validos}/{len(manifesto)}")
    if falhas:
        print("\nNão foi possível baixar automaticamente:")
        for nome in falhas:
            print(f"  - {nome}\n    abra a página oficial: {manifesto[nome]['fonte']}")
        print(f"\nBaixe esses PDFs no navegador, renomeie EXATAMENTE como acima e coloque em:\n  {DATA_DIR}\n"
              "(No Colab: painel de arquivos → pasta data/originals → upload.) Depois execute de novo.")
        return 1
    print(f"Base pronta em: {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
