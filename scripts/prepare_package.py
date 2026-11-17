"""Copia os dados que o pacote precisa para src/data/ (rode UMA vez, antes de commitar).

    python scripts/prepare_package.py

Origem: data/chunks.jsonl (ou base/chunks.jsonl) e results/threshold.json, gerados pelos notebooks 01 e 02.
"""
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "src" / "data"


def achar(*candidatos: str) -> Path:
    for c in candidatos:
        p = RAIZ / c
        if p.exists():
            return p
    sys.exit(f"Arquivo não encontrado (procurei {candidatos}). Rode antes os notebooks 01 e 02.")


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    for origem, nome in [(achar("data/chunks.jsonl", "base/chunks.jsonl"), "chunks.jsonl"),
                         (achar("results/threshold.json"), "threshold.json")]:
        shutil.copy(origem, DESTINO / nome)
        print(f"{origem.relative_to(RAIZ)} -> src/data/{nome} ({(DESTINO / nome).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
