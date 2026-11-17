"""Ingestão: PDFs -> trechos (chunks) com fonte rastreável.

Cada trecho guarda doc_id, título, instituição, página inicial/final e um
chunk_id estável (ex.: DOC005-p012-c01). É isso que permite citar a fonte.

Uso:
    python -m src.ingest
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3
except ImportError:  # versões antigas
    import fitz

from . import config


def load_docs(docs_json: Path = config.DOCS_JSON) -> list[dict]:
    with open(docs_json, encoding="utf-8") as f:
        return json.load(f)


def find_pdf(arquivo: str, kb_dir: Path = config.KB_DIR) -> Path | None:
    """Procura o PDF em qualquer subpasta (tolera pasta diferente da categoria)."""
    hits = list(kb_dir.rglob(arquivo))
    return hits[0] if hits else None


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")                 # hífen suave
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # palavra quebrada na linha
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # junta quebras de linha "no meio da frase", preserva parágrafos
    text = re.sub(r"(?<![.!?:;\n])\n(?!\n)", " ", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Retorna [(numero_da_pagina_1_based, texto)]."""
    pages = []
    with fitz.open(pdf_path) as pdf:
        for i, page in enumerate(pdf, start=1):
            pages.append((i, clean_text(page.get_text("text"))))
    return pages


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    out = []
    for p in parts:  # parágrafo gigante: quebra em sentenças
        if len(p) <= config.CHUNK_CHARS * 1.5:
            out.append(p)
        else:
            out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip())
    return out


def chunk_pages(pages: list[tuple[int, str]], chunk_chars: int = config.CHUNK_CHARS,
                overlap: int = config.CHUNK_OVERLAP,
                min_chars: int = config.MIN_CHUNK_CHARS) -> list[dict]:
    """Agrupa parágrafos em trechos de ~chunk_chars, lembrando as páginas."""
    units: list[tuple[int, str]] = []  # (pagina, parágrafo)
    for pg, txt in pages:
        for para in _split_paragraphs(txt):
            units.append((pg, para))

    chunks, buf, buf_pages = [], "", []

    def flush():
        nonlocal buf, buf_pages
        if len(buf) >= min_chars:
            chunks.append({"text": buf.strip(),
                           "pagina_ini": min(buf_pages), "pagina_fim": max(buf_pages)})
        # sobreposição: mantém o final do trecho anterior
        tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
        buf, buf_pages = tail, (buf_pages[-1:] if tail else [])

    for pg, para in units:
        if buf and len(buf) + len(para) + 1 > chunk_chars:
            flush()
        buf = f"{buf} {para}".strip() if buf else para
        buf_pages.append(pg)
    if buf and len(buf) >= min_chars:
        chunks.append({"text": buf.strip(),
                       "pagina_ini": min(buf_pages), "pagina_fim": max(buf_pages)})
    return chunks


def build_chunks(docs: list[dict] | None = None, kb_dir: Path = config.KB_DIR,
                 chunk_chars: int = config.CHUNK_CHARS,
                 overlap: int = config.CHUNK_OVERLAP) -> tuple[list[dict], list[dict]]:
    """Gera todos os trechos. Retorna (chunks, relatorio_por_documento)."""
    docs = docs or load_docs()
    all_chunks, report = [], []
    for d in docs:
        pdf = find_pdf(d["arquivo"], kb_dir)
        if pdf is None:
            report.append({"id": d["id"], "arquivo": d["arquivo"], "erro": "PDF não encontrado"})
            continue
        pages = extract_pages(pdf)
        n_vazias = sum(1 for _, t in pages if len(t) < 50)
        cs = chunk_pages(pages, chunk_chars=chunk_chars, overlap=overlap)
        for j, c in enumerate(cs, start=1):
            c.update({
                "chunk_id": f"{d['id']}-p{c['pagina_ini']:03d}-c{j:03d}",
                "doc_id": d["id"],
                "titulo": d.get("titulo", ""),
                "instituicao": d.get("instituicao", ""),
                "categoria": d.get("categoria", ""),
                "prioridade": d.get("prioridade", 2),
                "status": d.get("status", ""),
            })
        all_chunks.extend(cs)
        report.append({"id": d["id"], "arquivo": d["arquivo"], "paginas": len(pages),
                       "paginas_sem_texto": n_vazias, "chunks": len(cs),
                       "possivel_scan": n_vazias > 0.3 * max(len(pages), 1)})
    return all_chunks, report


def save_chunks(chunks: list[dict], path: Path = config.CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_chunks(path: Path = config.CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    chunks, report = build_chunks()
    save_chunks(chunks)
    for r in report:
        print(r)
    print(f"\n{len(chunks)} trechos salvos em {config.CHUNKS_PATH}")
