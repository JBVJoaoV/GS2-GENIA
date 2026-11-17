"""API local (FastAPI) — o LLM integrado ao produto de forma segura e explicável.

Rodar:   uvicorn src.api:app --host 0.0.0.0 --port 8000
Testar:  curl -X POST localhost:8000/perguntar -H 'Content-Type: application/json' \
              -d '{"pergunta": "O que é o Sistema de Comando de Incidentes?"}'

Segurança / LGPD (resumo; detalhes no README):
- Inferência 100% local: nenhum dado sai da máquina (sem API de terceiros).
- Minimização: CPF, telefone, e-mail e CEP são mascarados ANTES de qualquer processamento.
- Logs sem texto da pergunta: só hash, tamanho, status e documentos citados.
- Explicabilidade: toda resposta traz as citações com documento, página e o trecho original.
- Recusa explícita ("nao_consta") em vez de chute; avisos de vigência vindos do documentos.json.
- Apoio à decisão: a resposta não substitui o comando do incidente (campo `aviso_legal`).
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config
from .ingest import load_chunks
from .rag import Generator, answer
from .retrieval import Retriever, load_threshold

log = logging.getLogger("defesa_civil_api")
logging.basicConfig(level=logging.INFO)

AVISO_LEGAL = ("Ferramenta de apoio à consulta de protocolos. Não substitui o comando do "
               "incidente, a decisão de profissional habilitado nem a norma oficial vigente.")

_PII = [
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[CPF]"),
    (re.compile(r"\b(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b"), "[TELEFONE]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b\d{5}-?\d{3}\b"), "[CEP]"),
]


def sanitize_pii(text: str) -> str:
    for rx, tag in _PII:
        text = rx.sub(tag, text)
    return text


class Pergunta(BaseModel):
    pergunta: str = Field(..., min_length=5, max_length=500)


STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks = load_chunks()
    adapter = config.ADAPTER_DIR if (config.ADAPTER_DIR / "adapter_config.json").exists() \
        and os.getenv("USE_LORA", "1") == "1" else None
    STATE["retriever"] = Retriever(chunks)
    STATE["generator"] = Generator(adapter_path=adapter)
    STATE["threshold"] = load_threshold()
    STATE["modelo"] = "qwen2.5-3b + LoRA" if adapter else "qwen2.5-3b (RAG simples)"
    log.info("API pronta: %s | limiar=%.3f | %d trechos",
             STATE["modelo"], STATE["threshold"], len(chunks))
    yield


app = FastAPI(title="Assistente de protocolos de incêndio — Defesa Civil", lifespan=lifespan)


@app.get("/saude")
def saude():
    return {"ok": True, "modelo": STATE.get("modelo"), "limiar": STATE.get("threshold")}


@app.post("/perguntar")
def perguntar(body: Pergunta):
    q = sanitize_pii(body.pergunta.strip())
    out = answer(q, STATE["generator"], STATE["retriever"], mode="rag",
                 threshold=STATE["threshold"])
    if out["status"] == "erro_formato":
        raise HTTPException(status_code=502, detail="O modelo não produziu resposta válida; tente reformular.")
    log.info("q=%s len=%d status=%s docs=%s gated=%s",
             hashlib.sha256(q.encode()).hexdigest()[:12], len(q), out["status"],
             sorted({c["doc_id"] for c in out["citacoes"]}), out["gated"])
    return {"status": out["status"], "resposta": out["resposta"], "citacoes": out["citacoes"],
            "avisos": out["avisos"], "confianca_busca": out["top_score"],
            "aviso_legal": AVISO_LEGAL}
