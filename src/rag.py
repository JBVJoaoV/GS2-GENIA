"""Geração fundamentada (RAG): prompt, JSON validado, citação e recusa.

Saída padrão do sistema (contrato):
{
  "status": "ok" | "nao_consta",
  "resposta": "texto",
  "citacoes": [{"chunk_id": "DOC005-p012-c003"}]
}
O sistema depois enriquece cada citação com doc, página e o trecho original
(isso é o que torna a resposta explicável/auditável).
"""
from __future__ import annotations

import json
import re

from . import config

RECUSA_TXT = "Não consta nos documentos da base de conhecimento."

SYSTEM_RAG = (
    "Você é um assistente de apoio à Defesa Civil para protocolos de incêndio "
    "(florestal e em edificações). Responda SOMENTE com base nos TRECHOS fornecidos. "
    "Se os trechos não contêm a resposta, responda com status \"nao_consta\". "
    "Nunca invente números, prazos, siglas ou procedimentos. "
    "Responda em português do Brasil, de forma objetiva, e SOMENTE com um objeto JSON "
    "neste formato exato:\n"
    '{"status": "ok" ou "nao_consta", "resposta": "texto", '
    '"citacoes": [{"chunk_id": "<id de um trecho usado>"}]}\n'
    "Em \"citacoes\" use apenas ids de trechos fornecidos. Se status for "
    "\"nao_consta\", citacoes deve ser []. Não escreva nada fora do JSON."
)

SYSTEM_BASE = (
    "Você é um assistente de apoio à Defesa Civil para protocolos de incêndio. "
    "Responda em português do Brasil, e SOMENTE com um objeto JSON neste formato exato:\n"
    '{"status": "ok" ou "nao_consta", "resposta": "texto", "citacoes": []}\n'
    "Se você não souber a resposta com segurança, use status \"nao_consta\". "
    "Não escreva nada fora do JSON."
)


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        pag = (f"p. {c['pagina_ini']}" if c["pagina_ini"] == c["pagina_fim"]
               else f"pp. {c['pagina_ini']}-{c['pagina_fim']}")
        blocks.append(f"[{c['chunk_id']}] ({c['doc_id']} · {c['titulo']} · {pag})\n{c['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[dict] | None) -> list[dict]:
    """chunks=None -> configuração sem contexto (modelo base puro)."""
    if chunks is None:
        return [{"role": "system", "content": SYSTEM_BASE},
                {"role": "user", "content": f"Pergunta: {question}"}]
    user = f"TRECHOS:\n{format_context(chunks)}\n\nPergunta: {question}"
    return [{"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": user}]


def target_json(status: str, resposta: str, chunk_ids: list[str]) -> str:
    """Resposta-alvo usada no treino LoRA (mesmo contrato do sistema)."""
    return json.dumps({"status": status, "resposta": resposta,
                       "citacoes": [{"chunk_id": c} for c in chunk_ids]},
                      ensure_ascii=False)


# --------------------------------------------------------------------------
# Parse e validação
# --------------------------------------------------------------------------
def extract_json(text: str) -> dict | None:
    """Extrai o primeiro objeto JSON do texto (tolera ```json e texto ao redor)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def validate_output(obj: dict | None, allowed_ids: set[str] | None) -> tuple[bool, list[str], dict | None]:
    """Valida contrato. allowed_ids=None => não exige que citações existam (config base).

    Retorna (valido, erros, objeto_normalizado).
    """
    errs: list[str] = []
    if not isinstance(obj, dict):
        return False, ["não é um objeto JSON"], None
    status = obj.get("status")
    if status not in ("ok", "nao_consta"):
        errs.append(f"status inválido: {status!r}")
    resp = obj.get("resposta")
    if not isinstance(resp, str) or not resp.strip():
        errs.append("campo 'resposta' ausente ou vazio")
    cits = obj.get("citacoes", [])
    ids: list[str] = []
    if not isinstance(cits, list):
        errs.append("'citacoes' não é lista")
        cits = []
    for c in cits:
        cid = c.get("chunk_id") if isinstance(c, dict) else c
        if not isinstance(cid, str):
            errs.append("citação malformada")
            continue
        ids.append(cid)
    if allowed_ids is not None:
        bad = [i for i in ids if i not in allowed_ids]
        if bad:
            errs.append(f"citação de trecho inexistente/não fornecido: {bad}")
        if status == "ok" and not ids:
            errs.append("status ok sem citação")
    if status == "nao_consta" and ids:
        errs.append("nao_consta com citações")
    if errs:
        return False, errs, None
    return True, [], {"status": status, "resposta": resp.strip(), "citacoes": ids}


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
class Generator:
    """Qwen2.5-3B-Instruct (opcionalmente com adaptador LoRA)."""

    def __init__(self, model_name: str = config.MODEL_NAME, adapter_path=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto")
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self.model.eval()
        self.adapter_path = adapter_path

    def generate(self, messages: list[dict], max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
        import torch
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def _enrich(ids: list[str], by_id: dict[str, dict]) -> tuple[list[dict], list[str]]:
    cits, avisos = [], []
    for cid in ids:
        c = by_id.get(cid)
        if not c:
            continue
        cits.append({"chunk_id": cid, "doc_id": c["doc_id"], "titulo": c["titulo"],
                     "instituicao": c["instituicao"],
                     "paginas": [c["pagina_ini"], c["pagina_fim"]], "trecho": c["text"]})
        aviso = config.STATUS_COM_AVISO.get(c.get("status", ""))
        if aviso and aviso not in avisos:
            avisos.append(f"{c['doc_id']}: {aviso}")
    return cits, avisos


def answer(question: str, generator: Generator, retriever=None, mode: str = "rag",
           threshold: float | None = None, k: int = config.TOP_K,
           chunks_by_id: dict[str, dict] | None = None) -> dict:
    """mode='base' (sem contexto) ou 'rag' (com busca; vale p/ modelo base e com LoRA)."""
    base_out = {"pergunta": question, "modo": mode, "gated": False, "top_score": None,
                "json_valido": False, "erros": [], "raw": "", "recuperados": []}

    if mode == "base":
        messages, allowed = build_messages(question, None), None
        retrieved: list[dict] = []
    else:
        retrieved = retriever.search(question, k=k)
        base_out["recuperados"] = [r["chunk_id"] for r in retrieved]
        base_out["top_score"] = retrieved[0]["score"]
        if threshold is not None and retrieved[0]["score"] < threshold:
            # abaixo do limiar: recusa SEM chamar o LLM (barato e à prova de alucinação)
            base_out.update({"status": "nao_consta", "resposta": RECUSA_TXT, "citacoes": [],
                             "avisos": [], "gated": True, "json_valido": True})
            return base_out
        messages = build_messages(question, retrieved)
        allowed = {r["chunk_id"] for r in retrieved}
        chunks_by_id = {r["chunk_id"]: r for r in retrieved}

    raw = generator.generate(messages)
    ok, errs, norm = validate_output(extract_json(raw), allowed)
    if not ok:  # 1 nova tentativa avisando do erro
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Sua resposta não seguiu o formato. Corrija: "
             + "; ".join(errs) + ". Responda SOMENTE com o JSON válido."}]
        raw2 = generator.generate(messages)
        ok, errs, norm = validate_output(extract_json(raw2), allowed)
        raw = raw2 if ok else raw + "\n---retry---\n" + raw2

    base_out["raw"] = raw
    if not ok:
        base_out.update({"status": "erro_formato", "resposta": "", "citacoes": [],
                         "avisos": [], "erros": errs})
        return base_out

    cits, avisos = _enrich(norm["citacoes"], chunks_by_id or {})
    base_out.update({"status": norm["status"], "resposta": norm["resposta"],
                     "citacoes": cits, "avisos": avisos, "json_valido": True})
    return base_out
