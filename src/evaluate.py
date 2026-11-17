"""Avaliação das três configurações e métricas do placar.

Formato do gabarito (data/testset.jsonl, 1 objeto por linha):
{
  "id": "Q01", "split": "dev" | "teste",
  "pergunta": "...",
  "deve_recusar": false,                 # true => a base NÃO contém a resposta
  "resposta_esperada": "texto de referência (para humanos)",
  "fatos_chave": [["estiagem"], ["risco", "riscos"]],
        # lista de GRUPOS; cada grupo = alternativas aceitas (basta uma, comparação por
        # substring em texto normalizado: minúsculas, sem acento);
        # a resposta só é "correta" se TODOS os grupos aparecerem.
        # Os grupos são os fatos MÍNIMOS que a resposta precisa conter.
  "doc_esperado": "DOC006",              # ausente se deve_recusar
  "paginas_esperadas": [12, 13]          # páginas FÍSICAS do PDF (1-based)
}

Definições (todas reportadas no README/relatório):
- correta:    deve_recusar -> status == nao_consta ; senão status == ok e todos os fatos_chave presentes.
- alucinação: status == ok quando deveria recusar, OU status == ok com fatos_chave errados
              (afirmação confiante e incorreta). Recusar nunca é alucinação.
- recusa_indevida: recusou pergunta que a base responde.
- citação_correta: (entre as respondidas ok, com resposta esperada) alguma citação
              aponta para o doc_esperado e páginas esperadas.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

from . import config
from .retrieval import is_hit


def norm(s: str) -> str:
    s = s.replace("\u00ad ", "").replace("\u00ad", "")
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"(\w)- (\w)", r"\1\2", s)   # palavra hifenizada na quebra de linha


def load_testset(path: Path = config.TESTSET_PATH, split: str | None = None) -> list[dict]:
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [i for i in items if split is None or i.get("split") == split]


def validate_testset(items: list[dict], docs: list[dict], chunks: list[dict] | None = None) -> list[str]:
    """Checa o gabarito antes de rodar qualquer avaliação. Retorna lista de problemas."""
    problems, seen = [], set()
    doc_ids = {d["id"] for d in docs}
    max_page = {}
    for c in chunks or []:
        max_page[c["doc_id"]] = max(max_page.get(c["doc_id"], 0), c["pagina_fim"])
    for it in items:
        i = it.get("id", "<sem id>")
        if i in seen:
            problems.append(f"{i}: id duplicado")
        seen.add(i)
        if it.get("split") not in ("dev", "teste"):
            problems.append(f"{i}: split deve ser 'dev' ou 'teste'")
        if not it.get("pergunta"):
            problems.append(f"{i}: sem pergunta")
        if it.get("deve_recusar"):
            continue
        if it.get("doc_esperado") not in doc_ids:
            problems.append(f"{i}: doc_esperado inválido ({it.get('doc_esperado')})")
        if not it.get("fatos_chave"):
            problems.append(f"{i}: sem fatos_chave")
        elif not all(isinstance(g, list) and g for g in it["fatos_chave"]):
            problems.append(f"{i}: fatos_chave deve ser lista de listas não vazias")
        pg = it.get("paginas_esperadas")
        if pg and it.get("doc_esperado") in max_page and max(pg) > max_page[it["doc_esperado"]]:
            problems.append(f"{i}: página {max(pg)} maior que o total do documento")
    n_rec = sum(bool(i.get("deve_recusar")) for i in items)
    if items and n_rec == 0:
        problems.append("nenhuma pergunta 'não consta' no gabarito (exigido pela rubrica)")
    return problems


def check_testset_against_chunks(items: list[dict], chunks: list[dict]) -> list[str]:
    """Confere o gabarito contra o TEXTO REAL dos trechos extraídos.

    Para cada pergunta respondível, junta o texto dos trechos que dão hit (doc + páginas)
    e exige que todos os grupos de fatos_chave apareçam ali. Se não aparecem, ou a página
    está errada ou o fato-chave não está na fonte -> o gabarito precisa de correção ANTES
    de qualquer métrica. Também avisa se algum termo de uma pergunta 'não consta' é achado.
    """
    problems = []
    for it in items:
        if it.get("deve_recusar"):
            continue
        txt = norm(" ".join(c["text"] for c in chunks if is_hit(c, it)))
        if not txt:
            problems.append(f"{it['id']}: nenhum trecho em {it.get('doc_esperado')} p.{it.get('paginas_esperadas')}")
            continue
        miss = [g for g in it["fatos_chave"] if not any(norm(a) in txt for a in g)]
        if miss:
            problems.append(f"{it['id']}: grupos ausentes nas páginas esperadas: {miss}")
    return problems


def facts_ok(resposta: str, fatos_chave: list[list[str]]) -> bool:
    r = norm(resposta)
    return all(any(norm(alt) in r for alt in grupo) for grupo in fatos_chave)


def score_item(item: dict, out: dict) -> dict:
    status = out.get("status")
    deve = bool(item.get("deve_recusar"))
    fmt_ok = bool(out.get("json_valido"))
    respondeu = status == "ok"
    recusou = status == "nao_consta"

    if deve:
        correta = recusou
        fatos = None
        halluc = respondeu
        recusa_indevida = False
        cit_ok = None
    else:
        fatos = respondeu and facts_ok(out.get("resposta", ""), item.get("fatos_chave", []))
        correta = bool(fatos)
        halluc = respondeu and not fatos
        recusa_indevida = recusou
        if respondeu:
            cit_ok = any(is_hit({"doc_id": c["doc_id"], "pagina_ini": c["paginas"][0],
                                 "pagina_fim": c["paginas"][1]}, item)
                         for c in out.get("citacoes", []))
        else:
            cit_ok = False
    return {"id": item["id"], "split": item.get("split"), "deve_recusar": deve,
            "status": status, "json_valido": fmt_ok, "correta": correta,
            "alucinacao": halluc, "recusa_indevida": recusa_indevida,
            "citacao_correta": cit_ok, "gated": out.get("gated", False),
            "top_score": out.get("top_score"), "pergunta": item["pergunta"],
            "resposta": out.get("resposta", ""), "esperada": item.get("resposta_esperada", ""),
            "citacoes": [c["chunk_id"] for c in out.get("citacoes", [])],
            "recuperados": out.get("recuperados", []), "raw": out.get("raw", "")}


def error_type(r: dict) -> str | None:
    """Classifica o erro de uma linha (None = sem erro)."""
    if not r["json_valido"]:
        return "formato inválido"
    if r["deve_recusar"]:
        return "alucinação: respondeu pergunta fora da base" if r["status"] == "ok" else None
    if r["recusa_indevida"]:
        return "recusa indevida (limiar do gate)" if r["gated"] else "recusa indevida (modelo)"
    if not r["correta"]:
        return "resposta incorreta/incompleta"
    if not r["citacao_correta"]:
        return "citação de trecho errado"
    return None


def pick_errors(rows: list[dict], n: int = 5) -> list[dict]:
    """Escolhe n erros priorizando variedade de tipos (1 por tipo, depois completa)."""
    errs = [{**r, "tipo_erro": error_type(r)} for r in rows if error_type(r)]
    picked, seen = [], set()
    for e in errs:
        if e["tipo_erro"] not in seen:
            picked.append(e)
            seen.add(e["tipo_erro"])
    for e in errs:
        if len(picked) >= n:
            break
        if e not in picked:
            picked.append(e)
    return picked[:n]


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    resp = [r for r in rows if not r["deve_recusar"]]
    recus = [r for r in rows if r["deve_recusar"]]
    ok_resp = [r for r in resp if r["status"] == "ok"]
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    return {
        "n": n,
        "acuracia_%": pct(sum(r["correta"] for r in rows), n),
        "acerto_respondiveis_%": pct(sum(r["correta"] for r in resp), len(resp)),
        "recusa_correta_%": pct(sum(r["correta"] for r in recus), len(recus)),
        "alucinacao_%": pct(sum(r["alucinacao"] for r in rows), n),
        "recusa_indevida_%": pct(sum(r["recusa_indevida"] for r in resp), len(resp)),
        "json_valido_%": pct(sum(r["json_valido"] for r in rows), n),
        "citacao_correta_%": pct(sum(bool(r["citacao_correta"]) for r in ok_resp), len(ok_resp)),
        # entre as respostas que o sistema EFETIVAMENTE deu (status ok): quantas estão erradas/incompletas.
        # Evita premiar quem "não alucina" só porque recusa quase tudo (caso do modelo sem contexto).
        "taxa_resposta_%": pct(sum(r["status"] == "ok" for r in rows), n),
        "erro_entre_respondidas_%": pct(sum(r["alucinacao"] for r in rows if r["status"] == "ok"),
                                        sum(r["status"] == "ok" for r in rows)),
    }


def bootstrap_diff(rows_a: list[dict], rows_b: list[dict], key: str = "correta",
                   n_boot: int = 5000, seed: int = config.SEED) -> dict:
    """IC 95% (bootstrap pareado) da diferença de taxa B - A, por pergunta (mesma ordem de ids).

    Com n pequeno o IC é largo: reportar junto da diferença, sem afirmar significância
    quando o intervalo contém 0.
    """
    a = {r["id"]: bool(r[key]) for r in rows_a}
    b = {r["id"]: bool(r[key]) for r in rows_b}
    ids = sorted(set(a) & set(b))
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        samp = [ids[rng.randrange(len(ids))] for _ in ids]
        diffs.append(sum(b[i] - a[i] for i in samp) / len(ids))
    diffs.sort()
    obs = sum(b[i] - a[i] for i in ids) / len(ids)
    return {"n": len(ids), "diff": obs, "ic95": (diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)])}


def run_config(name: str, items: list[dict], answer_fn, save: bool = True) -> list[dict]:
    """answer_fn(pergunta) -> dict de saída do pipeline."""
    from tqdm.auto import tqdm
    rows = [score_item(it, answer_fn(it["pergunta"])) for it in tqdm(items, desc=name)]
    if save:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.RESULTS_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def scoreboard(results: dict[str, list[dict]]):
    import pandas as pd
    return pd.DataFrame({name: aggregate(rows) for name, rows in results.items()}).T
