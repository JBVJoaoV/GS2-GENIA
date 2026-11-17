"""Testes que rodam sem GPU nem download de modelos (geradores/buscadores falsos)."""
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.api import sanitize_pii  # noqa: E402
from src.evaluate import (aggregate, error_type, facts_ok, pick_errors, score_item,  # noqa: E402
                          validate_testset)
from src.ingest import build_chunks, chunk_pages, clean_text, save_chunks, load_chunks  # noqa: E402
from src.rag import (RECUSA_TXT, answer, build_messages, extract_json, target_json,  # noqa: E402
                     validate_output)
from src.retrieval import calibrate_threshold, is_hit  # noqa: E402
from src import train_lora  # noqa: E402


# ----------------------------------------------------------------- helpers
def make_pdf(path: Path, pages: list[str]):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    import textwrap
    for txt in pages:
        y = 800
        for line in textwrap.wrap(txt, 85):
            c.drawString(50, y, line)
            y -= 14
        c.showPage()
    c.save()


LOREM = ("O Sistema de Comando de Incidentes organiza a resposta a emergencias com funcoes claras. "
         "O comandante do incidente coordena operacoes, planejamento, logistica e financas. ") * 6


class FakeRetriever:
    def __init__(self, chunks, score=0.9):
        self.chunks, self.score = chunks, score

    def search(self, q, k=3):
        return [{**c, "score": self.score - 0.01 * i} for i, c in enumerate(self.chunks[:k])]


class FakeGen:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def generate(self, messages, max_new_tokens=0):
        self.calls.append(messages)
        return self.outputs.pop(0)


def mk_chunk(i, doc="DOC006", p=1, status="referencia_tecnica"):
    return {"chunk_id": f"{doc}-p{p:03d}-c{i:03d}", "doc_id": doc, "titulo": "Titulo", "instituicao": "X",
            "pagina_ini": p, "pagina_fim": p, "text": "texto " * 100, "status": status}


# ----------------------------------------------------------------- ingestão
def test_clean_text_joins_hyphen_and_lines():
    assert clean_text("incen-\ndio florestal\nsegue aqui.\n\nNovo paragrafo") == \
        "incendio florestal segue aqui.\n\nNovo paragrafo"


def test_build_chunks_from_real_pdfs(tmp_path):
    kb = tmp_path / "Base_Conhecimento"
    make_pdf(kb / "03_comando" / "sci.pdf", [LOREM[:600], LOREM[:600], LOREM[:600]])
    make_pdf(kb / "x" / "it16.pdf", [LOREM[:500]])
    docs = [{"id": "DOC006", "arquivo": "sci.pdf", "titulo": "SCI", "instituicao": "IBAMA",
             "categoria": "comando_incidentes", "prioridade": 1, "status": "referencia_tecnica"},
            {"id": "DOC009", "arquivo": "it16.pdf", "titulo": "IT16", "status": "verificar_vigencia"},
            {"id": "DOC099", "arquivo": "naoexiste.pdf", "titulo": "?"}]
    chunks, rep = build_chunks(docs, kb, chunk_chars=500, overlap=50)
    assert any("erro" in r for r in rep if r["id"] == "DOC099")
    assert {c["doc_id"] for c in chunks} == {"DOC006", "DOC009"}
    for c in chunks:
        assert c["pagina_ini"] >= 1 and c["pagina_fim"] >= c["pagina_ini"]
        assert c["chunk_id"].startswith(c["doc_id"])
    assert max(c["pagina_fim"] for c in chunks if c["doc_id"] == "DOC006") == 3
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)          # ids únicos
    p = tmp_path / "c.jsonl"
    save_chunks(chunks, p)
    assert load_chunks(p) == chunks


def test_chunk_pages_tracks_page_range():
    pages = [(1, "a" * 400), (2, "b" * 400), (3, "c" * 400)]
    cs = chunk_pages(pages, chunk_chars=700, overlap=0, min_chars=100)
    assert cs[0]["pagina_ini"] == 1 and cs[-1]["pagina_fim"] == 3


# ----------------------------------------------------------------- JSON / validação
def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Claro! {"a": "x}y", "b": {"c": 2}} fim') == {"a": "x}y", "b": {"c": 2}}
    assert extract_json("sem json") is None
    assert extract_json('{"a": ') is None


def test_validate_output_rules():
    ids = {"D-1", "D-2"}
    ok, errs, n = validate_output({"status": "ok", "resposta": "x", "citacoes": [{"chunk_id": "D-1"}]}, ids)
    assert ok and n["citacoes"] == ["D-1"]
    assert not validate_output({"status": "ok", "resposta": "x", "citacoes": []}, ids)[0]          # ok sem citação
    assert not validate_output({"status": "ok", "resposta": "x", "citacoes": ["Z"]}, ids)[0]       # id inventado
    assert not validate_output({"status": "talvez", "resposta": "x", "citacoes": []}, ids)[0]
    assert not validate_output({"status": "nao_consta", "resposta": "x", "citacoes": ["D-1"]}, ids)[0]
    assert validate_output({"status": "ok", "resposta": "x", "citacoes": []}, None)[0]             # config base
    assert not validate_output(None, ids)[0]


# ----------------------------------------------------------------- pipeline answer()
def test_answer_gate_refuses_without_calling_llm():
    gen = FakeGen([])
    out = answer("pergunta", gen, FakeRetriever([mk_chunk(1)], score=0.5), threshold=0.8)
    assert out["status"] == "nao_consta" and out["gated"] and out["resposta"] == RECUSA_TXT and not gen.calls


def test_answer_ok_enriches_citation_and_warning():
    ch = [mk_chunk(1, "DOC009", status="verificar_vigencia"), mk_chunk(2)]
    raw = json.dumps({"status": "ok", "resposta": "R", "citacoes": [{"chunk_id": ch[0]["chunk_id"]}]})
    out = answer("q", FakeGen([raw]), FakeRetriever(ch), threshold=0.8)
    assert out["status"] == "ok" and out["json_valido"]
    assert out["citacoes"][0]["doc_id"] == "DOC009" and out["citacoes"][0]["trecho"]
    assert out["avisos"] and "vigência" in out["avisos"][0]


def test_answer_retries_once_then_succeeds_and_fails():
    ch = [mk_chunk(1)]
    good = json.dumps({"status": "ok", "resposta": "R", "citacoes": [{"chunk_id": ch[0]["chunk_id"]}]})
    out = answer("q", FakeGen(["lixo", good]), FakeRetriever(ch), threshold=0.5)
    assert out["status"] == "ok"
    out = answer("q", FakeGen(["lixo", "mais lixo"]), FakeRetriever(ch), threshold=0.5)
    assert out["status"] == "erro_formato" and not out["json_valido"]


def test_answer_blocks_invented_citation():
    ch = [mk_chunk(1)]
    bad = json.dumps({"status": "ok", "resposta": "R", "citacoes": [{"chunk_id": "INVENTADO"}]})
    out = answer("q", FakeGen([bad, bad]), FakeRetriever(ch), threshold=0.5)
    assert out["status"] == "erro_formato"


def test_answer_base_mode_has_no_context():
    gen = FakeGen([json.dumps({"status": "ok", "resposta": "chute", "citacoes": []})])
    out = answer("q", gen, mode="base")
    assert out["status"] == "ok" and out["citacoes"] == []
    assert "TRECHOS" not in gen.calls[0][1]["content"]


# ----------------------------------------------------------------- avaliação
ITEM_OK = {"id": "T1", "split": "teste", "pergunta": "q", "deve_recusar": False,
           "fatos_chave": [["Sistema de Comando de Incidentes"], ["comandante"]],
           "doc_esperado": "DOC006", "paginas_esperadas": [5, 6], "resposta_esperada": "x"}
ITEM_NO = {"id": "T2", "split": "teste", "pergunta": "q2", "deve_recusar": True}


def _out(status, resp="", cits=None, valid=True, gated=False):
    return {"status": status, "resposta": resp, "citacoes": cits or [], "json_valido": valid, "gated": gated}


def test_facts_ok_accent_and_alternatives():
    assert facts_ok("O sistema de comando de incidentes, liderado pelo COMANDANTE.", ITEM_OK["fatos_chave"])
    assert not facts_ok("Só o comandante.", ITEM_OK["fatos_chave"])
    assert facts_ok("Instrucao tecnica", [["Instrução Técnica"]])


def test_score_item_cases():
    cit = [{"chunk_id": "c", "doc_id": "DOC006", "paginas": [5, 5]}]
    r = score_item(ITEM_OK, _out("ok", "Sistema de Comando de Incidentes e comandante", cit))
    assert r["correta"] and not r["alucinacao"] and r["citacao_correta"] is True
    r = score_item(ITEM_OK, _out("ok", "resposta errada", cit))
    assert not r["correta"] and r["alucinacao"]
    r = score_item(ITEM_OK, _out("nao_consta", gated=True))
    assert r["recusa_indevida"] and not r["alucinacao"]
    r = score_item(ITEM_NO, _out("ok", "inventei"))
    assert r["alucinacao"] and not r["correta"]
    r = score_item(ITEM_NO, _out("nao_consta"))
    assert r["correta"] and not r["alucinacao"]
    wrong_page = [{"chunk_id": "c", "doc_id": "DOC006", "paginas": [90, 91]}]
    r = score_item(ITEM_OK, _out("ok", "Sistema de Comando de Incidentes comandante", wrong_page))
    assert r["correta"] and r["citacao_correta"] is False and error_type(r) == "citação de trecho errado"


def test_aggregate_and_pick_errors():
    cit = [{"chunk_id": "c", "doc_id": "DOC006", "paginas": [5, 5]}]
    rows = [score_item(ITEM_OK, _out("ok", "Sistema de Comando de Incidentes comandante", cit)),
            score_item(ITEM_NO, _out("ok", "inventei")),
            score_item({**ITEM_OK, "id": "T3"}, _out("nao_consta", gated=True)),
            score_item({**ITEM_OK, "id": "T4"}, _out("", valid=False))]
    a = aggregate(rows)
    assert a["n"] == 4 and a["acuracia_%"] == 25.0 and a["alucinacao_%"] == 25.0
    assert a["json_valido_%"] == 75.0 and a["recusa_correta_%"] == 0.0
    errs = pick_errors(rows, n=5)
    assert {e["tipo_erro"] for e in errs} == {"alucinação: respondeu pergunta fora da base",
                                              "recusa indevida (limiar do gate)", "formato inválido"}


def test_validate_testset_catches_problems():
    docs = [{"id": "DOC006"}]
    assert validate_testset([ITEM_OK, ITEM_NO], docs) == []
    bad = [{"id": "T1", "split": "x", "pergunta": "q", "doc_esperado": "DOC999", "fatos_chave": []}]
    probs = validate_testset(bad, docs)
    assert any("split" in p for p in probs) and any("doc_esperado" in p for p in probs)
    assert any("não consta" in p for p in probs)


# ----------------------------------------------------------------- busca
def test_is_hit_doc_and_pages():
    c = {"doc_id": "DOC006", "pagina_ini": 4, "pagina_fim": 5}
    assert is_hit(c, {"doc_esperado": "DOC006", "paginas_esperadas": [5, 6]})
    assert not is_hit(c, {"doc_esperado": "DOC006", "paginas_esperadas": [7, 8]})
    assert not is_hit(c, {"doc_esperado": "DOC005"})
    assert is_hit(c, {"doc_esperado": "DOC006"})


def test_calibrate_threshold_separates_classes():
    scores = [(0.9, False), (0.85, False), (0.8, False), (0.6, True), (0.55, True), (0.5, True)]
    info = calibrate_threshold(scores)
    assert 0.6 < info["threshold"] <= 0.8 and info["f1"] == 1.0


# ----------------------------------------------------------------- LoRA (dados e máscara)
class FakeTok:
    eos_token = "<E>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join(m["content"] for m in msgs) + "<A>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 250 for c in text]}


def test_tokenize_masks_prompt_and_drops_too_long():
    row = {"messages": build_messages("pergunta?", [mk_chunk(1)]), "target": target_json("ok", "r", ["x"])}
    d = train_lora._tokenize(FakeTok(), row, max_len=10_000)
    n_prompt = sum(1 for l in d["labels"] if l == -100)
    assert 0 < n_prompt < len(d["labels"]) and len(d["input_ids"]) == len(d["labels"])
    assert train_lora._tokenize(FakeTok(), row, max_len=10) is None


def test_build_train_set_excludes_gabarito_chunks_and_leakage(tmp_path):
    chunks = [{**mk_chunk(i, "DOC00" + str(1 + i % 3), p=i), "text": "palavra util " * 60} for i in range(1, 31)]
    testset = [{"id": "T", "split": "teste", "pergunta": "Pergunta do teste?", "deve_recusar": False,
                "doc_esperado": "DOC002", "paginas_esperadas": [1, 4], "fatos_chave": [["x"]]}]
    banned_ids = {c["chunk_id"] for c in chunks if is_hit(c, testset[0])}
    assert banned_ids

    class G:
        n = 0

        def generate(self, messages, max_new_tokens=0):
            G.n += 1
            if "NÃO possa ser respondida" in messages[1]["content"]:
                return json.dumps({"pergunta": f"Qual o prazo especifico numero {G.n} do assunto?"})
            return json.dumps({"pergunta": f"Qual a funcao numero {G.n} descrita no texto?",
                               "resposta": "Uma resposta suficientemente longa para passar."})

    out = tmp_path / "train.jsonl"
    rows = train_lora.build_train_set(G(), FakeRetriever(chunks), chunks, testset, n_ok=8, n_hard=3,
                                      n_ood=4, out_path=out)
    assert {r["tipo"] for r in rows} == {"ok", "hardneg", "ood"}
    assert all(r["gold"] not in banned_ids for r in rows if r["gold"])
    assert train_lora.check_leakage(rows, testset) == []
    for r in rows:
        t = json.loads(r["target"])
        assert (t["status"] == "ok") == (r["tipo"] == "ok")
        if r["tipo"] == "ok":   # trecho certo sempre presente no contexto
            assert r["gold"] in r["messages"][1]["content"]
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == len(rows)


# ----------------------------------------------------------------- LGPD
def test_sanitize_pii():
    s = sanitize_pii("CPF 123.456.789-09, tel (11) 91234-5678, a@b.com, CEP 01310-100: como evacuar?")
    for raw in ("123.456.789-09", "91234-5678", "a@b.com", "01310-100"):
        assert raw not in s
    assert "como evacuar?" in s
