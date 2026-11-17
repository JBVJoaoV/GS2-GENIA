"""Testes do gabarito (sem GPU, sem download): estrutura, balanceamento e semântica de páginas."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.evaluate import (bootstrap_diff, check_testset_against_chunks, facts_ok,  # noqa: E402
                          load_testset, norm, validate_testset)
from src.retrieval import is_hit, metrics_at  # noqa: E402


def test_testset_estrutura_e_balanceamento():
    items = load_testset()
    assert validate_testset(items, json.load(open(config.DOCS_JSON, encoding="utf-8"))) == []
    dev = [i for i in items if i["split"] == "dev"]
    teste = [i for i in items if i["split"] == "teste"]
    assert len(dev) == len(teste) == 20
    for parte in (dev, teste):               # cada split tem respondíveis E "não consta"
        assert any(i["deve_recusar"] for i in parte)
        assert sum(not i["deve_recusar"] for i in parte) >= 10
    assert len({i["pergunta"] for i in items}) == len(items)


def test_fatos_chave_sao_curtos_e_checaveis():
    for i in load_testset():
        for grupo in i["fatos_chave"]:
            assert all(len(norm(a).split()) <= 6 for a in grupo), (i["id"], grupo)


def test_is_hit_paginas_nao_contiguas():
    c = {"doc_id": "D", "pagina_ini": 30, "pagina_fim": 31}
    assert not is_hit(c, {"doc_esperado": "D", "paginas_esperadas": [17, 36]})   # não é intervalo 17..36
    assert is_hit({**c, "pagina_ini": 36, "pagina_fim": 36}, {"doc_esperado": "D", "paginas_esperadas": [17, 36]})


def test_check_testset_against_chunks():
    item = {"id": "T", "deve_recusar": False, "doc_esperado": "D", "paginas_esperadas": [3],
            "fatos_chave": [["estiagem"], ["risco"]]}
    ch = [{"doc_id": "D", "pagina_ini": 3, "pagina_fim": 3, "text": "Período de ESTIAGEM e risco alto."}]
    assert check_testset_against_chunks([item], ch) == []
    ch[0]["text"] = "Só estiagem."
    assert "grupos ausentes" in check_testset_against_chunks([item], ch)[0]
    assert "nenhum trecho" in check_testset_against_chunks([item], [])[0]


def test_facts_ok_numeros_e_hifen():
    assert facts_ok("19,6% do território (1.672.142 km²)", [["19,6"], ["1.672.142"]])
    assert norm("pro- posito") == "proposito"


def test_bootstrap_e_metrics_at():
    a = [{"id": str(i), "correta": False} for i in range(20)]
    b = [{"id": str(i), "correta": True} for i in range(20)]
    r = bootstrap_diff(a, b, n_boot=500)
    assert r["diff"] == 1.0 and r["ic95"][0] > 0.9
    m = metrics_at([(0.9, False), (0.6, True)], 0.8)
    assert m["f1"] == 1.0 and m["tn"] == 1


def test_doc_alias_cobre_todos_os_documentos():
    docs = json.load(open(config.DOCS_JSON, encoding="utf-8"))
    assert set(config.DOC_ALIAS) == {d["id"] for d in docs}
    assert config.DOC_ALIAS["DOC008"] == "SCI-300"
