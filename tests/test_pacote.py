"""Testes da interface de biblioteca (sem GPU nem download de modelos)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import AVISO_LEGAL, DefesaCivilLLM, mascarar_pii  # noqa: E402
from src.retrieval import _encode_cached  # noqa: E402


class FakeBusca:
    def __init__(self, chunks, score=0.9):
        self.chunks, self.score = chunks, score

    def search(self, q, k=3):
        return [{**c, "score": self.score - 0.01 * i} for i, c in enumerate(self.chunks[:k])]


class FakeGen:
    adapter_path = None

    def __init__(self, saidas):
        self.saidas, self.chamadas = list(saidas), []

    def generate(self, messages, max_new_tokens=0):
        self.chamadas.append(messages)
        return self.saidas.pop(0)


def mk(i, doc="DOC006", p=19, status="referencia_tecnica"):
    return {"chunk_id": f"{doc}-p{p:03d}-c{i:03d}", "doc_id": doc, "titulo": "Título", "instituicao": "X",
            "pagina_ini": p, "pagina_fim": p, "text": "texto " * 80, "status": status}


def ok_json(cid, resp="Resposta."):
    return json.dumps({"status": "ok", "resposta": resp, "citacoes": [{"chunk_id": cid}]})


def test_gerar_resposta_com_fontes_e_aviso():
    ch = [mk(1, "DOC009", 3, "verificar_vigencia"), mk(2)]
    llm = DefesaCivilLLM(limiar=0.8, gerador=FakeGen([ok_json(ch[0]["chunk_id"])]), buscador=FakeBusca(ch))
    txt = llm.gerar_resposta("Como montar um plano de emergência?")
    assert txt.startswith("Resposta.") and "Fontes:" in txt
    assert "DOC009 · Título, p. 3" in txt and "vigência" in txt
    assert "Fontes:" not in DefesaCivilLLM(limiar=0.8, gerador=FakeGen([ok_json(ch[0]["chunk_id"])]),
                                           buscador=FakeBusca(ch)).gerar_resposta("Como montar um plano?", com_fontes=False)


def test_perguntar_devolve_contrato_completo():
    ch = [mk(1)]
    llm = DefesaCivilLLM(limiar=0.8, gerador=FakeGen([ok_json(ch[0]["chunk_id"])]), buscador=FakeBusca(ch))
    r = llm.perguntar("O que é o SCI?")
    assert r["status"] == "ok" and r["aviso_legal"] == AVISO_LEGAL
    assert r["citacoes"][0]["doc_id"] == "DOC006" and r["citacoes"][0]["trecho"]
    assert r["confianca_busca"] == pytest.approx(0.9)


def test_abaixo_do_limiar_recusa_sem_chamar_o_modelo():
    gen = FakeGen([])
    llm = DefesaCivilLLM(limiar=0.8, gerador=gen, buscador=FakeBusca([mk(1)], score=0.5))
    r = llm.perguntar("Qual a capital da Austrália?")
    assert r["status"] == "nao_consta" and not gen.chamadas
    assert "Não consta" in llm.gerar_resposta("Qual a capital da Austrália?")


def test_pii_e_mascarada_antes_do_modelo():
    gen = FakeGen([ok_json(mk(1)["chunk_id"])])
    llm = DefesaCivilLLM(limiar=0.8, gerador=gen, buscador=FakeBusca([mk(1)]))
    llm.perguntar("Meu CPF é 123.456.789-09, e-mail a@b.com: como evacuar?")
    prompt = json.dumps(gen.chamadas[0], ensure_ascii=False)
    assert "123.456.789-09" not in prompt and "a@b.com" not in prompt and "[CPF]" in prompt
    m = mascarar_pii("tel (11) 91234-5678")
    assert "91234" not in m and "[TELEFONE]" in m


def test_erro_de_formato_vira_mensagem_amigavel():
    llm = DefesaCivilLLM(limiar=0.5, gerador=FakeGen(["lixo", "mais lixo"]), buscador=FakeBusca([mk(1)]))
    assert "reformular" in llm.gerar_resposta("O que é o SCI?")


def test_pergunta_curta_e_base_ausente(monkeypatch, tmp_path):
    llm = DefesaCivilLLM(limiar=0.5, gerador=FakeGen([]), buscador=FakeBusca([mk(1)]))
    with pytest.raises(ValueError):
        llm.perguntar("  ab ")
    monkeypatch.setattr("src.llm.DATA_DIR", tmp_path)       # sem chunks.jsonl
    with pytest.raises(FileNotFoundError, match="prepare_package"):
        DefesaCivilLLM(limiar=0.5, gerador=FakeGen([]))


def test_cache_de_embeddings(tmp_path):
    class Modelo:
        chamadas = 0

        def encode(self, textos, **kw):
            Modelo.chamadas += 1
            return np.array([[len(t), 1.0] for t in textos], dtype=float)

    cache = tmp_path / "sub" / "emb.npz"
    a = _encode_cached(Modelo(), ["aa", "bbb"], cache, "m|True")
    b = _encode_cached(Modelo(), ["aa", "bbb"], cache, "m|True")
    assert Modelo.chamadas == 1 and np.array_equal(a, b)                 # 2ª vez lê do cache
    _encode_cached(Modelo(), ["aa", "bbb", "c"], cache, "m|True")        # texto mudou: recalcula
    _encode_cached(Modelo(), ["aa", "bbb", "c"], cache, "m|False")       # cabeçalho mudou: recalcula
    assert Modelo.chamadas == 3
