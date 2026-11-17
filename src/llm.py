"""Interface de biblioteca (pip install) do assistente de protocolos de incêndio.

    from defesacivil_llm import DefesaCivilLLM

    llm = DefesaCivilLLM()          # 1ª execução baixa os pesos (Hugging Face) e calcula os embeddings
    print(llm.gerar_resposta("O que é o Sistema de Comando de Incidentes?"))
    r = llm.perguntar("...")        # dict: status, resposta, citacoes, avisos, confianca_busca, aviso_legal

ATENÇÃO: não é um modelo de uso geral. Ele responde SOMENTE com base nos 9 documentos da base
(manejo do fogo, brigadistas, SCI-100/200/300, IT-16) e responde "não consta" para o resto.

Segurança/LGPD: tudo roda localmente; CPF, telefone, e-mail e CEP são mascarados antes de qualquer
processamento; a pergunta não é gravada em log.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

from . import config
from .rag import answer

DATA_DIR = Path(__file__).resolve().parent / "data"   # chunks.jsonl e threshold.json (ver scripts/prepare_package.py)
ADAPTER_PADRAO = "JPSV/jarvis-rescue-lora"     # "usuario/repositorio" no Hugging Face, depois de publicar o adaptador LoRA
K_PADRAO = 5            # trechos no prompt (experimento do notebook 06: melhor que 3 no nosso gabarito)

AVISO_LEGAL = ("Ferramenta de apoio à consulta de protocolos. Não substitui o comando do "
               "incidente, a decisão de profissional habilitado nem a norma oficial vigente.")

# Mesmas expressões de src/api.py (sanitize_pii); mantidas aqui para não exigir fastapi na biblioteca.
_PII = [
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[CPF]"),
    (re.compile(r"\b(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b"), "[TELEFONE]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b\d{5}-?\d{3}\b"), "[CEP]"),
]


def mascarar_pii(texto: str) -> str:
    for rx, tag in _PII:
        texto = rx.sub(tag, texto)
    return texto


def _ler_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Base de trechos não encontrada em {path}. Gere com o notebook 01 e rode "
            "`python scripts/prepare_package.py` antes de instalar o pacote.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(linha) for linha in f if linha.strip()]


def _limiar_padrao() -> float:
    path = DATA_DIR / "threshold.json"
    if path.exists():
        return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
    warnings.warn(f"threshold.json não encontrado em {DATA_DIR}; usando o limiar padrão "
                  f"{config.DEFAULT_THRESHOLD} (não calibrado).")
    return config.DEFAULT_THRESHOLD


def _resolver_adapter(adapter, usar_lora: bool):
    if not usar_lora:
        return None
    ad = adapter or os.getenv("DEFESACIVIL_ADAPTER") or ADAPTER_PADRAO
    if not ad:
        warnings.warn("Nenhum adaptador LoRA configurado (parâmetro adapter= ou variável "
                      "DEFESACIVIL_ADAPTER): usando RAG simples, sem a especialização.")
        return None
    return ad


class DefesaCivilLLM:
    """Assistente local de protocolos de incêndio (RAG + LoRA opcional).

    adapter: pasta local ou id do Hugging Face ("usuario/repo") com o adaptador LoRA.
    usar_lora=False força o RAG simples. k: trechos por pergunta. limiar: similaridade mínima do
    melhor trecho (abaixo dele responde "não consta" sem chamar o modelo).
    chunks/gerador/buscador: injeção de dependências (testes).
    """

    def __init__(self, adapter=None, usar_lora: bool = True, k: int = K_PADRAO,
                 limiar: float | None = None, mascarar: bool = True,
                 modelo_base: str = config.MODEL_NAME, cache_dir=None,
                 chunks: list[dict] | None = None, gerador=None, buscador=None):
        self.k = k
        self.mascarar = mascarar
        self.limiar = limiar if limiar is not None else _limiar_padrao()

        if buscador is None:
            chunks = chunks if chunks is not None else _ler_chunks(DATA_DIR / "chunks.jsonl")
            from .retrieval import Retriever
            base = Path(cache_dir or os.getenv("DEFESACIVIL_CACHE")
                        or Path.home() / ".cache" / "defesacivil_llm")
            buscador = Retriever(chunks, cache_path=base / "embeddings.npz")

        adapter_res = None
        if gerador is None:
            adapter_res = _resolver_adapter(adapter, usar_lora)
            from .rag import Generator
            gerador = Generator(modelo_base, adapter_res)
        self.usa_lora = bool(adapter_res) or (gerador is not None and getattr(gerador, "adapter_path", None) is not None)
        self._buscador, self._gerador = buscador, gerador

    def perguntar(self, pergunta: str) -> dict:
        """Resposta completa: status (ok | nao_consta | erro_formato), texto, citações e avisos."""
        q = (pergunta or "").strip()
        if self.mascarar:
            q = mascarar_pii(q)
        if len(q) < 5:
            raise ValueError("A pergunta precisa ter pelo menos 5 caracteres.")
        out = answer(q, self._gerador, self._buscador, mode="rag",
                     threshold=self.limiar, k=self.k)
        return {"status": out["status"], "resposta": out["resposta"],
                "citacoes": out.get("citacoes", []), "avisos": out.get("avisos", []),
                "confianca_busca": out.get("top_score"), "aviso_legal": AVISO_LEGAL}

    def gerar_resposta(self, pergunta: str, com_fontes: bool = True) -> str:
        """Texto pronto para exibir: resposta + fontes (documento e página) + avisos."""
        r = self.perguntar(pergunta)
        if r["status"] == "erro_formato":
            return "Não foi possível gerar uma resposta válida; tente reformular a pergunta."
        partes = [r["resposta"]]
        if r["status"] == "ok" and com_fontes and r["citacoes"]:
            fontes = []
            for c in r["citacoes"]:
                ini, fim = c["paginas"]
                pag = f"p. {ini}" if ini == fim else f"pp. {ini}-{fim}"
                fontes.append(f"- {c['doc_id']} · {c['titulo']}, {pag}")
            partes.append("Fontes:\n" + "\n".join(fontes))
        partes += [f"Aviso: {a}" for a in r["avisos"]]
        return "\n\n".join(partes)
