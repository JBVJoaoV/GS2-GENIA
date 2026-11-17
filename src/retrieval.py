"""Busca semântica + métricas recall@1 / recall@3 + calibração do limiar.

Hit = o trecho recuperado vem do documento esperado E alguma das páginas esperadas
(numeração FÍSICA do PDF, 1-based) cai dentro do intervalo de páginas do trecho.
Se o gabarito não informar páginas, basta o documento.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


def _encode_cached(model, texts: list[str], cache_path=None, extra_key: str = ""):
    """Embeddings dos trechos com cache em disco (evita recalcular a cada execução).

    O cache vale só se o conteúdo dos textos (e o modelo/cabeçalho em extra_key) for idêntico.
    """
    import hashlib
    key = hashlib.sha1((extra_key + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()
    if cache_path is not None:
        try:
            z = np.load(cache_path, allow_pickle=False)
            if str(z["key"]) == key:
                return z["emb"]
        except Exception:      # sem cache, arquivo corrompido ou de outra versão: recalcula
            pass
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    if cache_path is not None:
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, emb=emb, key=np.array(key))
        except OSError:        # pasta sem permissão de escrita: segue sem cache
            pass
    return emb


class Retriever:
    def __init__(self, chunks: list[dict], model_name: str = config.EMBED_MODEL,
                 device: str | None = None, header: bool = True, cache_path=None):
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.header = header
        self.model = SentenceTransformer(model_name, device=device)
        # e5 exige prefixos "passage: " / "query: ". Com header=True o nome do documento entra
        # SÓ no embedding (o texto do trecho e as citações não mudam).
        def head(c):
            return f"{config.DOC_ALIAS.get(c['doc_id'], c['doc_id'])} - {c.get('titulo', '')}. " if header else ""
        texts = [f"passage: {head(c)}{c['text']}" for c in chunks]
        self.emb = _encode_cached(self.model, texts, cache_path, f"{model_name}|{header}")

    def search(self, query: str, k: int = config.TOP_K) -> list[dict]:
        q = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        sims = self.emb @ q
        idx = np.argsort(-sims)[:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in idx]


def is_hit(chunk: dict, item: dict) -> bool:
    """Trecho recuperado atende ao gabarito da pergunta?

    paginas_esperadas é um CONJUNTO de páginas físicas (não um intervalo): o trecho
    acerta se cobrir pelo menos uma delas.
    """
    if chunk["doc_id"] != item.get("doc_esperado"):
        return False
    pags = item.get("paginas_esperadas")
    if not pags:
        return True
    return any(chunk["pagina_ini"] <= p <= chunk["pagina_fim"] for p in pags)


@dataclass
class RecallResult:
    recall_at_1: float
    recall_at_3: float
    n: int
    misses: list[dict]


def evaluate_recall(retriever: Retriever, testset: list[dict]) -> RecallResult:
    """Só perguntas que TÊM resposta na base (deve_recusar == False)."""
    items = [t for t in testset if not t.get("deve_recusar")]
    h1 = h3 = 0
    misses = []
    for it in items:
        res = retriever.search(it["pergunta"], k=3)
        ok1 = is_hit(res[0], it)
        ok3 = any(is_hit(r, it) for r in res)
        h1 += ok1
        h3 += ok3
        if not ok3:
            misses.append({"id": it["id"], "pergunta": it["pergunta"],
                           "esperado": it.get("doc_esperado"),
                           "top1": res[0]["chunk_id"], "score_top1": res[0]["score"]})
    n = max(len(items), 1)
    return RecallResult(h1 / n, h3 / n, len(items), misses)


def evaluate_recall_fatos(retriever: Retriever, testset: list[dict], k: int = 3) -> float:
    """Fração das perguntas respondíveis cujos top-k trechos JUNTOS contêm todos os fatos-chave.

    Complementa o recall por documento/página: manuais que repetem o mesmo conteúdo contam
    como acerto se o fato necessário para responder está no contexto entregue ao modelo.
    """
    from .evaluate import norm
    items = [t for t in testset if not t.get("deve_recusar")]
    ok = 0
    for it in items:
        txt = norm(" ".join(r["text"] for r in retriever.search(it["pergunta"], k=k)))
        ok += all(any(norm(a) in txt for a in g) for g in it["fatos_chave"])
    return ok / max(len(items), 1)


def top1_scores(retriever: Retriever, testset: list[dict]) -> list[tuple[float, bool]]:
    """[(score_top1, deve_recusar)] para calibrar o limiar."""
    return [(retriever.search(t["pergunta"], k=1)[0]["score"], bool(t.get("deve_recusar")))
            for t in testset]


def calibrate_threshold(scores: list[tuple[float, bool]], n_grid: int = 200) -> dict:
    """Escolhe o limiar que maximiza F1 em separar 'tem resposta' de 'não consta'.

    Prevê "tem resposta" quando score >= limiar. Positivo = pergunta respondível.
    Retorna o melhor limiar e a curva (para plotar/justificar no relatório).
    """
    s = np.array([x[0] for x in scores])
    answerable = np.array([not x[1] for x in scores])
    grid = np.linspace(s.min() - 1e-3, s.max() + 1e-3, n_grid)
    curve, best = [], None
    for th in grid:
        pred = s >= th
        tp = int((pred & answerable).sum())
        fp = int((pred & ~answerable).sum())
        fn = int((~pred & answerable).sum())
        tn = int((~pred & ~answerable).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        row = {"threshold": float(th), "precision": prec, "recall": rec, "f1": f1,
               "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        curve.append(row)
        if best is None or f1 > best["f1"] + 1e-12:
            best = row
    # centro do patamar de F1 máximo (evita escolher a borda de um empate)
    top = [r for r in curve if abs(r["f1"] - best["f1"]) < 1e-12]
    chosen = top[len(top) // 2]
    return {"threshold": chosen["threshold"], "f1": chosen["f1"],
            "precision": chosen["precision"], "recall": chosen["recall"], "curve": curve}


def metrics_at(scores: list[tuple[float, bool]], threshold: float) -> dict:
    """Precisão/recall/F1 de um limiar FIXO (usado para confirmar no split de teste)."""
    s = np.array([x[0] for x in scores])
    answerable = np.array([not x[1] for x in scores])
    pred = s >= threshold
    tp = int((pred & answerable).sum()); fp = int((pred & ~answerable).sum())
    fn = int((~pred & answerable).sum()); tn = int((~pred & ~answerable).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": threshold, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def save_threshold(info: dict, path=None) -> None:
    path = path or (config.RESULTS_DIR / "threshold.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in info.items() if k != "curve"}
    path.write_text(json.dumps(slim, indent=2), encoding="utf-8")


def load_threshold(path=None) -> float:
    path = path or (config.RESULTS_DIR / "threshold.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))["threshold"]
    except FileNotFoundError:
        return config.DEFAULT_THRESHOLD
