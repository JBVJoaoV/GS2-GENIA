"""Gera avaliacao/RESULTADOS.md a partir dos arquivos REAIS de resultados (sem números digitados à mão).

Uso: python -m src.make_report   (depois dos notebooks 02-05)
"""
from __future__ import annotations

import json

from . import config

AV = config.ROOT / "avaliacao"


def _md_table(df) -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def main():
    import pandas as pd
    parts = ["# Resultados (gerado por `python -m src.make_report`)\n"]
    if (AV / "recall.csv").exists():
        rec = pd.read_csv(AV / "recall.csv")
        if rec.shape[1] > 1 and not pd.api.types.is_numeric_dtype(rec.iloc[:, 1]):   # índice duplo (versão, split) do notebook 02
            rec.index = rec.iloc[:, 0].astype(str) + " · " + rec.iloc[:, 1].astype(str)
            rec = rec.iloc[:, 2:]
        else:
            rec = rec.set_index(rec.columns[0])
        parts += ["## Busca: recall@1 / recall@3 / recall dos fatos\n", _md_table(rec), ""]
    th = config.RESULTS_DIR / "threshold.json"
    if th.exists():
        t = json.loads(th.read_text(encoding="utf-8"))
        parts += ["## Limiar de 'não consta'\n", "```json", json.dumps(t, indent=1, ensure_ascii=False), "```", ""]
    for nome, titulo in [("placar_teste", "Placar - split teste (principal)"), ("placar_dev", "Placar - split dev")]:
        f = AV / f"{nome}.csv"
        if f.exists():
            parts += [f"## {titulo}\n", _md_table(pd.read_csv(f, index_col=0)), ""]
    f = AV / "erros.json"
    if f.exists():
        parts += ["## 5 erros reais\n", "| # | config | id | tipo | pergunta | causa provável | correção |", "|---|---|---|---|---|---|---|"]
        for i, e in enumerate(json.loads(f.read_text(encoding="utf-8")), 1):
            parts.append(f"| {i} | {e['config']} | {e['id']} | {e['tipo_erro']} | {e['pergunta']} | {e['causa_provavel']} | {e['correcao_proposta']} |")
    (AV / "RESULTADOS.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("gravado em", AV / "RESULTADOS.md")


if __name__ == "__main__":
    main()
