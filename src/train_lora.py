"""Especialização por LoRA (QLoRA 4-bit) para o contrato RAG + JSON + citação + recusa.

O que o LoRA ensina: SEGUIR O FORMATO, usar só os trechos, citar chunk_id certo e
dizer "nao_consta" quando o contexto não responde. Ele NÃO precisa decorar os fatos
(os fatos vêm da busca) — por isso os trechos usados nos testes são excluídos do treino.

Fluxo:
  1) build_train_set(...)  -> data/train_lora.jsonl (dados sintéticos, auto-gerados pelo próprio Qwen)
  2) train(...)            -> adapters/qwen25-3b-defesa-civil/
"""
from __future__ import annotations

import json
import random
import re

from . import config
from .evaluate import norm
from .rag import build_messages, extract_json, target_json, RECUSA_TXT
from .retrieval import is_hit

# Perguntas fora do escopo da base (alvo: nao_consta). NÃO reutilize estas no testset.
OOD_QUESTIONS = [
    "Qual o procedimento de evacuação em caso de terremoto?",
    "Como funciona o alerta de tsunami no litoral?",
    "Qual a dose da vacina contra febre amarela?",
    "Como declarar imposto de renda de pessoa física?",
    "Qual a receita de bolo de cenoura?",
    "Quais são os sintomas da dengue?",
    "Como é calculada a aposentadoria por tempo de contribuição?",
    "Qual o protocolo de resgate em deslizamentos de terra?",
    "Como operar um drone de combate a incêndio modelo DJI Agras?",
    "Qual a capital da Austrália?",
    "Como fazer um currículo para vaga de TI?",
    "Qual o procedimento para atendimento a vítimas de afogamento?",
    "Qual o limite de velocidade em rodovias federais?",
    "Como configurar um roteador Wi-Fi?",
    "Quais os procedimentos para contenção de vazamento de produtos perigosos em rodovias?",
    "Como funciona o seguro-desemprego?",
    "Qual o tempo de cozimento ideal do arroz?",
    "Como tratar queimaduras de terceiro grau no hospital?",
    "Qual a previsão do tempo para amanhã em Brasília?",
    "Quais os protocolos de resposta a ataques cibernéticos?",
    "Como combater incêndio em plataforma de petróleo offshore?",
    "Qual o procedimento para resgate em espaço confinado?",
    "Qual o valor da multa por queimada ilegal no estado do Pará em 2026?",
    "Quem é o atual comandante do Corpo de Bombeiros do Rio de Janeiro?",
    "Como calcular a carga de incêndio de uma indústria química?",
    "Qual a legislação sobre drones no Brasil?",
    "Como funciona a distribuição de cestas básicas em desastres?",
    "Quais os critérios para decretar estado de calamidade pública por seca?",
    "Qual o protocolo de atendimento pré-hospitalar para parada cardiorrespiratória?",
    "Como instalar um sistema de alarme residencial?",
]


def good_chunk(c: dict) -> bool:
    t = c["text"]
    letras = sum(ch.isalpha() for ch in t)
    return len(t) >= 400 and letras / max(len(t), 1) > 0.7


def _gen_json(generator, instr: str, chunk_text: str) -> dict | None:
    msgs = [{"role": "system", "content": "Você prepara dados de treino em português do Brasil. Responda SOMENTE com JSON válido."},
            {"role": "user", "content": f"{instr}\n\nTRECHO:\n{chunk_text}"}]
    return extract_json(generator.generate(msgs, max_new_tokens=220))


GEN_OK = ('Escreva: "pergunta" — uma pergunta que um operador da Defesa Civil ou brigadista faria, '
          'respondível SOMENTE com este trecho, sem mencionar "o trecho" ou "o texto"; e "resposta" — '
          'a resposta correta em 1 a 3 frases, usando apenas informações do trecho. '
          'Formato: {"pergunta": "...", "resposta": "..."}')
GEN_HARD = ('Escreva "pergunta" sobre o MESMO assunto do trecho, mas que NÃO possa ser respondida com ele '
            '(pede um número, prazo, nome ou procedimento específico que o trecho não menciona). '
            'Sem mencionar "o trecho". Formato: {"pergunta": "..."}')


def _context_for(question: str, retriever, gold: dict | None, rng: random.Random, k: int) -> list[dict]:
    res = retriever.search(question, k=k)
    if gold is not None and all(r["chunk_id"] != gold["chunk_id"] for r in res):
        res = res[:-1] + [gold]
    rng.shuffle(res)  # a posição do trecho certo não pode ser pista
    return res


def build_train_set(generator, retriever, chunks: list[dict], testset: list[dict],
                    n_ok: int = 250, n_hard: int = 50, n_ood: int = 30,
                    seed: int = config.SEED, out_path=config.TRAIN_PATH) -> list[dict]:
    from tqdm.auto import tqdm
    rng = random.Random(seed)

    # exclui trechos "do gabarito" -> sem vazamento do teste para o treino
    banned = {c["chunk_id"] for c in chunks for t in testset
              if not t.get("deve_recusar") and is_hit(c, t)}
    pool = [c for c in chunks if c["chunk_id"] not in banned and good_chunk(c)]
    rng.shuffle(pool)
    print(f"{len(banned)} trechos excluídos (usados no gabarito); {len(pool)} elegíveis")

    test_qs = {norm(t["pergunta"]) for t in testset}
    rows: list[dict] = []

    # 1) exemplos com resposta
    for c in tqdm(pool[:n_ok * 2], desc="ok"):
        if sum(r["tipo"] == "ok" for r in rows) >= n_ok:
            break
        j = _gen_json(generator, GEN_OK, c["text"])
        if not j or not isinstance(j.get("pergunta"), str) or not isinstance(j.get("resposta"), str):
            continue
        q, a = j["pergunta"].strip(), j["resposta"].strip()
        if len(q) < 15 or not q.endswith("?") or len(a) < 20 or norm(q) in test_qs:
            continue
        ctx = _context_for(q, retriever, c, rng, config.TOP_K)
        rows.append({"tipo": "ok", "gold": c["chunk_id"], "pergunta": q,
                     "messages": build_messages(q, ctx),
                     "target": target_json("ok", a, [c["chunk_id"]])})

    # 2) negativos difíceis (mesmo assunto, resposta ausente) — REVISAR AMOSTRA À MÃO
    for c in tqdm(pool[n_ok * 2:n_ok * 2 + n_hard * 2], desc="hard-neg"):
        if sum(r["tipo"] == "hardneg" for r in rows) >= n_hard:
            break
        j = _gen_json(generator, GEN_HARD, c["text"])
        if not j or not isinstance(j.get("pergunta"), str):
            continue
        q = j["pergunta"].strip()
        if len(q) < 15 or not q.endswith("?") or norm(q) in test_qs:
            continue
        ctx = retriever.search(q, k=config.TOP_K)
        rows.append({"tipo": "hardneg", "gold": None, "pergunta": q,
                     "messages": build_messages(q, ctx),
                     "target": target_json("nao_consta", RECUSA_TXT, [])})

    # 3) fora do escopo
    for q in rng.sample(OOD_QUESTIONS, min(n_ood, len(OOD_QUESTIONS))):
        if norm(q) in test_qs:
            continue
        ctx = retriever.search(q, k=config.TOP_K)
        rows.append({"tipo": "ood", "gold": None, "pergunta": q,
                     "messages": build_messages(q, ctx),
                     "target": target_json("nao_consta", RECUSA_TXT, [])})

    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print("Exemplos de treino:", dict(Counter(r["tipo"] for r in rows)))
    return rows


def check_leakage(train_rows: list[dict], testset: list[dict]) -> list[str]:
    """Perguntas do teste que aparecem (normalizadas) no treino. Deve voltar vazio."""
    tq = {norm(t["pergunta"]) for t in testset}
    return [r["pergunta"] for r in train_rows if norm(r["pergunta"]) in tq]


# --------------------------------------------------------------------------
# Treino
# --------------------------------------------------------------------------
def _tokenize(tok, row: dict, max_len: int):
    prompt = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(row["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(t_ids) > max_len:
        return None  # descarta em vez de truncar (truncar cortaria contexto/resposta)
    return {"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids}


def train(train_path=config.TRAIN_PATH, out_dir=config.ADAPTER_DIR,
          model_name: str = config.MODEL_NAME):
    import inspect

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments)

    rows = [json.loads(l) for l in open(train_path, encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(model_name)
    data = [d for d in (_tokenize(tok, r, config.MAX_SEQ_LEN) for r in rows) if d]
    print(f"{len(data)}/{len(rows)} exemplos cabem em {config.MAX_SEQ_LEN} tokens")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGETS, task_type="CAUSAL_LM", bias="none"))
    model.print_trainable_parameters()
    model.config.use_cache = False

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(data)
        def __getitem__(self, i): return data[i]

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        ids = [b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}

    # aquecimento de 5% dos passos, em PASSOS (o argumento warmup_ratio foi removido
    # nas versões novas do transformers; warmup_steps funciona em todas)
    passos_por_epoca = -(-len(data) // (config.TRAIN_BATCH * config.TRAIN_GRAD_ACC))
    warmup = max(1, int(0.05 * passos_por_epoca * config.TRAIN_EPOCHS))

    kwargs = dict(
        output_dir=str(out_dir) + "_ckpt", per_device_train_batch_size=config.TRAIN_BATCH,
        gradient_accumulation_steps=config.TRAIN_GRAD_ACC, num_train_epochs=config.TRAIN_EPOCHS,
        learning_rate=config.TRAIN_LR, lr_scheduler_type="cosine", warmup_steps=warmup,
        fp16=True, logging_steps=10, save_strategy="no", report_to="none",
        gradient_checkpointing=True, optim="paged_adamw_8bit", remove_unused_columns=False,
        seed=config.SEED)
    # se uma versão futura remover/renomear algum argumento, avisa em vez de quebrar
    aceitos = set(inspect.signature(TrainingArguments.__init__).parameters)
    ignorados = sorted(k for k in kwargs if k not in aceitos)
    if ignorados:
        print("AVISO: argumentos não suportados por esta versão do transformers (ignorados):", ignorados)
    args = TrainingArguments(**{k: v for k, v in kwargs.items() if k in aceitos})
    trainer = Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate)
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "train_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    return trainer.state.log_history