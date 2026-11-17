"""Configuração central do projeto. Tudo que é "número mágico" fica aqui."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Dados -----------------------------------------------------------------
KB_DIR = ROOT / "Base_Conhecimento"          # PDFs (podem estar em subpastas)
DOCS_JSON = KB_DIR / "documentos.json"       # metadados dos 9 PDFs
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"      # gerado por ingest.py
TESTSET_PATH = DATA_DIR / "testset.jsonl"    # gabarito feito à mão
TRAIN_PATH = DATA_DIR / "train_lora.jsonl"   # gerado por train_lora.py
RESULTS_DIR = ROOT / "results"
ADAPTER_DIR = ROOT / "adapters" / "qwen25-3b-defesa-civil"

# --- Modelos ---------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
EMBED_MODEL = "intfloat/multilingual-e5-base"  # multilíngue, bom em PT-BR

# --- Chunking --------------------------------------------------------------
CHUNK_CHARS = 1000      # tamanho alvo de cada trecho (caracteres)
CHUNK_OVERLAP = 150     # sobreposição entre trechos consecutivos
MIN_CHUNK_CHARS = 200   # trechos menores que isso são descartados (ruído)

# --- Busca -----------------------------------------------------------------
TOP_K = 3
# Nome curto como o operador cita o documento; entra no cabeçalho do embedding de cada trecho
# (os manuais SCI-100/200/300 repetem os mesmos conceitos: sem isso a busca os confunde).
DOC_ALIAS = {"DOC001": "Roteiro Metodológico", "DOC002": "Guia Prático", "DOC003": "Cartilha PNMIF",
             "DOC004": "É possível controlar o fogo", "DOC005": "Apostila do Brigadista",
             "DOC006": "SCI-100", "DOC007": "SCI-200", "DOC008": "SCI-300",
             "DOC009": "Instrução Técnica 16"}
# Limiar de similaridade (cosseno) abaixo do qual respondemos "não consta"
# SEM chamar o LLM. É calibrado no notebook 02 e gravado em results/threshold.json.
DEFAULT_THRESHOLD = 0.80

# --- Geração ---------------------------------------------------------------
MAX_NEW_TOKENS = 400
SEED = 42

# --- LoRA ------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
TRAIN_EPOCHS = 2
TRAIN_LR = 2e-4
TRAIN_BATCH = 1
TRAIN_GRAD_ACC = 8
MAX_SEQ_LEN = 2048

# Status de documentos que geram aviso na resposta (vem do documentos.json)
STATUS_COM_AVISO = {
    "verificar_vigencia": "Documento com vigência a verificar: confirme a versão atual da norma.",
    "historico": "Material histórico/informativo: não substitui o protocolo oficial vigente.",
}
