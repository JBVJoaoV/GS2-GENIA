# Assistente local de protocolos de incêndio para a Defesa Civil
**FIAP · Global Solution 2026.2 · CitizenTech · Generative AI for Engineering**
LLM local (Qwen2.5-3B-Instruct) com RAG sobre 9 documentos oficiais, JSON validado, citação de fonte, recusa ("não consta") e especialização por LoRA. Relatório completo: [`RELATORIO.md`](RELATORIO.md) · resultados: [`avaliacao/RESULTADOS.md`](avaliacao/RESULTADOS.md).

## Integrantes
| Nome completo | RM |
|---|---|
| João Vieira | 99805 |
| Lucas Speranzini | 98297 |

## Ficha Técnica do MVP
- **Produto:** Assistente de protocolos de incêndio para a Defesa Civil (Jarvis Rescue)
- **Cidadão e território:** agentes de Defesa Civil, brigadistas e gestores que atuam na prevenção e resposta a incêndios.
- **Problema:** consultar protocolos de incêndio em manuais extensos é lento e arriscado em emergências.
- **Proposta de valor:** respostas rápidas, com documento e página, a partir de fontes oficiais, e recusa quando a base não traz a resposta.
- **Arquitetura:** 9 PDFs → trechos com fonte → busca (embeddings e5-base, limiar calibrado) → Qwen2.5-3B-Instruct com LoRA → resposta em JSON com citação ou "não consta".
- **Fontes de dados:** manuais SCI-100/200/300, Instrução Técnica 16, apostila do brigadista (ICMBio/MMA), cartilhas e guias de manejo do fogo (ver `base/fontes.csv`).
- **Modelos de IA:** Qwen2.5-3B-Instruct (LoRA) e multilingual-e5-base.
- **Indicadores de impacto:** acerto nas perguntas respondíveis, taxa de alucinação e recusa correta fora da base (ver `avaliacao/RESULTADOS.md`).
- **ODS conectados:** 11, 13 e 3 (abaixo).

## ODS
- **ODS 11 (Cidades e comunidades sustentáveis, meta 11.5):** apoia a redução de danos causados por desastres e a gestão de riscos.
- **ODS 13 (Ação contra a mudança global do clima, meta 13.1):** reforça a capacidade de resposta a eventos climáticos extremos, como incêndios florestais.
- **ODS 3 (Saúde e bem-estar, meta 3.d):** apoia a gestão de riscos à saúde e à vida em emergências.

## Estrutura
| Pasta/arquivo | Conteúdo |
|---|---|
| `notebooks/` | 01 tarefa+base+gabarito · 02 busca/recall/limiar · 03 configs A e B · 04 LoRA e config C · 05 placar e erros · 06 experimento extra k=5 (post hoc) (**saídas salvas**) |
| `base/` | `chunks.jsonl` (trechos com doc, página e chunk_id) e `fontes.csv` |
| `avaliacao/` | `testset.jsonl` (gabarito, 40 perguntas), placares, `erros.json`, gráficos |
| `Base_Conhecimento/` | os 9 PDFs + `documentos.json` (metadados e status) |
| `src/` | `ingest`, `retrieval`, `rag`, `train_lora`, `evaluate`, `api`, `make_report` |
| `tests/` | testes sem GPU nem download de modelos |

## Como reproduzir (Colab ou local com GPU ≥ 16 GB para o notebook 04)
1. `git clone <URL-DO-REPO> && cd GS2-GENIA` (no Colab: abra a pasta pelo VSCode/Colab, ou preencha `REPO_URL` na 1ª célula dos notebooks).
2. Rode os notebooks **na ordem 01 → 05** (o 06 é um experimento extra, opcional). Cada um instala dependências (`requirements.txt`), acha a raiz do projeto e grava seus resultados em `base/`, `results/` e `avaliacao/`.
3. `python -m src.make_report` gera `avaliacao/RESULTADOS.md` com as tabelas a partir dos arquivos reais.
4. Testes: `pytest -q` (não precisam de GPU).
5. API local: `uvicorn src.api:app --port 8000` e `POST /perguntar {"pergunta": "..."}`. Use `USE_LORA=0` para ignorar o adaptador.

Semente fixa (`SEED=42`), decodificação gulosa (determinística) e limiar gravado em `results/threshold.json`.
O adaptador LoRA (> 100 MB) **não está no repositório** (`.gitignore`): é regenerado pelo notebook 04 (≈ 20–40 min numa T4). Adaptador publicado no Hugging Face: https://huggingface.co/JPSV/jarvis-rescue-lora

## Usar como biblioteca em outro projeto (pip)
O diretório `src/` é empacotado como `defesacivil_llm` (`pyproject.toml`).

```bash
pip install "git+https://github.com/<usuario>/<repositorio>.git"      # repositório privado: precisa de token
```
```python
from defesacivil_llm import DefesaCivilLLM

llm = DefesaCivilLLM()     # 1ª execução: baixa Qwen2.5-3B (Hugging Face) + adaptador e calcula embeddings (cache em ~/.cache/defesacivil_llm)
print(llm.gerar_resposta("O que é o Sistema de Comando de Incidentes?"))   # texto + fontes + avisos
r = llm.perguntar("...")   # dict: status (ok | nao_consta | erro_formato), resposta, citacoes, avisos, confianca_busca
```
- **Não é um modelo de uso geral**: responde só com base nos 9 documentos e diz "não consta" para o resto.
- **Adaptador LoRA**: `DefesaCivilLLM(adapter="usuario/repo-no-hf")`, ou a variável `DEFESACIVIL_ADAPTER`, ou `ADAPTER_PADRAO` em `src/llm.py`. Sem adaptador, usa RAG simples (config B).
- **Hardware**: GPU com ≥ 8 GB recomendada; em CPU funciona, mas lento (~6 GB de RAM).
- **Antes de commitar o pacote** (mantenedor): rode os notebooks 01 e 02 e depois `python scripts/prepare_package.py`, que copia `chunks.jsonl` e `threshold.json` para `src/data/`.
- O adaptador (120 MB) **não** vai no repositório: está publicado no Hugging Face Hub. Confira a licença do modelo base (Qwen2.5-3B) antes de uso comercial.
- Dados pessoais (CPF, telefone, e-mail, CEP) são mascarados antes do processamento, a inferência é local e nada é registrado em log.

## Contrato de saída
```json
{"status": "ok | nao_consta", "resposta": "...", "citacoes": [{"chunk_id": "DOC006-p019-c012"}]}
```
O sistema valida o JSON (status válido, citações só de trechos fornecidos, `ok` exige citação), tenta corrigir 1 vez e, se falhar, devolve erro em vez de resposta inventada. Cada citação é devolvida com documento, páginas e trecho original.

## Segurança, explicabilidade e LGPD
- **Local**: modelo e busca rodam na máquina; nenhum texto vai a API de terceiros.
- **Minimização**: CPF, telefone, e-mail e CEP são mascarados antes de qualquer processamento (`sanitize_pii`).
- **Logs** guardam só hash, tamanho, status e documentos citados, nunca o texto da pergunta.
- **Explicável**: citação com documento/página/trecho; `confianca_busca` (score) na resposta.
- **Anti-alucinação**: limiar recusa sem chamar o LLM; validação de citação; recusa treinada no LoRA.
- **Avisos**: documentos `historico` ou `verificar_vigencia` geram aviso na resposta; `aviso_legal` lembra que a ferramenta não substitui o comando do incidente.
- A base contém só documentos públicos institucionais; não há dados pessoais nela.

## Vídeo
Segue link:
