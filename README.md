# Classificador BNCC

Pipeline para classificação automática de materiais didáticos nas habilidades da Base Nacional Comum Curricular (BNCC) — Educação Infantil, Ensino Fundamental e Ensino Médio — usando embeddings semânticos com [multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-base) e similaridade de cosseno.

---

## Arquitetura

```
Preparação (uma vez só)
  bncc.json → e5 ("passage:") → bncc_embeddings.npz

Classificação (por documento)
  documento
      ↓
  extração de texto (PDF / Word)
      ↓
  ┌─────────────────┬──────────────────┐
  │   Chunking      │  Título + Resumo │
  │ (400 tok, ov50) │   (≤ 512 tok)    │
  └────────┬────────┴────────┬─────────┘
           │                 │
   e5 ("query:")     e5 ("query:")
           │                 │
  score por habilidade:  vetor do
  média dos 3 melhores     resumo
        chunks
           └────────┬────────┘
          cosseno vs. habilidades
                    ↓
         combinação dos scores
            (média ponderada)
                    ↓
            top-k habilidades
```

---

## Estrutura do projeto

```
bncc-pipeline/
├── data/
│   ├── bncc.json                  # Habilidades e competências da BNCC
│   ├── bncc_embeddings.npz        # Vetores das habilidades (gerado)
│   ├── rotulados.exemplo.json     # Exemplo de arquivo para avaliação
│   └── rotulados.json             # Seus documentos rotulados (criar)
├── scripts/
│   ├── gerar_embeddings_bncc.py   # Passo 1: vetoriza as habilidades
│   ├── classificar_documento.py   # Passo 2: classifica um documento
│   └── avaliar.py                 # Passo 3: avalia com documentos rotulados
├── outputs/                       # Resultados gerados automaticamente
├── requirements.txt
└── README.md
```

---

## Instalação

```bash
# 1. Clone o repositório
git clone <seu-repositório>
cd bncc-pipeline

# 2. Crie e ative ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 3. Instale dependências
pip install -r requirements.txt
```

> Na primeira execução, o modelo multilingual-e5-base (~1,1 GB) será baixado automaticamente do Hugging Face.

---

## Uso

### Passo 1 — Gerar vetores das habilidades (uma vez só)

```bash
cd scripts
python gerar_embeddings_bncc.py
```

Gera `data/bncc_embeddings.npz`. Não precisa rodar novamente a menos que o `bncc.json` mude.

---

### Passo 2 — Classificar um documento

```bash
# PDF com título e resumo
python classificar_documento.py \
  --arquivo ../data/meu_material.pdf \
  --titulo "Título do material" \
  --resumo "Resumo do conteúdo..."

# Word sem resumo
python classificar_documento.py --arquivo ../data/aula.docx

# Texto direto
python classificar_documento.py --texto "Conteúdo do material aqui..."

# Retornar top-10 em vez de top-5
python classificar_documento.py --arquivo material.pdf --top 10

# Considerar só algumas etapas (EI, EF, EM; padrão: todas)
python classificar_documento.py --arquivo material.pdf --etapas EF EM
```

Saída no terminal e em `outputs/resultado_classificacao.json`.

---

### Passo 3 — Avaliar com documentos rotulados

1. Crie `data/rotulados.json` seguindo o formato de `data/rotulados.exemplo.json`
2. Execute:

```bash
python avaliar.py
```

Gera métricas de precisão, recall e F1 por documento e globais, salvas em `outputs/avaliacao.json`.

---

### Interface web

Requer os vetores gerados no Passo 1. A partir da pasta `web`:

```bash
cd web
uvicorn app:app --port 8000
```

Acesse http://localhost:8000. O modelo e os vetores são carregados na inicialização — aguarde a mensagem `Prêt.` no terminal antes de enviar um documento.

Na interface é possível enviar PDF, Word ou TXT (até 10 MB), informar título e resumo opcionais, escolher as etapas consideradas (Educação Infantil, Ensino Fundamental, Ensino Médio) e exportar o resultado como PDF.

> Durante o desenvolvimento, `--reload` reinicia o servidor a cada alteração, mas recarrega o modelo a cada vez.

---

## Formato do arquivo de avaliação

```json
[
  {
    "arquivo": "relatorio.pdf",
    "titulo": "Título opcional",
    "resumo": "Resumo opcional",
    "habilidades_corretas": ["EM13CNT101", "EM13CNT201"]
  },
  {
    "texto": "Ou cole o texto diretamente aqui...",
    "habilidades_corretas": ["EM13LP01"]
  }
]
```

---

## Parâmetros ajustáveis

Em `classificar_documento.py`:

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `CHUNK_SIZE` | 400 | Tokens por chunk |
| `CHUNK_OVERLAP` | 50 | Tokens de sobreposição |
| `TOP_K` | 20 | Habilidades retornadas |
| `TOP_CHUNKS` | 3 | Chunks mais similares considerados no score de cada habilidade |
| `LIMIAR_CONV_ALTA` / `LIMIAR_CONV_MEDIA` | 0.65 / 0.50 | Limiares de convergência |
| `PESO_CHUNKS` | 0.5 | Peso da representação por chunks |
| `PESO_RESUMO` | 0.5 | Peso da representação por resumo |

---

## Convergência

Quando há título/resumo, o pipeline mede se as duas representações (chunks e título+resumo) **ordenam as habilidades do mesmo jeito**, pela correlação de Spearman entre os dois rankings:

| Nível | Correlação | Interpretação |
|---|---|---|
| Alta | ≥ 0.65 | Resumo e documento apontam para as mesmas habilidades |
| Média | ≥ 0.50 | Concordância parcial |
| Baixa | < 0.50 | Divergem — documento ambíguo ou resumo pouco representativo; revisão humana recomendada |

Limiares iniciais (`LIMIAR_CONV_ALTA`, `LIMIAR_CONV_MEDIA`), calibrados com um documento: resumo correto ≈ 0.74, resumos de outras disciplinas entre 0.01 e 0.48. Devem ser revistos com mais documentos rotulados.

## Relevância

Com o e5, o cosseno das habilidades mais bem colocadas fica numa faixa estreita (~0.80–0.85). Por isso a interface exibe a **relevância**: 100% para a melhor habilidade e 0% para a mediana das candidatas. O cosseno bruto continua disponível no campo `confianca` (e no tooltip da barra).

---

## Referências

- [multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-base) — modelo de embeddings multilíngue treinado para busca (query → passage)
- [BNCC — MEC](https://basenacional.mec.gov.br) — Base Nacional Comum Curricular
