# Classificador BNCC — Ensino Médio

Pipeline para classificação automática de materiais didáticos nas habilidades da Base Nacional Comum Curricular (BNCC) do Ensino Médio, usando embeddings semânticos com BERTimbau e similaridade de cosseno.

---

## Arquitetura

```
Preparação (uma vez só)
  bncc.json → BERTimbau → bncc_embeddings.npz

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
      BERTimbau          BERTimbau
           │                 │
      média dos          vetor do
       embeddings          resumo
           └────────┬────────┘
               combinação
            (média ponderada)
                    ↓
         similaridade de cosseno
          vs. vetores das habilidades
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

> Na primeira execução, o BERTimbau (~440MB) será baixado automaticamente do Hugging Face.

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
| `TOP_K` | 5 | Habilidades retornadas |
| `PESO_CHUNKS` | 0.5 | Peso da representação por chunks |
| `PESO_RESUMO` | 0.5 | Peso da representação por resumo |

---

## Convergência

O pipeline calcula a similaridade entre a representação por chunks e a por título+resumo. Valores abaixo de 0.70 indicam que as duas representações divergem — o documento pode ser ambíguo ou complexo e merece revisão humana.

---

## Referências

- [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) — BERT pré-treinado em português
- [BNCC — MEC](https://basenacional.mec.gov.br) — Base Nacional Comum Curricular
