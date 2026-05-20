"""
Gera embeddings das habilidades da BNCC usando BERTimbau
e salva em disco para uso posterior na classificação.

Uso:
    python gerar_embeddings_bncc.py

Saída:
    bncc_embeddings.npz  — vetores + metadados (código, área, descrição)
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# ── Configurações ────────────────────────────────────────────────────────────

MODELO = "neuralmind/bert-base-portuguese-cased"  # BERTimbau base
ARQUIVO_BNCC = "data/bncc.json"
ARQUIVO_SAIDA = "data/bncc_embeddings.npz"
BATCH_SIZE = 16   # reduza para 8 se tiver pouca RAM
MAX_TOKENS = 512


# ── Funções ──────────────────────────────────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    """
    Média dos token embeddings ponderada pela attention mask.
    Ignora tokens de padding no cálculo da média.
    """
    token_embeddings = model_output.last_hidden_state  # (batch, seq, hidden)
    mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
    soma = (token_embeddings * mask_expanded).sum(dim=1)
    contagem = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return soma / contagem  # (batch, hidden)


def gerar_embeddings(textos, tokenizer, model, device):
    """
    Processa uma lista de textos em batches e retorna embeddings numpy.
    """
    todos = []

    for i in range(0, len(textos), BATCH_SIZE):
        batch = textos[i : i + BATCH_SIZE]
        print(f"  Batch {i // BATCH_SIZE + 1} / {len(textos) // BATCH_SIZE + 1} ...")

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        embeddings = mean_pooling(output, encoded["attention_mask"])
        todos.append(embeddings.cpu().numpy())

    return np.vstack(todos)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Carrega habilidades
    print(f"Carregando {ARQUIVO_BNCC} ...")
    with open(ARQUIVO_BNCC, "r", encoding="utf-8") as f:
        data = json.load(f)

    habilidades = data["habilidades"]
    textos = [h["texto_embedding"] for h in habilidades]
    codigos = [h["codigo"] for h in habilidades]
    areas = [h["area"] for h in habilidades]
    descricoes = [h["descricao"] for h in habilidades]

    print(f"{len(textos)} habilidades carregadas.")

    # 2. Carrega modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nCarregando BERTimbau ({MODELO}) em {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    model = AutoModel.from_pretrained(MODELO).to(device)
    model.eval()

    # 3. Gera embeddings
    print("\nGerando embeddings das habilidades ...")
    embeddings = gerar_embeddings(textos, tokenizer, model, device)
    print(f"Shape dos embeddings: {embeddings.shape}")  # (209, 768)

    # 4. Salva em disco
    np.savez(
        ARQUIVO_SAIDA,
        embeddings=embeddings,
        codigos=np.array(codigos),
        areas=np.array(areas),
        descricoes=np.array(descricoes),
        textos=np.array(textos),
    )
    print(f"\nSalvo em {ARQUIVO_SAIDA}")
    print("Pronto! Este arquivo pode ser carregado uma única vez na classificação.")


if __name__ == "__main__":
    main()
