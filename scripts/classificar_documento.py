"""
Classifica um documento nas habilidades da BNCC do Ensino Médio.

Estratégia:
  - Representação 1: chunking com overlap → média dos embeddings
  - Representação 2: título + resumo (se disponível)
  - Combinação: média ponderada das duas representações
  - Classificação: similaridade de cosseno contra vetores das habilidades

Uso:
    python classificar_documento.py --arquivo relatorio.pdf
    python classificar_documento.py --arquivo aula.docx --titulo "Aula de Genética" --resumo "Estudo dos genes..."
    python classificar_documento.py --texto "Texto do material aqui..."

Dependências:
    pip install torch transformers numpy pymupdf python-docx
"""

import argparse
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pymupdf
import docx
import io


# ── Configurações ────────────────────────────────────────────────────────────

#MODELO = "neuralmind/bert-base-portuguese-cased"
MODELO = "rufimelo/bert-large-portuguese-cased-sts"
ARQUIVO_EMBEDDINGS = "../data/bncc_embeddings.npz"

CHUNK_SIZE = 400       # tokens por chunk
CHUNK_OVERLAP = 50     # tokens de sobreposição
MAX_TOKENS = 512       # limite do BERTimbau
TOP_K = 20              # habilidades retornadas
PESO_CHUNKS = 0.5      # peso da representação por chunks
PESO_RESUMO = 0.5      # peso da representação por título+resumo


# ── Carregamento do modelo ───────────────────────────────────────────────────

def carregar_modelo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando BERTimbau em {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    model = AutoModel.from_pretrained(MODELO).to(device)
    model.eval()
    return tokenizer, model, device


def carregar_embeddings_bncc():
    print(f"Carregando vetores das habilidades de {ARQUIVO_EMBEDDINGS} ...")
    data = np.load(ARQUIVO_EMBEDDINGS, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],   # (N, 768)
        "codigos":    data["codigos"],
        "areas":      data["areas"],
        "descricoes": data["descricoes"],
    }


# ── Extração de texto ────────────────────────────────────────────────────────

def extrair_texto_pdf(caminho: str) -> str:
    doc = pymupdf.open(caminho)
    texto = ""
    for page in doc:
        texto += page.get_text()
    doc.close()
    return texto.strip()


def extrair_texto_docx(caminho: str) -> str:
    doc = docx.Document(caminho)
    paragrafos = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragrafos).strip()


def extrair_texto(caminho: str) -> str:
    if caminho.lower().endswith(".pdf"):
        return extrair_texto_pdf(caminho)
    elif caminho.lower().endswith((".docx", ".doc")):
        return extrair_texto_docx(caminho)
    else:
        with open(caminho, "r", encoding="utf-8") as f:
            return f.read().strip()


# ── Chunking ─────────────────────────────────────────────────────────────────

def fazer_chunks(texto: str, tokenizer) -> list[str]:
    """
    Divide o texto em chunks de CHUNK_SIZE tokens com CHUNK_OVERLAP de sobreposição.
    Retorna lista de strings (chunks decodificados).
    """
    tokens = tokenizer.encode(texto, add_special_tokens=False)
    chunks = []
    inicio = 0

    while inicio < len(tokens):
        fim = min(inicio + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[inicio:fim]
        chunk_texto = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_texto)

        if fim == len(tokens):
            break
        inicio += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ── Embeddings ───────────────────────────────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).float()
    soma = (token_embeddings * mask_expanded).sum(dim=1)
    contagem = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return soma / contagem


def gerar_embedding(texto: str, tokenizer, model, device) -> np.ndarray:
    """Gera embedding de um único texto."""
    encoded = tokenizer(
        texto,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)

    emb = mean_pooling(output, encoded["attention_mask"])
    return emb.cpu().numpy()[0]  # (768,)


def embedding_por_chunks(texto: str, tokenizer, model, device) -> np.ndarray:
    """Chunking + média dos embeddings."""
    chunks = fazer_chunks(texto, tokenizer)
    print(f"  {len(chunks)} chunk(s) gerado(s)")

    embeddings = [gerar_embedding(c, tokenizer, model, device) for c in chunks]
    return np.mean(embeddings, axis=0)  # (768,)


# ── Similaridade ─────────────────────────────────────────────────────────────

def cosseno(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def classificar(vetor_doc: np.ndarray, bncc: dict, top_k: int = TOP_K) -> list[dict]:
    """Compara vetor do documento com todos os vetores das habilidades."""
    scores = []
    for i, emb_hab in enumerate(bncc["embeddings"]):
        score = cosseno(vetor_doc, emb_hab)
        scores.append((score, i))

    scores.sort(reverse=True)
    resultado = []
    for score, i in scores[:top_k]:
        resultado.append({
            "codigo":     bncc["codigos"][i],
            "area":       bncc["areas"][i],
            "descricao":  bncc["descricoes"][i],
            "confianca":  round(score, 4),
        })
    return resultado


# ── Pipeline principal ───────────────────────────────────────────────────────

def pipeline(
    texto: str,
    titulo: str = None,
    resumo: str = None,
    tokenizer=None,
    model=None,
    device=None,
    bncc: dict = None,
) -> dict:

    vetores = []
    pesos = []

    # Representação 1: chunks
    print("\n[1/2] Gerando embedding por chunks ...")
    vetor_chunks = embedding_por_chunks(texto, tokenizer, model, device)
    vetores.append(vetor_chunks)
    pesos.append(PESO_CHUNKS)

    # Representação 2: título + resumo
    texto_resumo = " ".join(filter(None, [titulo, resumo])).strip()
    if texto_resumo:
        print("\n[2/2] Gerando embedding de título + resumo ...")
        vetor_resumo = gerar_embedding(texto_resumo, tokenizer, model, device)
        vetores.append(vetor_resumo)
        pesos.append(PESO_RESUMO)

        # Convergência: similaridade entre as duas representações
        convergencia = cosseno(vetor_chunks, vetor_resumo)
        print(f"  Convergência entre representações: {convergencia:.3f}")
    else:
        print("\n[2/2] Título/resumo não fornecido — usando só chunks.")
        convergencia = None

    # Combinação ponderada
    pesos_norm = np.array(pesos) / sum(pesos)
    vetor_final = sum(p * v for p, v in zip(pesos_norm, vetores))

    # Classificação
    print(f"\nClassificando contra {len(bncc['embeddings'])} habilidades da BNCC ...")
    habilidades = classificar(vetor_final, bncc)

    return {
        "habilidades": habilidades,
        "convergencia": convergencia,
        "num_chunks": len(fazer_chunks(texto, tokenizer)),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classificador BNCC — Ensino Médio")
    parser.add_argument("--arquivo", help="Caminho para PDF ou Word")
    parser.add_argument("--texto",   help="Texto direto do material")
    parser.add_argument("--titulo",  help="Título do documento (opcional)")
    parser.add_argument("--resumo",  help="Resumo do documento (opcional)")
    parser.add_argument("--top",     type=int, default=TOP_K, help="Número de habilidades a retornar")
    args = parser.parse_args()

    if not args.arquivo and not args.texto:
        parser.error("Forneça --arquivo ou --texto.")

    # Carrega modelo e vetores
    tokenizer, model, device = carregar_modelo()
    bncc = carregar_embeddings_bncc()

    # Extrai texto
    if args.arquivo:
        print(f"\nExtraindo texto de {args.arquivo} ...")
        texto = extrair_texto(args.arquivo)
    else:
        texto = args.texto

    print(f"Texto extraído: {len(texto)} caracteres")

    # Roda pipeline
    resultado = pipeline(
        texto=texto,
        titulo=args.titulo,
        resumo=args.resumo,
        tokenizer=tokenizer,
        model=model,
        device=device,
        bncc=bncc,
    )

    # Exibe resultado
    print("\n" + "═" * 60)
    print(f"{'HABILIDADES IDENTIFICADAS':^60}")
    print("═" * 60)

    if resultado["convergencia"] is not None:
        conv = resultado["convergencia"]
        nivel = "Alta" if conv > 0.85 else "Média" if conv > 0.70 else "Baixa"
        print(f"Convergência entre representações: {conv:.3f} ({nivel})")
        if nivel == "Baixa":
            print("⚠ Baixa convergência — recomenda-se revisão humana.")
        print()

    for i, h in enumerate(resultado["habilidades"][:args.top], 1):
        print(f"{i}. [{h['codigo']}] {h['area']}")
        print(f"   Confiança: {h['confianca']:.3f}")
        print(f"   {h['descricao'][:120]}...")
        print()

    # Salva JSON
    saida = "resultado_classificacao.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"Resultado salvo em {saida}")


if __name__ == "__main__":
    main()
