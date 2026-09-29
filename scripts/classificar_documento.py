"""
Classifica um documento nas habilidades da BNCC (Educação Infantil,
Ensino Fundamental e Ensino Médio).

Estratégia:
  - Representação 1: chunking com overlap → cada chunk é comparado com as
    habilidades; o score de uma habilidade é a média dos TOP_CHUNKS chunks
    mais similares a ela (um trecho forte não se dilui no resto do documento)
  - Representação 2: título + resumo (se disponível)
  - Combinação: média ponderada dos scores das duas representações
  - Similaridade de cosseno com multilingual-e5 (prefixos "query:"/"passage:")

Uso:
    python classificar_documento.py --arquivo relatorio.pdf
    python classificar_documento.py --arquivo aula.docx --titulo "Aula de Genética" --resumo "Estudo dos genes..."
    python classificar_documento.py --texto "Texto do material aqui..."
    python classificar_documento.py --arquivo aula.pdf --etapas EF EM

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


# ── Configurações ────────────────────────────────────────────────────────────

# Deve ser o mesmo modelo usado em gerar_embeddings_bncc.py
MODELO = "intfloat/multilingual-e5-base"
PREFIXO_DOC = "query: "   # e5: textos do documento são "query", habilidades são "passage"
ARQUIVO_EMBEDDINGS = "../data/bncc_embeddings.npz"

CHUNK_SIZE = 400       # tokens por chunk
CHUNK_OVERLAP = 50     # tokens de sobreposição
MAX_TOKENS = 512       # limite do modelo
TOP_K = 20              # habilidades retornadas
TOP_CHUNKS = 3         # chunks mais similares considerados por habilidade
PESO_CHUNKS = 0.5      # peso da representação por chunks
PESO_RESUMO = 0.5      # peso da representação por título+resumo

# Convergência = correlação de Spearman entre os rankings de habilidades gerados
# pelos chunks e pelo título+resumo. Calibração inicial (e5-base): resumo correto
# ≈ 0.74; resumos de outras disciplinas ≈ 0.01–0.48.
LIMIAR_CONV_ALTA = 0.65
LIMIAR_CONV_MEDIA = 0.50

# Etapas da BNCC, identificadas pelo prefixo do código da habilidade
ETAPAS = {
    "EI": "Educação Infantil",
    "EF": "Ensino Fundamental",
    "EM": "Ensino Médio",
}


# ── Carregamento do modelo ───────────────────────────────────────────────────

def carregar_modelo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando {MODELO} em {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    model = AutoModel.from_pretrained(MODELO).to(device)
    model.eval()
    return tokenizer, model, device


def carregar_embeddings_bncc():
    print(f"Carregando vetores das habilidades de {ARQUIVO_EMBEDDINGS} ...")
    data = np.load(ARQUIVO_EMBEDDINGS, allow_pickle=False)
    modelo_vetores = str(data["modelo"]) if "modelo" in data else "desconhecido"
    if modelo_vetores != MODELO:
        raise RuntimeError(
            f"{ARQUIVO_EMBEDDINGS} foi gerado com o modelo '{modelo_vetores}', "
            f"mas a classificação usa '{MODELO}'. Rode gerar_embeddings_bncc.py novamente."
        )
    return {
        "embeddings": data["embeddings"],   # (N, dim)
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
    """Gera embedding de um único texto do documento."""
    encoded = tokenizer(
        PREFIXO_DOC + texto,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)

    emb = mean_pooling(output, encoded["attention_mask"])
    return emb.cpu().numpy()[0]  # (dim,)


def embeddings_por_chunks(texto: str, tokenizer, model, device) -> np.ndarray:
    """Chunking + embedding de cada chunk."""
    chunks = fazer_chunks(texto, tokenizer)
    print(f"  {len(chunks)} chunk(s) gerado(s)")

    embeddings = [gerar_embedding(c, tokenizer, model, device) for c in chunks]
    return np.array(embeddings)  # (n_chunks, dim)


# ── Similaridade ─────────────────────────────────────────────────────────────

def similaridades(vetores: np.ndarray, bncc: dict) -> np.ndarray:
    """Cosseno de cada vetor (linhas) contra todas as habilidades → (n_vetores, n_habilidades)."""
    hab = bncc["embeddings"] / np.linalg.norm(bncc["embeddings"], axis=1, keepdims=True)
    vet = vetores / np.linalg.norm(vetores, axis=1, keepdims=True)
    return vet @ hab.T


def mascara_etapas(bncc: dict, etapas: list[str] = None) -> np.ndarray:
    """True para as habilidades das etapas escolhidas (todas, se etapas for None)."""
    if not etapas:
        return np.ones(len(bncc["codigos"]), dtype=bool)
    return np.array([str(c)[:2] in etapas for c in bncc["codigos"]])


def correlacao_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Correlação entre as ordenações de a e b (1 = mesma ordem, 0 = sem relação)."""
    posto_a = np.argsort(np.argsort(a))
    posto_b = np.argsort(np.argsort(b))
    return float(np.corrcoef(posto_a, posto_b)[0, 1])


def nivel_convergencia(conv: float) -> str:
    return "Alta" if conv >= LIMIAR_CONV_ALTA else "Média" if conv >= LIMIAR_CONV_MEDIA else "Baixa"


def classificar(scores_hab: np.ndarray, bncc: dict, top_k: int = TOP_K,
                etapas: list[str] = None) -> list[dict]:
    """
    Ordena as habilidades das etapas escolhidas (todas, se etapas for None) pelo score.

    "confianca" é o cosseno bruto — com o e5 fica numa faixa estreita (~0.80–0.85),
    então também é retornada "relevancia": 1.0 para a melhor habilidade e 0.0 para
    a mediana das candidatas, que diferencia melhor os resultados.
    """
    candidatos = np.flatnonzero(mascara_etapas(bncc, etapas))
    ordem = candidatos[np.argsort(-scores_hab[candidatos])]

    melhor = scores_hab[ordem[0]]
    mediana = np.median(scores_hab[candidatos])
    amplitude = max(melhor - mediana, 1e-9)

    resultado = []
    for i in ordem[:top_k]:
        score = float(scores_hab[i])
        resultado.append({
            "codigo":     bncc["codigos"][i],
            "area":       bncc["areas"][i],
            "etapa":      ETAPAS.get(str(bncc["codigos"][i])[:2], ""),
            "descricao":  bncc["descricoes"][i],
            "confianca":  round(score, 4),
            "relevancia": round(max(score - mediana, 0.0) / amplitude, 4),
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
    etapas: list[str] = None,
) -> dict:

    scores = []
    pesos = []

    # Representação 1: chunks — score = média dos TOP_CHUNKS chunks mais similares
    print("\n[1/2] Gerando embeddings por chunks ...")
    vetores_chunks = embeddings_por_chunks(texto, tokenizer, model, device)
    sim_chunks = similaridades(vetores_chunks, bncc)
    k = min(TOP_CHUNKS, len(vetores_chunks))
    scores.append(np.sort(sim_chunks, axis=0)[-k:].mean(axis=0))
    pesos.append(PESO_CHUNKS)

    # Representação 2: título + resumo (dividido em chunks se passar de MAX_TOKENS,
    # para não truncar resumos longos)
    texto_resumo = " ".join(filter(None, [titulo, resumo])).strip()
    if texto_resumo:
        print("\n[2/2] Gerando embedding de título + resumo ...")
        vetor_resumo = embeddings_por_chunks(texto_resumo, tokenizer, model, device).mean(axis=0)
        scores.append(similaridades(vetor_resumo[np.newaxis], bncc)[0])
        pesos.append(PESO_RESUMO)

        # Convergência: as duas representações ordenam as habilidades do mesmo jeito?
        mascara = mascara_etapas(bncc, etapas)
        convergencia = correlacao_spearman(scores[0][mascara], scores[1][mascara])
        nivel = nivel_convergencia(convergencia)
        print(f"  Convergência entre representações: {convergencia:.3f} ({nivel})")
    else:
        print("\n[2/2] Título/resumo não fornecido — usando só chunks.")
        convergencia = nivel = None

    # Combinação ponderada
    pesos_norm = np.array(pesos) / sum(pesos)
    score_final = sum(p * s for p, s in zip(pesos_norm, scores))

    # Classificação
    alvo = ", ".join(ETAPAS[e] for e in etapas) if etapas else "todas as etapas"
    print(f"\nClassificando contra as habilidades da BNCC ({alvo}) ...")
    habilidades = classificar(score_final, bncc, etapas=etapas)

    return {
        "habilidades": habilidades,
        "convergencia": convergencia,
        "convergencia_nivel": nivel,
        "num_chunks": len(vetores_chunks),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classificador BNCC")
    parser.add_argument("--arquivo", help="Caminho para PDF ou Word")
    parser.add_argument("--texto",   help="Texto direto do material")
    parser.add_argument("--titulo",  help="Título do documento (opcional)")
    parser.add_argument("--resumo",  help="Resumo do documento (opcional)")
    parser.add_argument("--top",     type=int, default=TOP_K, help="Número de habilidades a retornar")
    parser.add_argument("--etapas",  nargs="+", choices=list(ETAPAS),
                        help="Etapas consideradas (padrão: todas). Ex.: --etapas EF EM")
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
        etapas=args.etapas,
    )

    # Exibe resultado
    print("\n" + "═" * 60)
    print(f"{'HABILIDADES IDENTIFICADAS':^60}")
    print("═" * 60)

    if resultado["convergencia"] is not None:
        conv = resultado["convergencia"]
        nivel = resultado["convergencia_nivel"]
        print(f"Convergência entre representações: {conv:.3f} ({nivel})")
        if nivel == "Baixa":
            print("⚠ Baixa convergência — recomenda-se revisão humana.")
        print()

    for i, h in enumerate(resultado["habilidades"][:args.top], 1):
        print(f"{i}. [{h['codigo']}] {h['etapa']} · {h['area']}")
        print(f"   Relevância: {h['relevancia']:.0%}  (cosseno {h['confianca']:.3f})")
        print(f"   {h['descricao'][:120]}...")
        print()

    # Salva JSON
    saida = "resultado_classificacao.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"Resultado salvo em {saida}")


if __name__ == "__main__":
    main()
