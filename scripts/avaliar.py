"""
Avalia a qualidade do pipeline de classificação usando documentos
já rotulados manualmente.

Formato esperado do arquivo de avaliação (data/rotulados.json):
[
  {
    "arquivo": "relatorio_estagio_01.pdf",
    "titulo": "Relatório de Estágio — Física",
    "resumo": "Atividades desenvolvidas em laboratório...",
    "habilidades_corretas": ["EM13CNT101", "EM13CNT301"]
  },
  ...
]

Uso:
    python avaliar.py
    python avaliar.py --top 5 --rotulados data/rotulados.json

Métricas geradas:
    - Precisão, Recall e F1 por habilidade
    - Métricas macro e micro
    - Relatório salvo em outputs/avaliacao.json
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# Importa funções do pipeline principal
import sys
sys.path.append(str(Path(__file__).parent))
from classificar_documento import (
    carregar_modelo,
    extrair_texto,
    pipeline,
)


# ── Configurações ─────────────────────────────────────────────────────────────

MODELO = "neuralmind/bert-base-portuguese-cased"
ARQUIVO_EMBEDDINGS = "../data/bncc_embeddings.npz"
ARQUIVO_ROTULADOS  = "../data/rotulados.json"
ARQUIVO_SAIDA      = "../outputs/avaliacao.json"
TOP_K_PADRAO       = 5


# ── Métricas multi-label ─────────────────────────────────────────────────────

def calcular_metricas(corretos: set, preditos: set) -> dict:
    """Precisão, recall e F1 para um documento."""
    if not preditos:
        return {"precisao": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(corretos & preditos)
    precisao = tp / len(preditos) if preditos else 0.0
    recall   = tp / len(corretos) if corretos else 0.0
    f1 = (2 * precisao * recall / (precisao + recall)
          if (precisao + recall) > 0 else 0.0)

    return {"precisao": precisao, "recall": recall, "f1": f1}


def metricas_globais(resultados: list[dict]) -> dict:
    """Macro-média das métricas sobre todos os documentos."""
    precisoes = [r["metricas"]["precisao"] for r in resultados]
    recalls   = [r["metricas"]["recall"]   for r in resultados]
    f1s       = [r["metricas"]["f1"]       for r in resultados]

    return {
        "precisao_macro": round(float(np.mean(precisoes)), 4),
        "recall_macro":   round(float(np.mean(recalls)),   4),
        "f1_macro":       round(float(np.mean(f1s)),       4),
        "n_documentos":   len(resultados),
    }


# ── Avaliação ────────────────────────────────────────────────────────────────

def avaliar(args):
    # Carrega modelo e vetores das habilidades
    tokenizer, model, device = carregar_modelo()
    bncc = np.load(ARQUIVO_EMBEDDINGS, allow_pickle=True)
    bncc_dict = {
        "embeddings": bncc["embeddings"],
        "codigos":    bncc["codigos"],
        "areas":      bncc["areas"],
        "descricoes": bncc["descricoes"],
    }

    # Carrega documentos rotulados
    with open(args.rotulados, "r", encoding="utf-8") as f:
        rotulados = json.load(f)

    print(f"\n{len(rotulados)} documentos para avaliar | top-{args.top}\n")
    print("─" * 60)

    resultados = []

    for i, doc in enumerate(rotulados, 1):
        print(f"[{i}/{len(rotulados)}] {doc.get('arquivo', 'texto direto')}")

        # Extrai texto
        if "arquivo" in doc:
            caminho = Path(args.rotulados).parent / doc["arquivo"]
            texto = extrair_texto(str(caminho))
        else:
            texto = doc.get("texto", "")

        if not texto.strip():
            print("  ⚠ Sem texto — pulando.")
            continue

        # Classifica
        resultado = pipeline(
            texto=texto,
            titulo=doc.get("titulo"),
            resumo=doc.get("resumo"),
            tokenizer=tokenizer,
            model=model,
            device=device,
            bncc=bncc_dict,
        )

        # Compara predições com gabarito
        preditos  = {h["codigo"] for h in resultado["habilidades"][:args.top]}
        corretos  = set(doc["habilidades_corretas"])
        metricas  = calcular_metricas(corretos, preditos)

        acertos   = corretos & preditos
        erros     = preditos - corretos
        perdidos  = corretos - preditos

        print(f"  Corretos: {sorted(corretos)}")
        print(f"  Preditos: {sorted(preditos)}")
        print(f"  ✓ Acertos: {sorted(acertos)}")
        print(f"  ✗ Falsos positivos: {sorted(erros)}")
        print(f"  ✗ Não encontrados:  {sorted(perdidos)}")
        print(f"  P={metricas['precisao']:.2f}  R={metricas['recall']:.2f}  F1={metricas['f1']:.2f}")
        if resultado["convergencia"] is not None:
            print(f"  Convergência: {resultado['convergencia']:.3f}")
        print()

        resultados.append({
            "documento":   doc.get("arquivo", f"doc_{i}"),
            "corretos":    sorted(corretos),
            "preditos":    sorted(preditos),
            "acertos":     sorted(acertos),
            "metricas":    metricas,
            "convergencia": resultado["convergencia"],
        })

    # Métricas globais
    globais = metricas_globais(resultados)
    print("═" * 60)
    print("RESULTADO GERAL")
    print(f"  Precisão macro: {globais['precisao_macro']:.4f}")
    print(f"  Recall macro:   {globais['recall_macro']:.4f}")
    print(f"  F1 macro:       {globais['f1_macro']:.4f}")
    print(f"  Documentos:     {globais['n_documentos']}")
    print("═" * 60)

    # Salva relatório
    relatorio = {"global": globais, "documentos": resultados}
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório salvo em {args.saida}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Avaliação do pipeline BNCC")
    parser.add_argument("--rotulados", default=ARQUIVO_ROTULADOS,
                        help="JSON com documentos rotulados")
    parser.add_argument("--top",  type=int, default=TOP_K_PADRAO,
                        help="Top-k habilidades consideradas na avaliação")
    parser.add_argument("--saida", default=ARQUIVO_SAIDA,
                        help="Arquivo de saída com o relatório")
    args = parser.parse_args()
    avaliar(args)


if __name__ == "__main__":
    main()
