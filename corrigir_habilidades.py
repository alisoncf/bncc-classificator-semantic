import json, re

with open("data/bncc.json", encoding="utf-8") as f:
    d = json.load(f)

AREA_POR_PREFIXO = {
    "CHS": "Ciências Humanas e Sociais Aplicadas",
    "CNT": "Ciências da Natureza e suas Tecnologias",
    "MAT": "Matemática e suas Tecnologias",
    "LP":  "Linguagens e suas Tecnologias",
    "LGG": "Linguagens e suas Tecnologias",
}

for h in d["habilidades"]:
    prefixo = re.match(r"EM13([A-Z]+)", h["codigo"]).group(1)
    if prefixo in AREA_POR_PREFIXO:          # EM13CO fica como está, até você conferir
        h["area"] = AREA_POR_PREFIXO[prefixo]
    h["texto_embedding"] = h["descricao"]    # só a descrição, sem código nem área

with open("data/bncc.json", "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)