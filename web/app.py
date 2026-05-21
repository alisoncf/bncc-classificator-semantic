from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
import os, sys, tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # web/
ROOT_DIR = os.path.dirname(BASE_DIR)                    # racine du projet
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

import classificar_documento as cd

cd.ARQUIVO_EMBEDDINGS = os.path.join(ROOT_DIR, "data", "bncc_embeddings.npz")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Chargement du modèle BERTimbau et des embeddings BNCC...")
    _state["tokenizer"], _state["model"], _state["device"] = cd.carregar_modelo()
    _state["bncc"] = cd.carregar_embeddings_bncc()
    print("Prêt.")
    yield


app = FastAPI(title="Classificateur BNCC", lifespan=lifespan)


@app.post("/api/classify")
async def classify(
    file: UploadFile = File(...),
    titulo: str = Form(""),
    resumo: str = Form(""),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".pdf", ".docx", ".doc", ".txt"):
        raise HTTPException(400, "Formato não suportado. Use PDF, Word ou TXT.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        texto = cd.extrair_texto(tmp_path)
        if not texto:
            raise HTTPException(422, "Não foi possível extrair o texto do documento.")

        resultado = cd.pipeline(
            texto=texto,
            titulo=titulo or None,
            resumo=resumo or None,
            tokenizer=_state["tokenizer"],
            model=_state["model"],
            device=_state["device"],
            bncc=_state["bncc"],
        )

        for h in resultado["habilidades"]:
            h["codigo"] = str(h["codigo"])
            h["area"] = str(h["area"])
            h["descricao"] = str(h["descricao"])

        return resultado

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")
