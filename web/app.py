from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import os, sys, tempfile
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # web/
ROOT_DIR = os.path.dirname(BASE_DIR)                    # racine du projet
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

import classificar_documento as cd

cd.ARQUIVO_EMBEDDINGS = os.path.join(ROOT_DIR, "data", "bncc_embeddings.npz")

MAX_FILE_SIZE   = 10 * 1024 * 1024   # 10 Mo
MAX_TITULO_LEN  = 500
MAX_RESUMO_LEN  = 5_000

_state: dict = {}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]        = "DENY"
        response.headers["Referrer-Policy"]        = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "object-src 'none';"
        )
        return response


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("Chargement du modèle BERTimbau et des embeddings BNCC...")
    _state["tokenizer"], _state["model"], _state["device"] = cd.carregar_modelo()
    _state["bncc"] = cd.carregar_embeddings_bncc()
    print("Prêt.")
    yield


app = FastAPI(title="Classificateur BNCC", lifespan=lifespan)
app.add_middleware(SecurityHeadersMiddleware)


@app.post("/api/classify")
async def classify(
    file: UploadFile = File(...),
    titulo: str = Form(""),
    resumo: str = Form(""),
):
    if len(titulo) > MAX_TITULO_LEN:
        raise HTTPException(400, f"Título muito longo (máx. {MAX_TITULO_LEN} caracteres).")
    if len(resumo) > MAX_RESUMO_LEN:
        raise HTTPException(400, f"Resumo muito longo (máx. {MAX_RESUMO_LEN} caracteres).")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".pdf", ".docx", ".doc", ".txt"):
        raise HTTPException(400, "Formato não suportado. Use PDF, Word ou TXT.")

    tmp_path = None
    try:
        content = await file.read(MAX_FILE_SIZE + 1)
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, f"Arquivo muito grande (máx. {MAX_FILE_SIZE // 1024 // 1024} Mo).")

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
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
