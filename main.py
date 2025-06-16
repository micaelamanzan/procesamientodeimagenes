from fastapi import FastAPI, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from procesamiento import (
    detectar_colores,
    aplicar_recoloreo,
    detectar_cluster_fondo,
    crear_mascara
)
import os
import numpy as np
from PIL import Image
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/Imagenes", StaticFiles(directory="Imagenes"), name="Imagenes")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

global_store = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detectar-colores")
async def detectar(file: UploadFile):
    file_bytes = await file.read()
    colores, image, labels, flat_arr, shape, clustered = detectar_colores(file_bytes)

    global_store["data"] = (file_bytes, labels, flat_arr, shape)
    return {"colores": colores}

@app.post("/cambiar-colores-multiples")
async def cambiar_multiples(
    recolores: str = Form(...),
    quitar_fondo: bool = Form(False),
    preservar_blancos: bool = Form(False)
):
    file_bytes, labels, flat_arr, shape = global_store["data"]
    recolores_json = json.loads(recolores)

    fondo_cluster = detectar_cluster_fondo(flat_arr, labels)
    mask = crear_mascara(labels, fondo_cluster, shape, preservar_blancos, flat_arr)

    recoloreado = aplicar_recoloreo(flat_arr, labels, None, recolores_json)
    if quitar_fondo:
        recoloreado[mask.flatten() == 0] = [0, 0, 0, 0]

    imagen_final = recoloreado.reshape(shape)
    imagen = Image.fromarray(imagen_final.astype(np.uint8), "RGBA")

    os.makedirs("Imagenes", exist_ok=True)
    output_path = os.path.join("Imagenes", "resultado.png")
    imagen.save(output_path)

    return {"url": "/Imagenes/resultado.png"}
