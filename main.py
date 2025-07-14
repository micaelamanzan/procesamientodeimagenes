# main.py
from fastapi import FastAPI, UploadFile, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse
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
import uuid
import shutil # Para limpiar directorios
import logging # Para logging

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración de la Aplicación FastAPI ---
app = FastAPI(
    title="Aplicación de Procesamiento de Imágenes",
    description="API para detectar colores y recolorear imágenes."
)

# Configuración de CORS: Ajusta esto a tus dominios de frontend en producción
# Por ejemplo: allow_origins=["https://tu-dominio.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite todas las origenes para desarrollo. ¡Cambiar en producción!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorios para archivos estáticos y temporales
# Asegúrate de que estos directorios existan o créalos al inicio de la app
STATIC_DIR = "static"
IMAGES_DIR = "Imagenes" # Para imágenes finales de resultado
TEMP_DIR = "temp_data"  # Para datos intermedios de procesamiento

# Crear directorios si no existen
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Montar directorios estáticos
app.mount(f"/{IMAGES_DIR}", StaticFiles(directory=IMAGES_DIR), name=IMAGES_DIR)
app.mount(f"/{STATIC_DIR}", StaticFiles(directory=STATIC_DIR), name=STATIC_DIR)

# Configuración de Jinja2Templates
templates = Jinja2Templates(directory="templates")

# --- Funciones Auxiliares para Manejo de Archivos Temporales ---
def save_temp_data(session_id: str, data: dict):
    """Guarda datos intermedios en un archivo JSON temporal."""
    filepath = os.path.join(TEMP_DIR, f"{session_id}.json")
    try:
        with open(filepath, "w") as f:
            # Convertir arrays de numpy a listas para serialización JSON
            serializable_data = {
                "labels": data["labels"].tolist(),
                "flat_arr": data["flat_arr"].tolist(),
                "shape": data["shape"]
            }
            json.dump(serializable_data, f)
        logger.info(f"SAVE_TEMP_DATA: Datos de sesión {session_id} guardados en {filepath}")
    except Exception as e:
        logger.error(f"SAVE_TEMP_DATA ERROR: Error al guardar datos temporales para sesión {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al guardar datos temporales: {e}"
        )

def load_temp_data(session_id: str) -> dict:
    """Carga datos intermedios desde un archivo JSON temporal."""
    filepath = os.path.join(TEMP_DIR, f"{session_id}.json")
    logger.info(f"LOAD_TEMP_DATA: Intentando cargar datos para sesión {session_id} desde {filepath}")
    if not os.path.exists(filepath):
        logger.warning(f"LOAD_TEMP_DATA WARNING: Archivo de sesión {session_id} NO ENCONTRADO en {filepath}. Contenido de {TEMP_DIR}: {os.listdir(TEMP_DIR)}") # Added dir listing
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Datos de sesión no encontrados o expirados. Por favor, sube la imagen de nuevo."
        )
    try:
        with open(filepath, "r") as f:
            loaded_data = json.load(f)
            # Convertir listas de vuelta a arrays de numpy
            loaded_data["labels"] = np.array(loaded_data["labels"])
            loaded_data["flat_arr"] = np.array(loaded_data["flat_arr"])
            logger.info(f"LOAD_TEMP_DATA: Datos de sesión {session_id} cargados exitosamente desde {filepath}")
            return loaded_data
    except json.JSONDecodeError:
        logger.error(f"LOAD_TEMP_DATA ERROR: Error al decodificar JSON de sesión {session_id} en {filepath}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al decodificar datos de sesión (JSON inválido)."
        )
    except Exception as e:
        logger.error(f"LOAD_TEMP_DATA ERROR: Error al cargar datos temporales para sesión {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al cargar datos temporales: {e}"
        )

def delete_temp_data(session_id: str):
    """Elimina el archivo JSON temporal."""
    filepath = os.path.join(TEMP_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            logger.info(f"DELETE_TEMP_DATA: Archivo temporal de sesión {session_id} eliminado: {filepath}")
        except Exception as e:
            logger.error(f"DELETE_TEMP_DATA ERROR: Fallo al eliminar archivo temporal {filepath}: {e}")

# --- Rutas de la API ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Sirve la página HTML principal de la aplicación.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detectar-colores")
async def detectar(file: UploadFile):
    """
    Detecta los colores principales en una imagen subida.
    Guarda los datos intermedios para el siguiente paso y devuelve un ID de sesión.
    """
    logger.info(f"Petición recibida para /detectar-colores. Tipo de contenido: {file.content_type}")
    # 1. Validación del tipo de archivo
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Archivo subido no es una imagen: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo subido no es una imagen válida."
        )

    try:
        file_bytes = await file.read()
        # 2. Procesamiento de la imagen para detección de colores
        # La función detectar_colores ahora devuelve solo los colores, labels, flat_arr y shape
        colores, labels, flat_arr, shape = detectar_colores(file_bytes)

        # 3. Generar un ID de sesión único
        session_id = str(uuid.uuid4())

        # 4. Guardar datos intermedios asociados al ID de sesión
        intermediate_data = {
            "labels": labels,
            "flat_arr": flat_arr,
            "shape": shape
        }
        save_temp_data(session_id, intermediate_data)

        # 5. Devolver los colores detectados y el ID de sesión
        logger.info(f"DETECTAR_COLORES: Sesión {session_id} generada y datos guardados.")
        return {"colores": colores, "session_id": session_id}

    except Image.UnidentifiedImageError:
        logger.error("Error: No se pudo identificar el archivo como una imagen válida.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se pudo identificar el archivo como una imagen válida. Asegúrate de que sea PNG o JPG."
        )
    except ValueError as ve:
        logger.error(f"Error de valor durante la detección de colores: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.exception("Error interno inesperado durante la detección de colores.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor al detectar colores: {e}"
        )

@app.post("/cambiar-colores-multiples")
async def cambiar_multiples(
    session_id: str = Form(...), # Ahora esperamos un session_id
    recolores: str = Form(...),
    quitar_fondo: bool = Form(False),
    preservar_blancos: bool = Form(False)
):
    """
    Aplica recoloreo a la imagen basándose en los datos de sesión y las nuevas configuraciones.
    """
    logger.info(f"CAMBIAR_COLORES_MULTIPLES: Petición recibida para sesión: {session_id}. Quitar Fondo: {quitar_fondo}, Preservar Blancos: {preservar_blancos}")
    try:
        # 1. Cargar datos intermedios usando el ID de sesión
        loaded_data = load_temp_data(session_id)
        labels = loaded_data["labels"]
        flat_arr = loaded_data["flat_arr"]
        shape = loaded_data["shape"]

        # 2. Parsear la configuración de recolores
        try:
            recolores_json = json.loads(recolores)
            logger.info(f"CAMBIAR_COLORES_MULTIPLES: Recolores JSON recibido para sesión {session_id}: {recolores_json}")
        except json.JSONDecodeError:
            logger.warning(f"CAMBIAR_COLORES_MULTIPLES WARNING: JSON inválido en 'recolores' para sesión {session_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El formato de 'recolores' no es un JSON válido."
            )

        # 3. Procesamiento de la imagen (recoloreo y máscara)
        # CORRECCIÓN: Pasar el argumento 'shape' a detectar_cluster_fondo
        fondo_cluster = detectar_cluster_fondo(flat_arr, labels, shape)
        mask = crear_mascara(labels, fondo_cluster, shape, preservar_blancos, flat_arr)

        recoloreado = aplicar_recoloreo(flat_arr, labels, recolores_json)

        if quitar_fondo:
            logger.info(f"CAMBIAR_COLORES_MULTIPLES: Aplicando eliminación de fondo para sesión {session_id}.")
            # Asegurarse de que recoloreado tenga 4 canales (RGBA) para la transparencia
            if recoloreado.shape[1] == 3: # Si es RGB, convertir a RGBA
                alpha_channel = np.full((recoloreado.shape[0], 1), 255, dtype=np.uint8)
                recoloreado = np.concatenate((recoloreado, alpha_channel), axis=1)
            recoloreado[mask.flatten() == 0] = [0, 0, 0, 0] # Establecer fondo a transparente

        imagen_final = recoloreado.reshape(shape)

        # 4. Guardar la imagen resultante
        output_filename = f"resultado_{uuid.uuid4()}.png"
        output_path = os.path.join(IMAGES_DIR, output_filename)
        imagen = Image.fromarray(imagen_final.astype(np.uint8), "RGBA")
        imagen.save(output_path)
        logger.info(f"CAMBIAR_COLORES_MULTIPLES: Imagen procesada guardada en {output_path} para sesión {session_id}")

        # 5. Eliminar los datos temporales después de usarlos
        delete_temp_data(session_id)

        # 6. Devolver la URL de la imagen resultante
        return {"url": f"/{IMAGES_DIR}/{output_filename}"}

    except HTTPException as e:
        logger.error(f"CAMBIAR_COLORES_MULTIPLES ERROR: HTTPException en /cambiar-colores-multiples para sesión {session_id}: {e.detail}")
        raise e
    except Exception as e:
        logger.exception(f"CAMBIAR_COLORES_MULTIPLES ERROR: Error interno inesperado en /cambiar-colores-multiples para sesión {session_id}.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor al cambiar colores: {e}"
        )

# --- Limpieza de archivos temporales al inicio (opcional, para desarrollo) ---
@app.on_event("startup")
async def startup_event():
    """
    Evento que se ejecuta al iniciar la aplicación.
    Limpia el contenido del directorio temporal.
    """
    logger.info(f"STARTUP_EVENT: Iniciando aplicación. Limpiando directorio temporal: {TEMP_DIR}")
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f'STARTUP_EVENT ERROR: Fallo al eliminar {file_path}. Razón: {e}')
