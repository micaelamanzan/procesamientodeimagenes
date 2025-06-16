import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans
import cv2
import io

def cargar_imagen(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGBA")
        image.thumbnail((300, 300))  # conserva proporciones
        return image
    except UnidentifiedImageError:
        raise ValueError("No se pudo cargar la imagen.")

def detectar_colores(file_bytes, n_clusters=5):
    image = cargar_imagen(file_bytes)
    arr = np.array(image)
    if arr.size < 4:
        raise ValueError("La imagen no contiene suficientes datos para procesar.")
    original_shape = arr.shape
    flat_arr = arr.reshape((-1, 4))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    kmeans.fit(flat_arr[:, :3])
    clustered = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    colores_hex = ['#%02x%02x%02x' % tuple(color) for color in clustered]
    return colores_hex, image, labels, flat_arr, original_shape, clustered

def detectar_cluster_fondo(flat_arr, labels):
    medios = np.mean(flat_arr[:, :3], axis=1)
    idx_fondo = np.argmax(medios)
    return labels[idx_fondo]

def crear_mascara(labels, fondo_cluster, shape, preservar_blancos, flat_arr):
    mask = (labels != fondo_cluster).astype(np.uint8)
    if preservar_blancos:
        blancos = np.all(flat_arr[:, :3] > 240, axis=1)
        mask[blancos] = 1
    mask = mask.reshape((shape[0], shape[1]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask

def aplicar_recoloreo(flat_arr, labels, clustered, recolores):
    recoloreado = flat_arr.copy()
    for item in recolores:
        cluster_id = item["cluster"]
        color_hex = item["color"]
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        recoloreado[labels == cluster_id, :3] = color_rgb
    return recoloreado
