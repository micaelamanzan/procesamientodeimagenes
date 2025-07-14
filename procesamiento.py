import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans
import cv2
import io
import logging

logger = logging.getLogger(__name__)


def cargar_imagen(file_bytes):
    """
    Carga una imagen desde bytes, la convierte a RGBA y la redimensiona a una miniatura.
    Lanza ValueError si la imagen no puede ser cargada o es inválida.
    """
    try:
        # Usar Image.open con BytesIO para leer desde bytes
        image = Image.open(io.BytesIO(file_bytes)).convert("RGBA")
        # Redimensionar a una miniatura para un procesamiento más rápido
        # Mantiene la proporción original
        image.thumbnail((300, 300), Image.Resampling.LANCZOS)
        logger.info(f"CARGAR_IMAGEN: Imagen cargada y redimensionada a {image.size}.")
        return image
    except UnidentifiedImageError:
        logger.error("CARGAR_IMAGEN ERROR: No se pudo identificar el archivo como una imagen válida.")
        raise ValueError(
            "No se pudo cargar la imagen. Asegúrate de que sea un formato de imagen válido (PNG, JPG, etc.).")
    except Exception as e:
        logger.error(f"CARGAR_IMAGEN ERROR: Error inesperado al cargar la imagen: {e}")
        raise ValueError(f"Error al cargar la imagen: {e}")


def detectar_colores(file_bytes, n_clusters=8):
    """
    Detecta los colores principales en una imagen usando K-Means.
    Devuelve los colores en formato hexadecimal, las etiquetas de los píxeles,
    el array aplanado de la imagen y la forma original.
    """
    image = cargar_imagen(file_bytes)
    arr = np.array(image)

    if arr.size < 4:
        logger.warning("DETECTAR_COLORES WARNING: La imagen no contiene suficientes datos para procesar.")
        raise ValueError("La imagen es demasiado pequeña o no contiene suficientes datos para procesar.")

    original_shape = arr.shape
    flat_arr = arr.reshape((-1, 4))  # Aplanar la imagen a (píxeles, canales)
    logger.info(f"DETECTAR_COLORES: Imagen aplanada a forma {flat_arr.shape}.")

    unique_pixels = np.unique(flat_arr[:, :3], axis=0).shape[0]
    effective_n_clusters = min(n_clusters, unique_pixels if unique_pixels > 0 else 1)
    if effective_n_clusters == 0:
        logger.warning("DETECTAR_COLORES WARNING: No hay píxeles de color para agrupar.")
        return [], np.array([]), np.array([]).reshape(0, 4), (0, 0, 4)

    try:
        kmeans = KMeans(n_clusters=effective_n_clusters, n_init=10, random_state=0)
        kmeans.fit(flat_arr[:, :3])
        clustered_rgb = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        logger.info(
            f"DETECTAR_COLORES: K-Means completado. Clusters: {effective_n_clusters}, Etiquetas: {labels.shape}.")
    except Exception as e:
        logger.error(f"DETECTAR_COLORES ERROR: Error durante el clustering K-Means: {e}")
        raise ValueError(f"Error al detectar colores principales: {e}")

    colores_hex = ['#%02x%02x%02x' % tuple(color) for color in clustered_rgb]
    return colores_hex, labels, flat_arr, original_shape


def detectar_cluster_fondo(flat_arr, labels, shape):  # Añadido 'shape' como parámetro
    """
    Detecta el cluster que probablemente representa el fondo de la imagen.
    Ahora se basa en el muestreo de píxeles de los bordes de la imagen.
    """
    if flat_arr.size == 0 or labels.size == 0:
        logger.warning("DETECTAR_CLUSTER_FONDO WARNING: flat_arr o labels están vacíos.")
        return -1

    height, width, _ = shape

    # Muestrear píxeles de los bordes de la imagen
    # Tomar una pequeña franja de píxeles de cada borde
    border_pixels_indices = []
    # Borde superior
    for x in range(width):
        border_pixels_indices.append(x)  # (0,x) -> 0*width + x = x
    # Borde inferior
    for x in range(width):
        border_pixels_indices.append((height - 1) * width + x)
    # Borde izquierdo (excluyendo esquinas ya cubiertas)
    for y in range(1, height - 1):
        border_pixels_indices.append(y * width)
    # Borde derecho (excluyendo esquinas ya cubiertas)
    for y in range(1, height - 1):
        border_pixels_indices.append(y * width + (width - 1))

    # Asegurarse de que los índices sean únicos y válidos
    border_pixels_indices = np.unique(border_pixels_indices)
    border_pixels_indices = border_pixels_indices[
        border_pixels_indices < flat_arr.shape[0]]  # Filtrar índices fuera de rango

    if border_pixels_indices.size == 0:
        logger.warning("DETECTAR_CLUSTER_FONDO WARNING: No se encontraron píxeles de borde válidos.")
        # Fallback a la detección por píxel más claro si no hay bordes
        medios = np.mean(flat_arr[:, :3], axis=1)
        idx_fondo = np.argmax(medios)
        fondo_cluster = labels[idx_fondo]
        logger.info(f"DETECTAR_CLUSTER_FONDO: Fallback a píxel más claro. Cluster {fondo_cluster}.")
        return fondo_cluster

    # Obtener las etiquetas de los clusters de los píxeles de los bordes
    border_labels = labels[border_pixels_indices]

    # Contar la frecuencia de cada cluster en los bordes
    unique_labels, counts = np.unique(border_labels, return_counts=True)

    # El cluster más frecuente en los bordes es el candidato a fondo
    if counts.size > 0:
        fondo_cluster = unique_labels[np.argmax(counts)]
        logger.info(
            f"DETECTAR_CLUSTER_FONDO: Cluster más frecuente en los bordes es {fondo_cluster} (con {counts.max()} píxeles).")
        return fondo_cluster
    else:
        logger.warning(
            "DETECTAR_CLUSTER_FONDO WARNING: No se pudieron determinar clusters de fondo a partir de los bordes. Retornando -1.")
        return -1


def crear_mascara(labels, fondo_cluster, shape, preservar_blancos, flat_arr):
    """
    Crea una máscara binaria para separar el objeto del fondo.
    Los píxeles del objeto son 1, los del fondo son 0.
    Permite preservar los píxeles blancos si se especifica.
    """
    if labels.size == 0 or flat_arr.size == 0:
        logger.warning("CREAR_MASCARA WARNING: labels o flat_arr están vacíos.")
        # Retorna una máscara vacía del tamaño esperado si la forma es válida, de lo contrario un array vacío
        if len(shape) >= 2 and shape[0] is not None and shape[1] is not None:
            return np.zeros((shape[0], shape[1]), dtype=np.uint8)
        else:
            return np.array([])

    # Inicialmente, la máscara marca como 1 (objeto) todo lo que no es el cluster de fondo
    if fondo_cluster == -1:
        mask = np.ones(labels.shape, dtype=np.uint8)
        logger.info("CREAR_MASCARA: Fondo_cluster es -1, máscara inicial es todo 1s (no se quitará fondo).")
    else:
        mask = (labels != fondo_cluster).astype(np.uint8)
        logger.info(f"CREAR_MASCARA: Máscara inicial creada. Píxeles del fondo ({fondo_cluster}) son 0, otros 1.")

    if preservar_blancos:
        blancos = np.all(flat_arr[:, :3] > 240, axis=1)
        mask[blancos] = 1
        logger.info("CREAR_MASCARA: Preservando blancos. Píxeles blancos marcados como 1 en la máscara.")

    # Asegurarse de que el reshape sea válido
    if len(shape) < 2 or shape[0] is None or shape[1] is None or shape[0] * shape[1] != mask.size:
        logger.error(
            f"CREAR_MASCARA ERROR: Forma de imagen inválida para reshape: {shape}. Tamaño de máscara: {mask.size}")
        # Intentar redimensionar si es posible, de lo contrario, retornar una máscara vacía
        if len(shape) >= 2 and shape[0] is not None and shape[1] is not None:
            return np.zeros((shape[0], shape[1]), dtype=np.uint8)
        else:
            return np.array([])  # Retorna un array vacío si la forma es completamente inválida

    mask = mask.reshape((shape[0], shape[1]))
    logger.info(f"CREAR_MASCARA: Máscara redimensionada a {mask.shape}.")

    # Aplicar operaciones morfológicas para limpiar la máscara (cerrar pequeños huecos)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    logger.info("CREAR_MASCARA: Operación morfológica de cierre aplicada a la máscara.")
    return mask


def aplicar_recoloreo(flat_arr, labels, recolores_map):
    """
    Aplica los cambios de color a la imagen basándose en los clusters y los nuevos colores.
    """
    if flat_arr.size == 0 or labels.size == 0:
        logger.warning("APLICAR_RECOLOREO WARNING: flat_arr o labels están vacíos.")
        return np.array([]).reshape(0, 4)

    recoloreado = flat_arr.copy()

    if not recolores_map:  # Check if recolores_map is empty
        logger.info("APLICAR_RECOLOREO: recolores_map está vacío. No se aplicarán cambios de color.")
        return recoloreado  # Return original if no color changes are requested

    for item in recolores_map:
        cluster_id = item["cluster"]
        color_hex = item["color"]
        color_rgb = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))

        if np.any(labels == cluster_id):
            recoloreado[labels == cluster_id, :3] = color_rgb
            logger.info(f"APLICAR_RECOLOREO: Color del cluster {cluster_id} cambiado a {color_hex}.")
        else:
            logger.warning(
                f"APLICAR_RECOLOREO WARNING: Cluster ID {cluster_id} no encontrado en las etiquetas. Saltando recoloreo.")

    return recoloreado