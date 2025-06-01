from PIL import Image, ImageFilter
import numpy as np
import os
import sys
from sklearn.cluster import KMeans  # Necesario para K-means
import cv2  # Necesario para operaciones morfológicas de OpenCV

# Define el tamaño máximo para redimensionar imágenes grandes
MAX_SIZE = (800, 800)


# Función para convertir un color HEX a formato RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')  # Elimina el '#' inicial si existe
    if len(hex_color) != 6:
        raise ValueError("El código HEX debe tener 6 caracteres.")
    # Convierte cada par de caracteres HEX a su valor decimal
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


# 1. Solicitar la ruta de la imagen al usuario
ruta_img = input("📂 Ingrese la ruta de la imagen (por defecto: Imagenes/reloj.jpeg): ").strip()
if ruta_img == "":
    ruta_img = os.path.join("Imagenes", "reloj.jpeg")  # Ruta por defecto

# Verificar si el archivo de imagen existe en la ruta proporcionada
if not os.path.isfile(ruta_img):
    print(f"❌ No se encontró la imagen en: {ruta_img}")
    exit()  # Sale del programa si la imagen no se encuentra

# 2. Cargar la imagen y redimensionarla si excede el tamaño máximo
img = Image.open(ruta_img).convert("RGBA")  # Carga la imagen y la convierte a formato RGBA (con canal alfa)
width, height = img.size  # Obtiene las dimensiones originales antes de redimensionar
# Si la imagen es más grande que el tamaño máximo permitido, la redimensiona
if img.size[0] > MAX_SIZE[0] or img.size[1] > MAX_SIZE[1]:
    img.thumbnail(MAX_SIZE, Image.LANCZOS)  # Redimensiona usando un filtro de alta calidad
    # Actualiza width y height para el nuevo tamaño
    width, height = img.size

# Convierte la imagen a un array de NumPy para manipulación de píxeles
pixels = np.array(img)

# --- INICIO: SEGMENTACIÓN DEL OBJETO CON K-MEANS ---

# Reformar la imagen para K-means: cada píxel es un punto en el espacio de color (RGB)
data = pixels[:, :, 0:3].reshape(-1, 3).astype(np.float64)

# Número de clusters (grupos de colores) que K-means intentará encontrar.
n_clusters = 4
print("\n🧠 Realizando segmentación por colores (K-means)... esto puede tomar un momento.")
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
kmeans.fit(data)
labels = kmeans.labels_  # Etiqueta de cluster para cada píxel
centers = kmeans.cluster_centers_  # Colores promedio de cada cluster (RGB)
print("🧠 Segmentación K-means completada.")

# Crear una máscara inicial vacía
mask_np = np.zeros((height, width), dtype=np.uint8)

# Identificar los cluster(s) que pertenecen al fondo (basado en los píxeles de las esquinas)
corner_pixels_colors = [
    pixels[0, 0, 0:3],  # Arriba-izquierda
    pixels[0, width - 1, 0:3],  # Arriba-derecha
    pixels[height - 1, 0, 0:3],  # Abajo-izquierda
    pixels[height - 1, width - 1, 0:3]  # Abajo-derecha
]

background_cluster_labels = set()
for color in corner_pixels_colors:
    background_cluster_labels.add(kmeans.predict([color.astype(np.float64)])[0])

# Llenar la máscara: los píxeles que pertenecen a los clusters de fondo se marcan con 255
is_background_pixels_raw = np.isin(labels, list(background_cluster_labels)).reshape(height, width)
mask_np[is_background_pixels_raw] = 255

# Aplicar operaciones morfológicas y suavizado a la máscara para limpiarla
kernel_open = np.ones((5, 5), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)

mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel_open, iterations=1)
mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close, iterations=1)
mask_np = cv2.GaussianBlur(mask_np, (5, 5), 0)

# --- FIN: SEGMENTACIÓN DEL OBJETO CON K-MEANS ---


# --- INICIO: PREGUNTAS DE TRANSFORMACIÓN Y APLICACIÓN DE CAMBIOS ---

# Preguntar si se desea cambiar el color del objeto
cambiar_color_input = input("\n🎨 ¿Deseas cambiar el color del objeto? (sí/no): ").strip().lower()
cambiar_color = cambiar_color_input in ['si', 'sí', 's', 'yes', 'y']

nuevo_color_objeto = None
if cambiar_color:
    # Solicitar el color HEX para pintar el OBJETO solo si se elige cambiar el color
    hex_color = input("🎨 Ingrese un color HEX para pintar el OBJETO (ejemplo: #2471a3): ").strip()
    try:
        nuevo_color_objeto = np.array(hex_to_rgb(hex_color)).astype(np.float64)
    except Exception as e:
        print(f"❌ Error en el color HEX para el objeto: {e}")
        exit()

# Preguntar si se desea quitar el fondo
quitar_fondo_input = input("❓ ¿Deseas quitar el fondo (hacerlo transparente)? (sí/no): ").strip().lower()
quitar_fondo = quitar_fondo_input in ['si', 'sí', 's', 'yes', 'y']

# Crear un nuevo array de píxeles para el resultado, trabajando en float64
result_pixels = np.copy(pixels).astype(np.float64)

# 3. Procesar cada píxel de la imagen con barra de progreso
print("\n🔄 Aplicando transformaciones a la imagen...")
for i in range(height):  # Usamos 'height' como el total de filas
    # Calcula el porcentaje de progreso y lo imprime en la misma línea
    porcentaje = int((i + 1) / height * 100)
    print(f"\rProgreso: {porcentaje}% completado", end="")
    sys.stdout.flush()  # Fuerza la actualización de la salida en consola

    for j in range(width):  # Iteramos por columnas
        r_orig, g_orig, b_orig, a_orig = pixels[i, j]

        mask_value = mask_np[i, j]  # El valor de la máscara para este píxel (0=objeto, 255=fondo)

        # Si el píxel ya era transparente en la imagen original, y no vamos a quitar el fondo, lo mantenemos así
        if a_orig == 0 and not quitar_fondo:
            continue

        # Calcular el factor de opacidad del objeto (0=fondo puro, 1=objeto puro) a partir del valor de la máscara
        object_opacity_factor = 1.0 - (mask_value / 255.0)
        object_opacity_factor = max(0.0, min(1.0, object_opacity_factor))  # Clampear entre 0 y 1

        # --- APLICACIÓN DE ALFA (TRANSPARENCIA) ---
        if quitar_fondo:
            result_pixels[i, j, 3] = a_orig * object_opacity_factor
        else:
            result_pixels[i, j, 3] = a_orig

        # --- APLICACIÓN DE COLOR ---
        if object_opacity_factor > 0:  # Si hay alguna presencia del objeto en este píxel
            if cambiar_color:
                color_para_pintar_objeto_mezcla = nuevo_color_objeto

                # Mezcla del nuevo color con el color original del píxel
                result_pixels[i, j, 0] = (color_para_pintar_objeto_mezcla[0] * object_opacity_factor) + (
                            r_orig * (1 - object_opacity_factor))
                result_pixels[i, j, 1] = (color_para_pintar_objeto_mezcla[1] * object_opacity_factor) + (
                            g_orig * (1 - object_opacity_factor))
                result_pixels[i, j, 2] = (color_para_pintar_objeto_mezcla[2] * object_opacity_factor) + (
                            b_orig * (1 - object_opacity_factor))
            # Si no se quiere cambiar el color, el color RGB se mantiene como el original (ya lo es en result_pixels)
        # Si object_opacity_factor es 0 (fondo puro), el color RGB se mantiene como el original del fondo.

# Convertir el array de píxeles modificado de float a uint8 antes de crear la imagen PIL
result_pixels = result_pixels.astype(np.uint8)
new_img = Image.fromarray(result_pixels, mode="RGBA")

# Opcional: Aplicar un filtro de suavizado final si aún hay aspereza en los bordes
# new_img = new_img.filter(ImageFilter.SMOOTH)
# new_img = new_img.filter(ImageFilter.BoxBlur(0.5))

# 4. Solicitar el nombre del archivo de salida y guardar la imagen
nombre_archivo = input(
    "\n💾 Ingrese un nombre para la imagen de salida (sin extensión, por defecto: imagen_transformada): ").strip()
if nombre_archivo == "":
    nombre_archivo = "imagen_transformada"  # Nombre por defecto si no se ingresa nada

output_dir = "Imagenes"  # Directorio de salida
os.makedirs(output_dir, exist_ok=True)  # Crea el directorio si no existe
output_path = os.path.join(output_dir, f"{nombre_archivo}.png")  # Construye la ruta completa del archivo

new_img.save(output_path)  # Guarda la imagen en la ruta especificada

# Muestra la ruta donde se guardó la imagen y la abre
print(f"\n✅ Imagen procesada guardada en: {output_path}")
new_img.show()  # Abre la imagen con el visor predeterminado del sistema