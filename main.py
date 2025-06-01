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
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


# Función para convertir RGB a HEX (útil para mostrar colores al usuario)
def rgb_to_hex(rgb_color):
    # Asegúrate de que los valores sean enteros antes de formatear a HEX y que estén entre 0 y 255
    r, g, b = np.clip(rgb_color, 0, 255).astype(int)
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


# 1. Solicitar la ruta de la imagen al usuario
ruta_img = input("📂 Ingrese la ruta de la imagen (por defecto: Imagenes/reloj.jpeg): ").strip()
if ruta_img == "":
    ruta_img = os.path.join("Imagenes", "reloj.jpeg")

if not os.path.isfile(ruta_img):
    print(f"❌ No se encontró la imagen en: {ruta_img}")
    sys.exit()

# 2. Cargar imagen y redimensionar si es muy grande
img = Image.open(ruta_img).convert("RGBA")
width, height = img.size  # Obtiene las dimensiones de la imagen
if img.size[0] > MAX_SIZE[0] or img.size[1] > MAX_SIZE[1]:
    img.thumbnail(MAX_SIZE, Image.LANCZOS)
    width, height = img.size  # Actualiza las dimensiones si se redimensionó

pixels = np.array(img)

# --- INICIO: SEGMENTACIÓN DEL OBJETO CON K-MEANS ---

# Reformar la imagen para K-means: cada píxel es un punto en el espacio de color (RGB)
data = pixels[:, :, 0:3].reshape(-1, 3).astype(np.float64)

# Número de clusters (grupos de colores) que K-means intentará encontrar.
n_clusters = 8  # Ajustado para mayor granularidad de color
print(f"\n🧠 Realizando segmentación por colores (K-means) con {n_clusters} clusters... esto puede tomar un momento.")
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
kmeans.fit(data)
labels = kmeans.labels_  # Etiqueta de cluster para cada píxel
centers = kmeans.cluster_centers_  # Colores promedio de cada cluster (RGB)
print("🧠 Segmentación K-means completada.")

# --- IDENTIFICAR CLUSTERS DE FONDO Y OBJETO ---
# Heurística: asumimos que los píxeles de las esquinas pertenecen al fondo.
corner_pixels_colors = [
    pixels[0, 0, 0:3],  # Arriba-izquierda
    pixels[0, width - 1, 0:3],  # Arriba-derecha
    pixels[height - 1, 0, 0:3],  # Abajo-izquierda
    pixels[height - 1, width - 1, 0:3]  # Abajo-derecha
]

background_cluster_labels = set()
for color in corner_pixels_colors:
    background_cluster_labels.add(kmeans.predict([color.astype(np.float64)])[0])

# Identificar los clusters que no son de fondo (es decir, son del objeto)
all_cluster_ids = set(range(n_clusters))
object_cluster_labels = list(all_cluster_ids - background_cluster_labels)

# Crear la máscara de fondo/objeto (necesaria para el anti-aliasing y transparencia)
print("✨ Creando máscara de fondo/objeto para el anti-aliasing...")
mask_np = np.zeros((height, width), dtype=np.uint8)
is_background_pixels_raw = np.isin(labels, list(background_cluster_labels)).reshape(height, width)
mask_np[is_background_pixels_raw] = 255  # Pixeles de fondo en 255 en la máscara

# Aplicar operaciones morfológicas y suavizado a la máscara para limpiarla y suavizar bordes
kernel_open = np.ones((5, 5), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)
mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel_open, iterations=1)
mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close, iterations=1)
mask_np = cv2.GaussianBlur(mask_np, (5, 5), 0)  # Suavizar la máscara
print("✨ Máscara de fondo/objeto creada.")

# --- FIN: SEGMENTACIÓN DEL OBJETO CON K-MEANS ---


# --- INICIO: PREGUNTAS DE TRANSFORMACIÓN Y APLICACIÓN DE CAMBIOS ---

# Preguntar si se desea quitar el fondo
quitar_fondo_input = input("\n❓ ¿Deseas quitar el fondo (hacerlo transparente)? (sí/no): ").strip().lower()
quitar_fondo = quitar_fondo_input in ['si', 'sí', 's', 'yes', 'y']

# --- NUEVA PREGUNTA: PRESERVAR BLANCOS ---
preserve_white_input = input("\n⚪ ¿Deseas preservar los colores blancos dentro del objeto? (sí/no): ").strip().lower()
preserve_white = preserve_white_input in ['si', 'sí', 's', 'yes', 'y']
# Define el color blanco puro y la tolerancia para considerarlo "blanco"
white_rgb_fixed = np.array([255.0, 255.0, 255.0])
white_threshold = 30  # Ajusta este valor. Cuanto más bajo, más estricto con el blanco.

# Diccionario para guardar las reglas de cambio de color: cluster_id -> nuevo_color_rgb
recolor_map = {}

print("\n--- Colores detectados en el OBJETO principal (Clusters) ---")
if not object_cluster_labels:
    print(
        "⚠️ No se detectaron clusters de objeto distintivos. El objeto podría ser muy uniforme o no detectarse correctamente.")
    print("   El objeto mantendrá sus colores originales.")
else:
    for cluster_id in object_cluster_labels:
        center_rgb = centers[cluster_id]
        print(
            f"  Cluster {cluster_id}: RGB({int(center_rgb[0])},{int(center_rgb[1])},{int(center_rgb[2])}) - HEX({rgb_to_hex(center_rgb)})")

    # Pedir al usuario cuántos colores quiere cambiar
    num_cambios_input = input(
        f"\n🎨 ¿Cuántas partes/colores específicos del objeto quieres cambiar? (0 para no cambiar): ").strip()
    try:
        num_cambios = int(num_cambios_input)
    except ValueError:
        num_cambios = 0  # Si no es un número válido, no se hacen cambios

    if num_cambios > 0:
        clusters_a_procesar = list(object_cluster_labels)  # Crear una copia de los IDs de los clusters de objeto

        for i in range(min(num_cambios, len(clusters_a_procesar))):
            target_cluster_id = clusters_a_procesar[i]  # Tomar el siguiente cluster de la lista

            print(f"\n--- Configurando cambio {i + 1} de {num_cambios} ---")

            center_rgb_target = centers[target_cluster_id]
            print(f"  Cluster a cambiar: {target_cluster_id}")
            print(
                f"  Color actual: RGB({int(center_rgb_target[0])},{int(center_rgb_target[1])},{int(center_rgb_target[2])}) - HEX({rgb_to_hex(center_rgb_target)})")

            new_hex_color = input(
                f"  Ingrese el NUEVO color HEX para el Cluster {target_cluster_id} (o dejar vacío para no cambiar este cluster): ").strip()
            if new_hex_color:  # Si el usuario ingresó algo
                try:
                    new_rgb_color = np.array(hex_to_rgb(new_hex_color)).astype(np.float64)
                    recolor_map[target_cluster_id] = new_rgb_color
                    print(f"  ✅ Cluster {target_cluster_id} marcado para cambiar a {new_hex_color}.")
                except Exception as e:
                    print(f"  ❌ Error en el color HEX: {e}. Este cambio no se aplicará.")
            else:
                print(f"  ✅ Cluster {target_cluster_id} mantendrá su color original.")
    else:  # num_cambios es 0 o inválido
        print("   ✅ El objeto principal mantendrá sus colores originales.")

# Crear un nuevo array de píxeles para el resultado, trabajando en float64
result_pixels = np.copy(pixels).astype(np.float64)

# Procesar cada píxel de la imagen con barra de progreso
print("\n🔄 Aplicando transformaciones a la imagen...")
total_filas = height

for i in range(total_filas):
    porcentaje = int((i + 1) / total_filas * 100)
    print(f"\rProgreso: {porcentaje}% completado", end="")
    sys.stdout.flush()

    for j in range(width):
        r_orig, g_orig, b_orig, a_orig = pixels[i, j]
        pixel_rgb_orig = np.array([r_orig, g_orig, b_orig]).astype(np.float64)

        # Obtener el cluster al que pertenece este píxel
        pixel_cluster_id = labels[i * width + j]

        # El valor de la máscara para este píxel (0=objeto, 255=fondo)
        mask_value = mask_np[i, j]

        # Si el píxel ya era transparente en la imagen original, y no vamos a quitar el fondo, lo mantenemos así
        if a_orig == 0 and not quitar_fondo:
            continue

        # Calcular el factor de opacidad del objeto (0=fondo puro, 1=objeto puro) a partir del valor de la máscara
        object_opacity_factor = 1.0 - (mask_value / 255.0)
        object_opacity_factor = max(0.0, min(1.0, object_opacity_factor))  # Clampear entre 0 y 1

        # --- APLICACIÓN DE ALFA (TRANSPARENCIA) ---
        if quitar_fondo:
            # El alfa final se basa en el alfa original y el factor de opacidad del objeto
            result_pixels[i, j, 3] = a_orig * object_opacity_factor
        else:
            result_pixels[i, j, 3] = a_orig  # Si no se quita el fondo, mantener el alfa original

        # --- APLICACIÓN DE COLOR ---
        if object_opacity_factor > 0:  # Solo aplicar cambios si es parte del objeto (o transición)
            # --- NUEVA LÓGICA: PRESERVAR BLANCOS ---
            if preserve_white and np.linalg.norm(pixel_rgb_orig - white_rgb_fixed) < white_threshold:
                result_pixels[i, j, 0:3] = white_rgb_fixed  # Fuerza el píxel a blanco puro
            elif pixel_cluster_id in recolor_map:
                # Si el cluster de este píxel está en el mapa de colores a cambiar
                target_color = recolor_map[pixel_cluster_id]

                # Mezclar el nuevo color con el color original del píxel para el anti-aliasing y mantener textura
                result_pixels[i, j, 0] = (target_color[0] * object_opacity_factor) + (
                            pixel_rgb_orig[0] * (1 - object_opacity_factor))
                result_pixels[i, j, 1] = (target_color[1] * object_opacity_factor) + (
                            pixel_rgb_orig[1] * (1 - object_opacity_factor))
                result_pixels[i, j, 2] = (target_color[2] * object_opacity_factor) + (
                            pixel_rgb_orig[2] * (1 - object_opacity_factor))
            else:
                # Si el píxel es objeto/transición pero su cluster no está mapeado para cambiar, mantener su color original
                result_pixels[i, j, 0:3] = pixel_rgb_orig
                # Si object_opacity_factor es 0 (fondo puro), el color RGB se mantiene como el original del fondo.

# Convertir el array de píxeles modificado de float a uint8
result_pixels = result_pixels.astype(np.uint8)
final_pil_img = Image.fromarray(result_pixels, mode="RGBA")

# Opcional: Aplicar un filtro de suavizado final si aún hay aspereza en los bordes
# final_pil_img = final_pil_img.filter(ImageFilter.SMOOTH)
# final_pil_img = final_pil_img.filter(ImageFilter.BoxBlur(0.5))

# 4. Solicitar el nombre del archivo de salida y guardar la imagen
nombre_archivo = input(
    "\n💾 Ingrese un nombre para la imagen de salida (sin extensión, por defecto: imagen_transformada): ").strip()
if nombre_archivo == "":
    nombre_archivo = "imagen_transformada"

output_dir = "Imagenes"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{nombre_archivo}.png")

final_pil_img.save(output_path)
print(f"\n✅ Imagen procesada guardada en: {output_path}")
final_pil_img.show()