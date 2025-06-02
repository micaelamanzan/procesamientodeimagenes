from PIL import Image, ImageFilter
import numpy as np
import os
import sys
from sklearn.cluster import KMeans  # Necesario para K-means
import cv2  # Necesario para operaciones morfológicas de OpenCV


# --- Funciones para manejar colores ANSI en la consola ---
def get_ansi_bg_color_code(r, g, b):
    return f"\x1b[48;2;{int(r)};{int(g)};{int(b)}m"


def get_ansi_fg_color_code(r, g, b):
    return f"\x1b[38;2;{int(r)};{int(g)};{int(b)}m"


def reset_ansi_code():
    return "\x1b[0m"


def print_colored_block(rgb_color, label_text, block_size=4):
    r, g, b = rgb_color
    bg_code = get_ansi_bg_color_code(r, g, b)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    fg_code = get_ansi_fg_color_code(0, 0, 0) if luminance > 0.5 else get_ansi_fg_color_code(255, 255, 255)
    formatted_label = f"{label_text:^{block_size}}"
    sys.stdout.write(f"{bg_code}{fg_code}{formatted_label}{reset_ansi_code()}")
    sys.stdout.flush()


# --- Fin de funciones ANSI ---


# Define el tamaño máximo para redimensionar imágenes grandes
MAX_SIZE = (800, 800)


# Función para convertir un color HEX a formato RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("El código HEX debe tener 6 caracteres.")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


# Función para convertir RGB a HEX (útil para mostrar colores al usuario)
def rgb_to_hex(rgb_color):
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
width, height = img.size
if img.size[0] > MAX_SIZE[0] or img.size[1] > MAX_SIZE[1]:
    img.thumbnail(MAX_SIZE, Image.LANCZOS)
    width, height = img.size

pixels = np.array(img)

# --- INICIO: SEGMENTACIÓN DEL OBJETO CON K-MEANS ---

# Reformar la imagen para K-means: cada píxel es un punto en el espacio de color (RGB)
data = pixels[:, :, 0:3].reshape(-1, 3).astype(np.float64)

# Número de clusters (grupos de colores) que K-means intentará encontrar.
n_clusters = 8
print(f"\n🧠 Realizando segmentación por colores (K-means) con {n_clusters} clusters... esto puede tomar un momento.")
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
kmeans.fit(data)
labels = kmeans.labels_  # Etiqueta de cluster para cada píxel
centers = kmeans.cluster_centers_  # Colores promedio de cada cluster (RGB)
print("🧠 Segmentación K-means completada.")

# --- IDENTIFICAR CLUSTERS DE FONDO Y OBJETO ---
initial_background_cluster_labels = set()
corner_pixels_colors = [
    pixels[0, 0, 0:3],  # Arriba-izquierda
    pixels[0, width - 1, 0:3],  # Arriba-derecha
    pixels[height - 1, 0, 0:3],  # Abajo-izquierda
    pixels[height - 1, width - 1, 0:3]  # Abajo-derecha
]
for color in corner_pixels_colors:
    initial_background_cluster_labels.add(kmeans.predict([color.astype(np.float64)])[0])

# --- INTERACCIÓN MEJORADA: CONFIRMAR/CORREGIR CLUSTERS DE FONDO ---
print("\n--- TODOS los Clusters de Color Detectados por K-means (por ID) ---")
print("  ")
for i, center_rgb in enumerate(centers):
    print_colored_block(center_rgb, str(i))
print(f" {reset_ansi_code()}\n")

print(f"Predicción inicial de clusters de fondo (basada en esquinas): {list(initial_background_cluster_labels)}")
confirm_bg_input = input(
    "🔍 Confirma los IDs de los clusters que son PARTE del FONDO (separados por espacios, ej. '0 5 7'). Deja vacío para aceptar la predicción inicial: ").strip()

background_cluster_labels = set()
if confirm_bg_input:
    try:
        user_bg_clusters = set(int(x) for x in confirm_bg_input.split())
        if all(0 <= c < n_clusters for c in user_bg_clusters):
            background_cluster_labels = user_bg_clusters
            print(f"✅ Clusters de fondo actualizados a: {list(background_cluster_labels)}")
        else:
            print("❌ IDs de clusters inválidos. Usando la predicción inicial de fondo.")
            background_cluster_labels = initial_background_cluster_labels
    except ValueError:
        print("❌ Entrada inválida. Usando la predicción inicial de fondo.")
        background_cluster_labels = initial_background_cluster_labels
else:
    print("✅ Confirmada la predicción inicial de fondo (no se ingresaron cambios).")
    background_cluster_labels = initial_background_cluster_labels

# Identificar los clusters que no son de fondo (es decir, son del objeto)
all_cluster_ids = set(range(n_clusters))
object_cluster_labels = list(all_cluster_ids - background_cluster_labels)

# Crear la máscara de fondo/objeto (necesaria para el anti-aliasing y transparencia)
print("✨ Creando máscara de fondo/objeto para el anti-aliasing...")
mask_np = np.zeros((height, width), dtype=np.uint8)
is_background_pixels_raw = np.isin(labels, list(background_cluster_labels)).reshape(height, width)
mask_np[is_background_pixels_raw] = 255

# Aplicar operaciones morfológicas y suavizado a la máscara para limpiarla y suavizar bordes
kernel_open = np.ones((5, 5), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)
mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel_open, iterations=1)
mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close, iterations=1)
mask_np = cv2.GaussianBlur(mask_np, (5, 5), 0)
print("✨ Máscara de fondo/objeto creada.")

# --- FIN: SEGMENTACIÓN DEL OBJETO CON K-MEANS ---


# --- INICIO: PREGUNTAS DE TRANSFORMACIÓN Y APLICACIÓN DE CAMBIOS ---

# Preguntar si se desea quitar el fondo
quitar_fondo_input = input("\n❓ ¿Deseas quitar el fondo (hacerlo transparente)? (sí/no): ").strip().lower()
quitar_fondo = quitar_fondo_input in ['si', 'sí', 's', 'yes', 'y']

# --- PRESERVAR BLANCOS (opcional) ---
preserve_white_input = input("\n⚪ ¿Deseas preservar los colores blancos dentro del objeto? (sí/no): ").strip().lower()
preserve_white = preserve_white_input in ['si', 'sí', 's', 'yes', 'y']
white_rgb_fixed = np.array([255.0, 255.0, 255.0])
white_threshold = 30

# Diccionario para guardar las reglas de cambio de color: cluster_id -> nuevo_color_rgb
recolor_map = {}

# Interacción para cambiar colores del objeto
if not object_cluster_labels:
    print(
        "⚠️ No se detectaron clusters de objeto distintivos. El objeto podría ser muy uniforme o no detectarse correctamente.")
    print("   El objeto mantendrá sus colores originales.")
else:
    print("\n--- Colores detectados en el OBJETO principal (Clusters) ---")
    print("  ")
    for cluster_id in object_cluster_labels:
        center_rgb = centers[cluster_id]
        print_colored_block(center_rgb, str(cluster_id))
    print(f" {reset_ansi_code()}\n")

    num_cambios_input = input(
        f"\n🎨 ¿Cuántas partes/colores específicos del objeto quieres cambiar? (0 para no cambiar): ").strip()
    try:
        num_cambios = int(num_cambios_input)
    except ValueError:
        num_cambios = 0

    if num_cambios > 0:
        cambios_realizados = 0
        while cambios_realizados < num_cambios:
            print(f"\n--- Configurando cambio {cambios_realizados + 1} de {num_cambios} ---")

            # --- CAMBIO CLAVE AQUÍ: Pedir el ID directamente ---
            # El usuario ingresa el ID del cluster que ve en la lista de bloques de color
            target_cluster_id_input = input("  Ingrese el ID del Cluster que desea cambiar: ").strip()

            try:
                target_cluster_id = int(target_cluster_id_input)

                # Validación: el ID debe ser válido y debe ser un cluster del objeto
                if not (0 <= target_cluster_id < n_clusters):
                    print(f"   ❌ ID de Cluster inválido. Debe estar entre 0 y {n_clusters - 1}.")
                    continue  # Volver a pedir el ID para este mismo slot de cambio

                if target_cluster_id not in object_cluster_labels:
                    print(
                        f"   ❌ El Cluster {target_cluster_id} no es parte del objeto principal (o ya ha sido clasificado como fondo).")
                    continue  # Volver a pedir el ID para este mismo slot de cambio

                # Advertir si ya se ha configurado este cluster en la misma sesión
                if target_cluster_id in recolor_map:
                    print(
                        f"   ⚠️ ADVERTENCIA: El Cluster {target_cluster_id} ya ha sido configurado para cambio. Se sobrescribirá.")

                # Mostrar el color actual del cluster seleccionado (visual y HEX)
                center_rgb_target = centers[target_cluster_id]
                print(f"  Cluster a cambiar: {target_cluster_id}. Color actual: ", end="")
                print_colored_block(center_rgb_target, " ")
                print(f" {reset_ansi_code()} HEX: {rgb_to_hex(center_rgb_target)}")

                new_hex_color = input(
                    f"  Ingrese el NUEVO color HEX para el Cluster {target_cluster_id} (o dejar vacío para no cambiar este cluster): ").strip()
                if new_hex_color:
                    try:
                        new_rgb_color = np.array(hex_to_rgb(new_hex_color)).astype(np.float64)
                        recolor_map[target_cluster_id] = new_rgb_color
                        print(f"  ✅ Cluster {target_cluster_id} marcado para cambiar a {new_hex_color}.")
                    except Exception as e:
                        print(f"   ❌ Error en el color HEX: {e}. Este cambio no se aplicará.")
                        continue
                else:
                    if target_cluster_id in recolor_map:
                        del recolor_map[target_cluster_id]
                    print(f"  ✅ Cluster {target_cluster_id} mantendrá su color original.")

                cambios_realizados += 1
            except ValueError:
                print("   ❌ Entrada inválida. Ingrese un número para el ID del Cluster.")

    else:
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

        pixel_cluster_id = labels[i * width + j]
        mask_value = mask_np[i, j]

        if a_orig == 0 and not quitar_fondo:
            continue

        object_opacity_factor = 1.0 - (mask_value / 255.0)
        object_opacity_factor = max(0.0, min(1.0, object_opacity_factor))

        if quitar_fondo:
            result_pixels[i, j, 3] = a_orig * object_opacity_factor
        else:
            result_pixels[i, j, 3] = a_orig

        if object_opacity_factor > 0:
            if preserve_white and np.linalg.norm(pixel_rgb_orig - white_rgb_fixed) < white_threshold:
                result_pixels[i, j, 0:3] = white_rgb_fixed
            elif pixel_cluster_id in recolor_map:
                target_color = recolor_map[pixel_cluster_id]

                result_pixels[i, j, 0] = (target_color[0] * object_opacity_factor) + (
                            pixel_rgb_orig[0] * (1 - object_opacity_factor))
                result_pixels[i, j, 1] = (target_color[1] * object_opacity_factor) + (
                            pixel_rgb_orig[1] * (1 - object_opacity_factor))
                result_pixels[i, j, 2] = (target_color[2] * object_opacity_factor) + (
                            pixel_rgb_orig[2] * (1 - object_opacity_factor))
            else:
                result_pixels[i, j, 0:3] = pixel_rgb_orig

            # Convertir el array de píxeles modificado de float a uint8
result_pixels = result_pixels.astype(np.uint8)
final_pil_img = Image.fromarray(result_pixels, mode="RGBA")

# Opcional: Aplicar un filtro de suavizado final
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
final_pil_img.show()C:\Users\Cecilia\Desktop\casas.jpg