from PIL import Image, ImageFilter
import numpy as np
import os


# Función para convertir HEX a RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("El código HEX debe tener 6 caracteres.")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


# Solicitar ruta de imagen al usuario
ruta_img = input("📂 Ingrese la ruta de la imagen (por defecto: Imagenes/reloj.jpeg): ").strip()
if ruta_img == "":
    ruta_img = os.path.join("Imagenes", "reloj.jpeg")

# Verificar si la imagen existe
if not os.path.isfile(ruta_img):
    print(f"❌ No se encontró la imagen en: {ruta_img}")
    exit()

# Cargar imagen con posible canal alfa
img = Image.open(ruta_img).convert("RGBA")
pixels = np.array(img)

# Solicitar color HEX al usuario
hex_color = input("🎨 Ingrese un color HEX para pintar el objeto sólido (ejemplo: #2471a3): ").strip()
try:
    nuevo_color = np.array(hex_to_rgb(hex_color))
except Exception as e:
    print(f"❌ Error en el color HEX: {e}")
    exit()

# Definir blanco y tolerancia
white = np.array([255, 255, 255])
# Puedes ajustar la tolerancia para considerar blanco cercano
# Un valor más bajo será más estricto con lo que considera "blanco"
# Un valor más alto podría incluir más del fondo.
tolerance = 95  # Aumenté la tolerancia ligeramente, puedes experimentar

# Crear nuevo arreglo de píxeles
new_pixels = np.copy(pixels)

# Recorrer cada píxel
for i in range(pixels.shape[0]):
    for j in range(pixels.shape[1]):
        r, g, b, a = pixels[i, j]

        # Si el píxel es transparente (alpha=0), mantenerlo
        if a == 0:
            continue

        # Si el píxel es blanco (dentro de tolerancia), mantenerlo
        # Aquí la condición se invierte para pintar lo que NO es blanco ni transparente
        if np.linalg.norm(np.array([r, g, b]) - white) < tolerance:
            continue

        # Sino, pintar con el nuevo color, manteniendo alpha original
        # Esto es crucial: mantenemos el canal alfa original para preservar cualquier anti-aliasing
        new_pixels[i, j, 0:3] = nuevo_color
        # new_pixels[i, j, 3] = a # Ya está implicito si no lo modificamos, pero lo dejo por claridad

# Crear la imagen desde el array de píxeles
new_img = Image.fromarray(new_pixels.astype(np.uint8), mode="RGBA")

# Opcional: Aplicar un filtro de suavizado
# Experimenta con diferentes filtros o intensidades
# new_img = new_img.filter(ImageFilter.SMOOTH)
# new_img = new_img.filter(ImageFilter.BLUR) # Puede ser demasiado fuerte

# Guardar imagen resultante en la carpeta Imagenes
output_dir = "Imagenes"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "imagen_transformada.png")  # PNG para conservar transparencia

new_img.save(output_path)

# Mostrar imagen
print(f"\n✅ Imagen procesada guardada en: {output_path}")
new_img.show()