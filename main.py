from PIL import Image
import numpy as np
import os

# Cargar imagen original
img_path = os.path.join("Imagenes", "reloj.jpeg")
img = Image.open(img_path).convert("RGB")
pixels = np.array(img)

# Paleta personalizada
palette = np.array([
    [95, 85, 90],
    [245, 200, 100],
    [30, 120, 200],
    [210, 100, 129],
    [255, 20, 225]
])

# Color blanco para tolerancia (opcional)
white = np.array([255, 255, 255])
tolerance = 0

# Función para encontrar el color más cercano
def closest_color(color, palette):
    distances = np.sqrt(np.sum((palette - color) ** 2, axis=1))
    return palette[np.argmin(distances)]

# Generar nueva imagen con la paleta
new_pixels = np.zeros_like(pixels)
for i in range(pixels.shape[0]):
    for j in range(pixels.shape[1]):
        pixel = pixels[i, j]
        if np.linalg.norm(pixel - white) < tolerance:
            new_pixels[i, j] = white
        else:
            new_pixels[i, j] = closest_color(pixel, palette)

# Guardar imagen resultante
new_img = Image.fromarray(new_pixels.astype(np.uint8))
new_img.save("imagen_transformada.jpg")
new_img.show()