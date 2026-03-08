from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def ensure_output_dir(path="outputs"):
    """
    Crea la carpeta de salida si no existe.

    Parámetros:
        path (str): ruta de la carpeta de salida

    Retorna:
        Path: objeto Path de la carpeta creada/existente
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def generate_sample_image(size=(512, 512), seed=42):
    """
    Genera una imagen sintética en escala de grises para pruebas.

    Incluye:
    - gradientes
    - formas geométricas
    - ruido gaussiano
    - ruido sal y pimienta

    Parámetros:
        size (tuple): tamaño (alto, ancho)
        seed (int): semilla aleatoria

    Retorna:
        np.ndarray: imagen 2D uint8
    """
    rng = np.random.default_rng(seed)
    h, w = size

    # Gradiente horizontal
    gradient_x = np.tile(np.linspace(0, 255, w, dtype=np.float32), (h, 1))

    # Gradiente vertical
    gradient_y = np.tile(np.linspace(0, 255, h, dtype=np.float32)[:, None], (1, w))

    # Mezcla de gradientes
    base = 0.55 * gradient_x + 0.45 * gradient_y

    # Convertir a imagen PIL para dibujar
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(img)

    # Dibujar formas para que Sobel tenga bordes claros
    draw.rectangle((40, 40, 200, 180), outline=255, width=4)
    draw.ellipse((260, 80, 430, 250), outline=40, width=6)
    draw.line((50, 420, 460, 310), fill=210, width=5)
    draw.polygon([(300, 330), (420, 440), (250, 470)], outline=20, width=5)

    arr = np.array(img, dtype=np.float32)

    # Ruido gaussiano
    noise = rng.normal(0, 18, size=(h, w))

    # Ruido sal y pimienta
    sp = rng.choice([0, 255, -1], size=(h, w), p=[0.01, 0.01, 0.98])

    arr = arr + noise
    arr = np.where(sp == -1, arr, sp)

    # Limitar a rango válido
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def save_image(array, path):
    """
    Guarda una matriz 2D como imagen en escala de grises.

    Parámetros:
        array (np.ndarray): imagen 2D
        path (str | Path): ruta del archivo
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def load_grayscale_image(path):
    """
    Carga una imagen y la convierte a escala de grises.

    Parámetros:
        path (str | Path): ruta de la imagen

    Retorna:
        np.ndarray: imagen 2D uint8
    """
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)