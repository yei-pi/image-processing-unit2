from pathlib import Path
import importlib
import numpy as np
import matplotlib.pyplot as plt

from utils import generate_sample_image, load_grayscale_image, save_image, ensure_output_dir
from pure_python_filters import (
    gaussian_filter_python,
    sobel_filter_python,
    median_filter_python,
)
from numpy_filters import (
    gaussian_filter_numpy,
    sobel_filter_numpy,
    median_filter_numpy,
)


def try_import_cython():
    """
    Intenta importar el módulo compilado de Cython.
    Si no existe, devuelve None.
    """
    try:
        return importlib.import_module("cython_filters")
    except Exception:
        return None


def main():
    """
    Ejecuta la demo completa:
    - genera o carga imagen
    - aplica filtros Pure Python y NumPy
    - intenta Cython si está compilado
    - guarda imágenes en outputs/
    """
    out_dir = ensure_output_dir("outputs")

    # Si quieres cargar una imagen externa, reemplaza esta línea:
    # image = load_grayscale_image("tu_imagen.png")
    image = generate_sample_image()

    image_list = image.tolist()
    cython_mod = try_import_cython()

    print("Aplicando filtros Pure Python...")
    gaussian_py = np.array(gaussian_filter_python(image_list), dtype=np.uint8)
    sobel_py = np.array(sobel_filter_python(image_list), dtype=np.uint8)
    median_py = np.array(median_filter_python(image_list), dtype=np.uint8)

    print("Aplicando filtros NumPy...")
    gaussian_np = gaussian_filter_numpy(image)
    sobel_np = sobel_filter_numpy(image)
    median_np = median_filter_numpy(image)

    results = [
        ("Original", image),
        ("Gaussian - Python", gaussian_py),
        ("Gaussian - NumPy", gaussian_np),
        ("Sobel - Python", sobel_py),
        ("Sobel - NumPy", sobel_np),
        ("Median - Python", median_py),
        ("Median - NumPy", median_np),
    ]

    if cython_mod is not None:
        print("Aplicando filtros Cython...")
        gaussian_cy = cython_mod.gaussian_filter_cython(image)
        sobel_cy = cython_mod.sobel_filter_cython(image)
        median_cy = cython_mod.median_filter_cython(image)

        results.extend([
            ("Gaussian - Cython", gaussian_cy),
            ("Sobel - Cython", sobel_cy),
            ("Median - Cython", median_cy),
        ])

        save_image(gaussian_cy, out_dir / "gaussian_cython.png")
        save_image(sobel_cy, out_dir / "sobel_cython.png")
        save_image(median_cy, out_dir / "median_cython.png")

    print("Guardando imágenes...")

    save_image(image, out_dir / "input.png")

    save_image(gaussian_py, out_dir / "gaussian_python.png")
    save_image(sobel_py, out_dir / "sobel_python.png")
    save_image(median_py, out_dir / "median_python.png")

    save_image(gaussian_np, out_dir / "gaussian_numpy.png")
    save_image(sobel_np, out_dir / "sobel_numpy.png")
    save_image(median_np, out_dir / "median_numpy.png")

    # Crear cuadrícula comparativa
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, arr) in zip(axes, results):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for ax in axes[len(results):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "comparison_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Demo terminada. Revisa la carpeta: {Path(out_dir).resolve()}")


if __name__ == "__main__":
    main()