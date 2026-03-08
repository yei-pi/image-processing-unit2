import csv
import statistics
import time
import importlib
from pathlib import Path

import numpy as np

from utils import generate_sample_image, save_image, ensure_output_dir
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


def time_callable(fn, repeats=5):
    """
    Mide una función varias veces y devuelve:
    - promedio
    - desviación estándar
    """
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append(end - start)

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_time, std_time


def try_import_cython():
    """
    Intenta importar el módulo compilado de Cython.
    """
    try:
        return importlib.import_module("cython_filters"), None
    except Exception as e:
        return None, str(e)


def main():
    out_dir = ensure_output_dir("outputs")

    image = generate_sample_image()
    image_list = image.tolist()

    cython_mod, cython_error = try_import_cython()

    methods = [
        ("Gaussian", "Pure Python", lambda: gaussian_filter_python(image_list)),
        ("Gaussian", "NumPy", lambda: gaussian_filter_numpy(image)),
        ("Sobel", "Pure Python", lambda: sobel_filter_python(image_list)),
        ("Sobel", "NumPy", lambda: sobel_filter_numpy(image)),
        ("Median", "Pure Python", lambda: median_filter_python(image_list)),
        ("Median", "NumPy", lambda: median_filter_numpy(image)),
    ]

    if cython_mod is not None:
        methods.extend([
            ("Gaussian", "NumPy + Cython", lambda: cython_mod.gaussian_filter_cython(image)),
            ("Sobel", "NumPy + Cython", lambda: cython_mod.sobel_filter_cython(image)),
            ("Median", "NumPy + Cython", lambda: cython_mod.median_filter_cython(image)),
        ])

    rows = []

    for filter_name, impl_name, fn in methods:
        mean_t, std_t = time_callable(fn, repeats=5)
        rows.append({
            "filter": filter_name,
            "implementation": impl_name,
            "mean_seconds": mean_t,
            "std_seconds": std_t,
        })

    if cython_mod is None:
        rows.extend([
            {"filter": "Gaussian", "implementation": "NumPy + Cython", "mean_seconds": "", "std_seconds": ""},
            {"filter": "Sobel", "implementation": "NumPy + Cython", "mean_seconds": "", "std_seconds": ""},
            {"filter": "Median", "implementation": "NumPy + Cython", "mean_seconds": "", "std_seconds": ""},
        ])

    csv_path = out_dir / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filter", "implementation", "mean_seconds", "std_seconds"]
        )
        writer.writeheader()
        writer.writerows(rows)

    if cython_error:
        note_path = out_dir / "cython_build_note.txt"
        note_path.write_text(
            "No se pudo importar la extensión de Cython.\n"
            f"Motivo: {cython_error}\n"
            "Compila primero con: python setup.py build_ext --inplace\n",
            encoding="utf-8"
        )

    print(f"Benchmark terminado. Archivo generado: {csv_path.resolve()}")


if __name__ == "__main__":
    main()