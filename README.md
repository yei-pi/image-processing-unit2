# Image Processing - Unit 2

Proyecto de implementación y comparación de rendimiento de tres filtros de procesamiento de imágenes en escala de grises:

- Gaussian filter
- Sobel filter
- Median filter

Cada filtro fue desarrollado en tres enfoques:
1. Pure Python
2. NumPy
3. NumPy + Cython

## Estructura del proyecto

- `utils.py`: carga, guardado y generación de imagen de prueba
- `pure_python_filters.py`: implementación manual con bucles
- `numpy_filters.py`: implementación vectorizada con NumPy
- `cython_filters.pyx`: implementación acelerada con Cython
- `setup.py`: compilación de Cython
- `run_demo.py`: ejecución y comparación visual
- `benchmark.py`: medición de tiempos
- `outputs/`: resultados generados

## Instalación

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python setup.py build_ext --inplace