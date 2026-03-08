from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="cython_filters",
        sources=["cython_filters.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="image-processing-unit2",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"}
    ),
)