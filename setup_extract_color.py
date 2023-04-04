from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = [
    Extension(
        "extract_color_cython",
        ["extract_color_cython.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="Extract main colors",
    ext_modules=cythonize(ext_modules),
)
