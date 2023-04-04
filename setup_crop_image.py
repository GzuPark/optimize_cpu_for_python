import os
from distutils.core import setup, Extension

import numpy as np


opencv_include_dir = "/opt/homebrew/include/opencv4"  # for macos
compiler_path = "/opt/homebrew/bin/g++-12"  # for macos (GCC or LLVM)

os.environ["CC"] = compiler_path
os.environ["CXX"] = compiler_path


crop_module_cpp = Extension(
    "image_crop_module",
    sources=["image_crop_module.cpp"],
    include_dirs=[np.get_include(), opencv_include_dir],
    libraries=["opencv_core", "opencv_imgproc"],
    library_dirs=["/opt/homebrew/lib"],
    extra_compile_args=["-std=c++17"],
)

crop_module_omp = Extension(
    "image_crop_module_omp",
    sources=["image_crop_module_omp.cpp"],
    include_dirs=[np.get_include(), opencv_include_dir],
    libraries=["opencv_core", "opencv_imgproc"],
    library_dirs=["/opt/homebrew/lib"],
    extra_compile_args=["-fopenmp", "-std=c++17"],
    extra_link_args=["-lgomp"],
)

setup(
    name="Optimize CPU modules",
    version="0.1",
    ext_modules=[crop_module_cpp, crop_module_omp],
)
