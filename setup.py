from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'engineer.utils.libmesh.triangle_hash',
    sources=[
        'engineer/utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

ext_modules = [
    triangle_hash_module
]

setup(
ext_modules=cythonize(ext_modules),
cmdclass={
        'build_ext': BuildExtension
    },
include_dirs=[np.get_include()]
)