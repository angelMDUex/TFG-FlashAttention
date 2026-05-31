from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

HEADER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cpp', 'headers'))

setup(
    name='custom_flash_attn',
    ext_modules=[
        CUDAExtension(
            name='custom_flash_attn',
            sources=[
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cpp', 'torch_bind.cpp')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cpp', 'v1.cu'))
            ],
            include_dirs=[HEADER_DIR],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
