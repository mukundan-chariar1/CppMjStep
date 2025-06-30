from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# MuJoCo paths 
# TODO: Change these to your own mujoco paths. Refer install.md
MUJOCO_INCLUDE = '/home/mukundan/opt/mujoco/mujoco-3.3.1/include' 
MUJOCO_LIB = '/home/mukundan/opt/mujoco/mujoco-3.3.1/lib'

setup(
    name='torchmj',
    ext_modules=[
        CppExtension(
            name='torchmj',  # Python import name (import mjmod)
            sources=[
                'src/mjstep_module.cpp',
                'src/utils.cpp',
            ],
            include_dirs=[
                MUJOCO_INCLUDE,
            ],
            libraries=['mujoco'],
            library_dirs=[MUJOCO_LIB],
            extra_compile_args=['-O3', '-std=c++17'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
